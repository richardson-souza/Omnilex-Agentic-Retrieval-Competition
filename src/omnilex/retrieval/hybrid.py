"""Hybrid search engine combining BM25 and Dense search using Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking."""

import os
import sqlite3
import gc
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union


class SQLiteTextLookup:
    """Memory-efficient disk-based lookup for large-scale document texts."""

    def __init__(self, db_path: str = "data/processed/corpus_text.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS documents (citation TEXT, text TEXT)")

    def insert_chunk(self, df: pd.DataFrame):
        """Insert a dataframe chunk into SQLite."""
        df.to_sql("documents", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def create_index(self):
        """Build index for O(1) retrieval and cleanup duplicates."""
        print("  Cleaning up duplicate citations from disk (Final Stage)...")
        # Keep only the first rowid for each citation
        self.conn.execute(
            "DELETE FROM documents WHERE rowid NOT IN (SELECT min(rowid) FROM documents GROUP BY citation)"
        )
        self.conn.commit()

        print("  Building UNIQUE SQLite index on citation...")
        # Making it UNIQUE ensures maximum B-Tree search speed
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_citation ON documents (citation)")
        self.conn.commit()

        # Shrink file size
        print("  Vacuuming database...")
        self.conn.execute("VACUUM")
        self.conn.commit()

    def get(self, citation: str) -> Optional[str]:
        """Fetch text for a single citation from disk."""
        cursor = self.conn.execute(
            "SELECT text FROM documents WHERE citation = ? LIMIT 1", (citation,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()


def build_text_lookup(
    laws_path: str, courts_path: str, db_path: str = "data/processed/corpus_text.db"
) -> SQLiteTextLookup:
    """
    ULTRA-MEMORY-EFFICIENT LOOKUP: Creates a SQLite database from CSVs.
    Uses chunking to keep RAM usage near zero during build.
    """
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}...")
        os.remove(db_path)

    lookup = SQLiteTextLookup(db_path)

    for path in [laws_path, courts_path]:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found. Skipping...")
            continue

        print(f"Streaming {path} to SQLite...")
        for chunk in pd.read_csv(path, usecols=["citation", "text"], chunksize=100000):
            # Pre-deduplicate inside chunk to save disk I/O
            chunk = chunk.drop_duplicates(subset=["citation"], keep="first")
            lookup.insert_chunk(chunk)
            del chunk
            gc.collect()

    lookup.create_index()
    gc.collect()
    return lookup


class HybridSearchEngine:
    """Hybrid search engine with RRF and Cross-Encoder Reranking."""

    def __init__(
        self,
        bm25_index,
        dense_index,
        reranker=None,
        text_lookup: Optional[Union[pd.DataFrame, SQLiteTextLookup]] = None,
        rrf_k: int = 60,
    ):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.reranker = reranker
        self.text_lookup = text_lookup
        self.rrf_k = rrf_k

    def query(
        self,
        text: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        top_k_rerank: int = 50,
    ) -> List[Dict[str, Any]]:
        start_time = time.time()

        # 1. First-Stage Retrieval (BM25 + FAISS)
        bm25_results = self.bm25_index.search(text, top_k=bm25_top_k)
        dense_results = self.dense_index.search(text, top_k=dense_top_k)

        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, res in enumerate(bm25_results):
            citation = res.get("citation")
            if citation:
                rrf_scores[citation] = rrf_scores.get(citation, 0.0) + (
                    1.0 / (self.rrf_k + rank + 1)
                )

        for rank, res in enumerate(dense_results):
            citation = res.get("citation")
            if citation:
                rrf_scores[citation] = rrf_scores.get(citation, 0.0) + (
                    1.0 / (self.rrf_k + rank + 1)
                )

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Cleanup first stage
        del rrf_scores, bm25_results, dense_results

        if not self.reranker or not self.text_lookup or not sorted_rrf:
            return self._normalize_results(sorted_rrf[:top_k])

        # 3. Second-Stage Retrieval (Cross-Encoder Reranking)
        candidates_to_rerank = sorted_rrf[:top_k_rerank]
        cross_inp = []
        valid_citations = []

        for citation, _ in candidates_to_rerank:
            if isinstance(self.text_lookup, pd.DataFrame):
                try:
                    doc_text = self.text_lookup.at[citation, "text"]
                    if isinstance(doc_text, pd.Series):
                        doc_text = doc_text.iloc[0]
                except KeyError:
                    doc_text = None
            else:
                doc_text = self.text_lookup.get(citation)

            if doc_text and pd.notna(doc_text):
                cross_inp.append([text, str(doc_text)])
                valid_citations.append(citation)

        if not cross_inp:
            return self._normalize_results(sorted_rrf[:top_k])

        # Batch prediction with explicit batch_size
        import torch

        with torch.inference_mode():
            logits = self.reranker.predict(cross_inp, batch_size=16, show_progress_bar=False)

        probabilities = 1.0 / (1.0 + np.exp(-logits))

        final_ranked = sorted(zip(valid_citations, probabilities), key=lambda x: x[1], reverse=True)

        # Cleanup
        del cross_inp, logits, candidates_to_rerank

        # Optional Telemetry
        # duration = time.time() - start_time
        # if np.random.random() < 0.01: print(f"Query latency: {duration:.2f}s")

        return [{"citation": cit, "score": float(prob)} for cit, prob in final_ranked[:top_k]]

    def _normalize_results(self, results: List[tuple]) -> List[Dict[str, Any]]:
        if not results:
            return []
        scores = np.array([res[1] for res in results])
        if len(scores) > 1:
            min_s, max_s = scores.min(), scores.max()
            norm_scores = (
                (scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(scores)
            )
        else:
            norm_scores = np.ones_like(scores)
        return [{"citation": cit, "score": float(s)} for (cit, _), s in zip(results, norm_scores)]
