"""Hybrid search engine combining BM25 and Dense search using Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking."""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union


def build_text_lookup(laws_path: str, courts_path: str) -> Dict[str, str]:
    """
    LAZY LOADING O(1): Builds a mapping from citation to document text.
    Maintains only the IDs and texts in RAM (~2GB for full corpus).
    """
    lookups = {}

    if os.path.exists(laws_path):
        print(f"Building text lookup from {laws_path}...")
        df_laws = pd.read_csv(laws_path, usecols=["citation", "text"])
        lookups.update(df_laws.set_index("citation")["text"].to_dict())
        del df_laws

    if os.path.exists(courts_path):
        print(f"Building text lookup from {courts_path}...")
        # Court considerations is very large (2.4GB), we use chunks if needed,
        # but here we follow the req_10 suggestion.
        df_courts = pd.read_csv(courts_path, usecols=["citation", "text"])
        lookups.update(df_courts.set_index("citation")["text"].to_dict())
        del df_courts

    return lookups


class HybridSearchEngine:
    """Hybrid search engine with RRF and Cross-Encoder Reranking."""

    def __init__(
        self,
        bm25_index,
        dense_index,
        reranker=None,
        text_lookup: Optional[Dict[str, str]] = None,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search engine.

        Args:
            bm25_index: Instance of BM25Index
            dense_index: Instance of DenseIndex
            reranker: Instance of sentence_transformers.CrossEncoder
            text_lookup: Dictionary mapping citation -> text for reranking
            rrf_k: Stability parameter for RRF (default 60)
        """
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.reranker = reranker
        self.text_lookup = text_lookup or {}
        self.rrf_k = rrf_k

    def query(
        self,
        text: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        top_k_rerank: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search, fuse results with RRF, and optionally rerank.

        Returns:
            List of dictionaries: [{'citation': ..., 'score': ...}, ...]
        """
        # 1. Fetch candidates from both indices (Parallel Stage)
        bm25_results = self.bm25_index.search(text, top_k=bm25_top_k)
        dense_results = self.dense_index.search(text, top_k=dense_top_k)

        # 2. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Process BM25 rankings
        for rank, res in enumerate(bm25_results):
            citation = res.get("citation")
            if not citation:
                continue
            rrf_scores[citation] = rrf_scores.get(citation, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Process Dense rankings
        for rank, res in enumerate(dense_results):
            citation = res.get("citation")
            if not citation:
                continue
            rrf_scores[citation] = rrf_scores.get(citation, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort by RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # If no reranker or no candidates, return RRF results normalized
        if not self.reranker or not sorted_rrf:
            return self._normalize_results(sorted_rrf[:top_k])

        # 3. Second-Stage Retrieval (Cross-Encoder Reranking)
        candidates_to_rerank = sorted_rrf[:top_k_rerank]

        cross_inp = []
        valid_citations = []

        for citation, _ in candidates_to_rerank:
            # Lazy Loading check
            doc_text = self.text_lookup.get(citation)
            if doc_text:
                cross_inp.append([text, doc_text])
                valid_citations.append(citation)

        if not cross_inp:
            return self._normalize_results(sorted_rrf[:top_k])

        # Prediction (Logits)
        logits = self.reranker.predict(cross_inp)

        # Sigmoid conversion for strict [0.0, 1.0] probabilities
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        # Final Rank by probability
        final_ranked = sorted(zip(valid_citations, probabilities), key=lambda x: x[1], reverse=True)

        # Limit to top_k
        top_final = final_ranked[:top_k]

        return [{"citation": cit, "score": float(prob)} for cit, prob in top_final]

    def _normalize_results(self, results: List[tuple]) -> List[Dict[str, Any]]:
        """Min-Max Normalization for RRF scores."""
        if not results:
            return []

        scores = np.array([res[1] for res in results])
        if len(scores) > 1:
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                norm_scores = (scores - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones_like(scores)
        else:
            norm_scores = np.ones_like(scores)

        return [{"citation": cit, "score": float(s)} for (cit, _), s in zip(results, norm_scores)]
