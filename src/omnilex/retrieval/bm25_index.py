"""Memory-efficient BM25 indexing using Scipy Sparse Matrices."""

import pickle
import json
import os
import gc
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class BM25Index:
    """Memory-efficient BM25 index for large-scale legal corpora."""

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        text_field: str = "text",
        citation_field: str = "citation",
    ):
        self.k1 = k1
        self.b = b
        self.text_field = text_field
        self.citation_field = citation_field

        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
        self.citations: List[str] = []
        self.tf_matrix: Optional[sp.csr_matrix] = None
        self.doc_lens: Optional[np.ndarray] = None
        self.idf: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

        if documents:
            self.build_from_docs(documents)

    def build_from_docs(self, documents: List[Dict[str, Any]]) -> None:
        """Compatibility method to build from list of dicts."""
        texts = [doc.get(self.text_field, "") for doc in documents]
        citations = [doc.get(self.citation_field, "Unknown") for doc in documents]
        self.build(texts, citations)

    def build(self, texts: List[str], citations: List[str]) -> None:
        """Build index using sparse matrices."""
        print(f"  Tokenizing and building sparse matrix for {len(texts)} docs...")
        self.citations = citations

        # Fit and transform
        self.tf_matrix = self.vectorizer.fit_transform(texts)

        # Precompute stats
        self.doc_lens = np.array(self.tf_matrix.sum(axis=1)).flatten()
        self.avgdl = self.doc_lens.mean()

        # Compute IDF
        n_docs = self.tf_matrix.shape[0]
        doc_freqs = np.array((self.tf_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((n_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0)

        print(f"  Index built. Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def search(
        self, query: str, top_k: int = 10, return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        ULTRA-MEMORY-EFFICIENT vectorized BM25 search.
        Processes term-by-term to avoid O(D*T) memory spikes.
        """
        if self.tf_matrix is None:
            raise ValueError("Index not built. Call .load() or .build() first.")

        query_vec = self.vectorizer.transform([query])
        if query_vec.nnz == 0:
            return []

        q_indices = query_vec.indices

        # Accumulator for scores (size: n_docs)
        # 2.65M docs * 8 bytes = ~21 MB (Safe)
        total_scores = np.zeros(self.tf_matrix.shape[0], dtype=np.float32)

        # Cache constant part of denominator
        denom_base = self.k1 * (1 - self.b + self.b * self.doc_lens / self.avgdl)

        # Process each term independently to save RAM
        for term_idx in q_indices:
            # Extract single column (still sparse)
            tf_col = self.tf_matrix[:, term_idx].toarray().flatten()

            # Apply BM25 formula for this term
            # score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            term_score = self.idf[term_idx] * (tf_col * (self.k1 + 1)) / (tf_col + denom_base)

            # Accumulate
            total_scores += term_score

            # Explicit cleanup
            del tf_col, term_score

        # Get top-k
        top_indices = np.argsort(total_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if total_scores[idx] <= 0:
                continue
            res = {"citation": self.citations[idx]}
            if return_scores:
                res["_score"] = float(total_scores[idx])
            results.append(res)

        return results

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "citations": self.citations,
                    "tf_matrix": self.tf_matrix,
                    "doc_lens": self.doc_lens,
                    "avgdl": self.avgdl,
                    "idf": self.idf,
                    "vectorizer": self.vectorizer,
                    "citation_field": self.citation_field,
                    "text_field": self.text_field,
                },
                f,
            )

    def load(self, path: Union[str, Path]):
        path = str(path)
        if os.path.isdir(path):
            path = os.path.join(path, "corpus_bm25.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"[X] Arquivo de índice não encontrado: {path}")

        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        self.__dict__.update(state_dict)

        if not hasattr(self, "tf_matrix") or self.tf_matrix is None:
            if "matrix" in state_dict:
                self.tf_matrix = state_dict["matrix"]
            else:
                raise KeyError(f"[X] Erro de desserialização: 'tf_matrix' ausente.")

        # Cast to float32 to save RAM if it was float64
        self.doc_lens = self.doc_lens.astype(np.float32)
        self.idf = self.idf.astype(np.float32)

        print(f"[V] BM25 Index montado! Shape: {self.tf_matrix.shape}")

    @classmethod
    def load_from_path(cls, path: Path | str) -> "BM25Index":
        instance = cls()
        instance.load(path)
        return instance


# --- Compatibility utility functions ---


def build_index(
    documents: List[Dict[str, Any]], text_field: str = "text", citation_field: str = "citation"
) -> BM25Index:
    return BM25Index(documents=documents, text_field=text_field, citation_field=citation_field)


def search(index: BM25Index, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    return index.search(query, top_k=top_k)


def load_jsonl_corpus(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def save_jsonl_corpus(documents: List[Dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
