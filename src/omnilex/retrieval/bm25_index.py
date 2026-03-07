"""Memory-efficient BM25 indexing using Scipy Sparse Matrices."""

import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from typing import List, Dict, Any, Optional


class BM25Index:
    """Memory-efficient BM25 index for large-scale legal corpora."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        citation_field: str = "citation",
    ):
        self.k1 = k1
        self.b = b
        self.citation_field = citation_field

        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
        self.citations: List[str] = []
        self.tf_matrix: Optional[sp.csr_matrix] = None
        self.doc_lens: Optional[np.ndarray] = None
        self.idf: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

    def build(self, texts: List[str], citations: List[str]) -> None:
        """Build index using sparse matrices."""
        print(f"  Tokenizing and building sparse matrix for {len(texts)} docs...")
        self.citations = citations

        # 1. Fit and transform texts to count matrix
        self.tf_matrix = self.vectorizer.fit_transform(texts)

        # 2. Precompute stats
        self.doc_lens = np.array(self.tf_matrix.sum(axis=1)).flatten()
        self.avgdl = self.doc_lens.mean()

        # 3. Compute IDF
        n_docs = self.tf_matrix.shape[0]
        # Number of docs containing each term (column-wise non-zero counts)
        doc_freqs = np.array((self.tf_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((n_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0)

        print(f"  Index built. Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def search(
        self, query: str, top_k: int = 10, return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Fast vectorized BM25 search."""
        if self.tf_matrix is None:
            raise ValueError("Index not built.")

        query_vec = self.vectorizer.transform([query])
        if query_vec.nnz == 0:
            return []

        # Get relevant columns from tf_matrix
        q_indices = query_vec.indices
        tf = self.tf_matrix[:, q_indices].toarray()

        # Apply BM25 formula to relevant terms
        # score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        num = tf * (self.k1 + 1)
        denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[:, None] / self.avgdl)

        # Weight by IDF and sum across terms
        scores = (self.idf[q_indices] * (num / denom)).sum(axis=1)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            res = {"citation": self.citations[idx]}
            if return_scores:
                res["_score"] = float(scores[idx])
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
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(citation_field=data["citation_field"])
        instance.citations = data["citations"]
        instance.tf_matrix = data["tf_matrix"]
        instance.doc_lens = data["doc_lens"]
        instance.avgdl = data["avgdl"]
        instance.idf = data["idf"]
        instance.vectorizer = data["vectorizer"]
        return instance
