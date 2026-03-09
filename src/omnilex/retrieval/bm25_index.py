"""Memory-efficient BM25 indexing using Scipy Sparse Matrices."""

import pickle
import json
import os
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

        # 1. Fit and transform texts to count matrix
        self.tf_matrix = self.vectorizer.fit_transform(texts)

        # 2. Precompute stats
        self.doc_lens = np.array(self.tf_matrix.sum(axis=1)).flatten()
        self.avgdl = self.doc_lens.mean()

        # 3. Compute IDF
        n_docs = self.tf_matrix.shape[0]
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

        q_indices = query_vec.indices
        tf = self.tf_matrix[:, q_indices].toarray()

        num = tf * (self.k1 + 1)
        denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[:, None] / self.avgdl)

        scores = (self.idf[q_indices] * (num / denom)).sum(axis=1)
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
                    "text_field": self.text_field,
                },
                f,
            )

    def load(self, path: Union[str, Path]):
        """
        Desserializa o estado salvo no disco e o injeta diretamente na instância atual.
        """
        path = str(path)
        if os.path.isdir(path):
            path = os.path.join(path, "corpus_bm25.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"[X] Arquivo de índice não encontrado: {path}")

        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        # Injeta todas as chaves do dicionário (ex: 'tf_matrix', 'vectorizer')
        # como atributos reais desta instância (self)
        self.__dict__.update(state_dict)

        # Validação estrutural de segurança (Fail-Fast)
        if not hasattr(self, "tf_matrix") or self.tf_matrix is None:
            # Fallback de compatibilidade caso você tenha salvo a chave com outro nome
            if "matrix" in state_dict:
                self.tf_matrix = state_dict["matrix"]
            elif "bm25_matrix" in state_dict:
                self.tf_matrix = state_dict["bm25_matrix"]
            else:
                raise KeyError(
                    f"[X] Erro de desserialização: 'tf_matrix' ausente. Chaves no disco: {list(state_dict.keys())}"
                )

        print(f"[V] BM25 Index montado com sucesso! Shape da tf_matrix: {self.tf_matrix.shape}")

    @classmethod
    def load_from_path(cls, path: Path | str) -> "BM25Index":
        """Convenience method to create a new instance and load data."""
        instance = cls()
        instance.load(path)
        return instance


# --- Compatibility utility functions ---


def build_index(
    documents: List[Dict[str, Any]],
    text_field: str = "text",
    citation_field: str = "citation",
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
