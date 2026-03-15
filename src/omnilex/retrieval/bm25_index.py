"""Ultra-fast memory-efficient BM25 indexing using Scipy CSC Matrices."""

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
    """Memory-efficient BM25 index optimized for column-access speed (CSC)."""

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

        self.vectorizer = CountVectorizer(
            lowercase=True, token_pattern=r"(?u)\b\w\w+\b"
        )
        self.citations: List[str] = []
        self.tf_matrix: Optional[sp.csc_matrix] = None
        self.doc_lens: Optional[np.ndarray] = None
        self.idf: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

        if documents:
            self.build_from_docs(documents)

    def build_from_docs(self, documents: List[Dict[str, Any]]) -> None:
        texts = [doc.get(self.text_field, "") for doc in documents]
        citations = [doc.get(self.citation_field, "Unknown") for doc in documents]
        self.build(texts, citations)

    def build(self, texts: List[str], citations: List[str]) -> None:
        """Build index and convert to CSC for fast retrieval."""
        print(f"  Building sparse matrix for {len(texts)} docs...")
        self.citations = citations

        # 1. Fit and transform
        csr_matrix = self.vectorizer.fit_transform(texts)

        # 2. Precompute stats
        self.doc_lens = np.array(csr_matrix.sum(axis=1)).flatten().astype(np.float32)
        self.avgdl = self.doc_lens.mean()

        # 3. Compute IDF
        n_docs = csr_matrix.shape[0]
        doc_freqs = np.array((csr_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((n_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0).astype(np.float32)

        # 4. Convert to CSC
        print("  Converting matrix to CSC format...")
        self.tf_matrix = csr_matrix.tocsc()
        
        del csr_matrix
        gc.collect()

    def search(
        self, query: str, top_k: int = 10, return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """ULTRA-FAST CSC-based BM25 search."""
        if self.tf_matrix is None:
            raise ValueError("Index not built.")

        query_vec = self.vectorizer.transform([query])
        if query_vec.nnz == 0:
            return []

        q_indices = query_vec.indices
        n_docs = self.tf_matrix.shape[0]
        
        total_scores = np.zeros(n_docs, dtype=np.float32)
        denom_base = (self.k1 * (1.0 - self.b + self.b * self.doc_lens / self.avgdl)).astype(np.float32)

        for term_idx in q_indices:
            # FIX: Convert to float32 immediately to avoid UFuncTypeError on +=
            tf_col = self.tf_matrix[:, term_idx].toarray().flatten().astype(np.float32)
            
            numerator = tf_col * (self.k1 + 1.0)
            tf_col += denom_base
            
            total_scores += self.idf[term_idx] * (numerator / tf_col)
            
            del tf_col, numerator

        if top_k < n_docs:
            top_indices = np.argpartition(total_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(total_scores[top_indices])][::-1]
        else:
            top_indices = np.argsort(total_scores)[::-1]

        results = []
        for idx in top_indices:
            if total_scores[idx] <= 0: continue
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
            raise FileNotFoundError(f"[X] BM25 Index not found: {path}")

        with open(path, "rb") as f:
            state_dict = pickle.load(f)
            
        self.__dict__.update(state_dict)
        
        if not hasattr(self, 'tf_matrix') or self.tf_matrix is None:
            for key in ['matrix', 'bm25_matrix']:
                if key in state_dict:
                    self.tf_matrix = state_dict[key]
                    break
            else:
                raise KeyError("[X] 'tf_matrix' missing in pkl.")
        
        if not isinstance(self.tf_matrix, sp.csc_matrix):
            self.tf_matrix = self.tf_matrix.tocsc()
                
        self.doc_lens = self.doc_lens.astype(np.float32)
        self.idf = self.idf.astype(np.float32)
        print(f"[V] BM25 Index pronto! Shape: {self.tf_matrix.shape}")

    @classmethod
    def load_from_path(cls, path: Path | str) -> "BM25Index":
        instance = cls()
        instance.load(path)
        return instance


# --- Compatibility utility functions ---

def build_index(documents: List[Dict[str, Any]], text_field: str = "text", citation_field: str = "citation") -> BM25Index:
    return BM25Index(documents=documents, text_field=text_field, citation_field=citation_field)

def search(index: BM25Index, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    return index.search(query, top_k=top_k)

def load_jsonl_corpus(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: documents.append(json.loads(line))
    return documents

def save_jsonl_corpus(documents: List[Dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
