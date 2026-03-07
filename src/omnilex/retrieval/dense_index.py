"""Dense indexing and search using sentence-transformers and FAISS."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseIndex:
    """Dense index for semantic search using FAISS."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        documents: Optional[List[Dict[str, Any]]] = None,
        text_field: str = "text",
        citation_field: str = "citation",
    ):
        self.model_name = model_name
        self.text_field = text_field
        self.citation_field = citation_field

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.documents: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None

        if documents:
            self.build(documents)

    def build(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> None:
        texts = [doc.get(self.text_field, "") for doc in documents]
        self.build_from_lists(texts, documents, batch_size=batch_size)

    def build_from_lists(
        self, texts: List[str], metadata: List[Dict[str, Any]], batch_size: int = 32
    ) -> None:
        """Memory-efficient build from separate lists."""
        self.documents = metadata

        print(f"Generating embeddings for {len(texts)} documents using {self.model_name}...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not built.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.documents[idx].copy()
            if return_scores:
                res_score = float(score)
                # Map [0, 1] range for RRF compatibility
                doc["_score"] = res_score
            results.append(doc)

        return results

    def save(self, path_prefix: Path | str) -> None:
        path_prefix = str(path_prefix)
        faiss.write_index(self.index, path_prefix + ".index")

        metadata = {
            "documents": self.documents,
            "model_name": self.model_name,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
        }
        with open(path_prefix + ".pkl", "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path_prefix: Path | str) -> "DenseIndex":
        path_prefix = str(path_prefix)
        with open(path_prefix + ".pkl", "rb") as f:
            metadata = pickle.load(f)

        instance = cls(
            model_name=metadata["model_name"],
            text_field=metadata["text_field"],
            citation_field=metadata["citation_field"],
        )
        instance.documents = metadata["documents"]
        instance.index = faiss.read_index(path_prefix + ".index")
        return instance
