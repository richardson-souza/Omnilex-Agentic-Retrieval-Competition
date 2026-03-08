"""Dense indexing and search using sentence-transformers and FAISS."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseIndex:
    """Memory-efficient dense index for large-scale semantic search."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
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

        # Store ONLY citations to save RAM
        self.citations: List[str] = []
        self.index: Optional[faiss.Index] = None

        if documents:
            self.build(documents)

    def build(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """Compatibility method to build from list of dicts."""
        texts = [doc.get(self.text_field, "") for doc in documents]
        citations = [doc.get(self.citation_field, "Unknown") for doc in documents]
        self.build_from_lists(texts, citations, batch_size=batch_size)

    def build_from_lists(
        self, texts: List[str], citations: List[str], batch_size: int = 128
    ) -> None:
        """Build FAISS index from texts and store corresponding citations."""
        self.citations = citations

        print(
            f"Generating embeddings for {len(texts)} documents using {self.model_name}..."
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search the index and map results to citations."""
        if self.index is None:
            raise ValueError("Index not built.")

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.citations):
                continue

            res = {"citation": self.citations[idx]}
            if return_scores:
                res["_score"] = float(score)
            results.append(res)

        return results

    def save(self, path_prefix: Path | str) -> None:
        path_prefix = str(path_prefix)
        faiss.write_index(self.index, path_prefix + ".index")

        metadata = {
            "citations": self.citations,
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
        if "citations" in metadata:
            instance.citations = metadata["citations"]
        elif "documents" in metadata:
            instance.citations = [
                doc.get(instance.citation_field, "Unknown")
                for doc in metadata["documents"]
            ]

        instance.index = faiss.read_index(path_prefix + ".index")
        return instance
