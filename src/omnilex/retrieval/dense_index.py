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
        self,
        texts: List[str],
        citations: List[str],
        batch_size: int = 128,
        multi_gpu: bool = False,
    ) -> None:
        """Build FAISS index from texts and store corresponding citations."""
        self.citations = citations

        print(f"Generating embeddings for {len(texts)} documents using {self.model_name}...")

        if multi_gpu:
            print("  Using multi-process multi-GPU pool...")
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(
                texts, pool, batch_size=batch_size, normalize_embeddings=True
            )
            self.model.stop_multi_process_pool(pool)
        else:
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
        """Save FAISS index and citations list separately."""
        path_prefix = str(path_prefix)

        # Save FAISS index (atomic)
        print(f"Saving FAISS index to {path_prefix}.index...")
        faiss.write_index(self.index, path_prefix + ".index")

        # Save metadata
        print(f"Saving metadata to {path_prefix}.pkl...")
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
        """Load FAISS index and citations list with fallback for different formats."""
        path_prefix = str(path_prefix)

        pkl_path = path_prefix + ".pkl"
        print(f"Loading metadata from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Handle various serialization formats
        if isinstance(data, dict):
            model_name = data.get("model_name", "intfloat/multilingual-e5-small")
            text_field = data.get("text_field", "text")
            citation_field = data.get("citation_field", "citation")
            citations = data.get("citations", [])
            if not citations and "documents" in data:
                citations = [doc.get(citation_field, "Unknown") for doc in data["documents"]]
        else:
            # Fallback for when the .pkl is just a list of strings
            print("  Warning: metadata is a raw list, using default model parameters.")
            model_name = "intfloat/multilingual-e5-small"
            text_field = "text"
            citation_field = "citation"
            citations = data

        instance = cls(
            model_name=model_name,
            text_field=text_field,
            citation_field=citation_field,
        )
        instance.citations = citations

        index_path = path_prefix + ".index"
        print(f"Loading FAISS index from {index_path}...")
        instance.index = faiss.read_index(index_path)

        return instance
