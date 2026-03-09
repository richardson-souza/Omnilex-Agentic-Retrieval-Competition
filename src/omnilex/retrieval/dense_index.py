"""Dense indexing and search using sentence-transformers and FAISS."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseIndex:
    """Memory-efficient dense index for large-scale semantic search."""

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

        # Initialize model
        print(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

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
        self, texts: List[str], citations: List[str], batch_size: int = 128, multi_gpu: bool = False
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
            raise ValueError("Index not built. Call .build() or .load() first.")

        # Final sanity check for dimensionality
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Dimensionality mismatch: Model generates {query_embedding.shape[1]} dims, "
                f"but FAISS index has {self.index.d} dims. Check if the correct model was loaded."
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

    def load(self, path_prefix: Union[str, Path]):
        """
        Desserializa o estado e sincroniza o modelo de embedding.
        """
        path_prefix = str(path_prefix)
        if os.path.isdir(path_prefix):
            path_prefix = os.path.join(path_prefix, "corpus_dense")

        pkl_path = path_prefix + ".pkl"
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[X] Metadados densos não encontrados em: {pkl_path}")

        print(f"Loading metadata from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Injeta atributos
        old_model_name = self.model_name
        if isinstance(data, dict):
            self.__dict__.update(data)
            if "citations" not in data and "documents" in data:
                self.citations = [
                    doc.get(self.citation_field, "Unknown") for doc in data["documents"]
                ]
        else:
            self.citations = data

        # CRITICAL FIX: Reload model if the name in metadata is different from currently loaded
        if self.model_name != old_model_name:
            print(
                f"  [!] Model mismatch detected. Reloading: {old_model_name} -> {self.model_name}"
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)

        # FAISS Index Loading
        index_path = path_prefix + ".index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"[X] Índice FAISS não encontrado em: {index_path}")

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        # Verify alignment
        if self.model.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError(
                f"Alignment Error: Model {self.model_name} has {self.model.get_sentence_embedding_dimension()} dims, "
                f"but loaded index has {self.index.d} dims."
            )

        print(
            f"[V] Dense Index montado com sucesso! Model: {self.model_name} | Dims: {self.index.d}"
        )

    @classmethod
    def load_from_path(cls, path_prefix: Path | str) -> "DenseIndex":
        """Convenience method to create a new instance and load data."""
        # Note: We can't know the model_name before opening the .pkl
        # so we instantiate with default and let .load() handle the switch
        instance = cls()
        instance.load(path_prefix)
        return instance
