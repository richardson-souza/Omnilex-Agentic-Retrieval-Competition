import pytest
import os
import pickle
from pathlib import Path
from omnilex.retrieval.bm25_index import BM25Index

class TestBM25IndexLoad:
    def test_load_from_directory_instance_method(self, tmp_path):
        """Test loading BM25 index using the new instance method load()."""
        index_dir = tmp_path / "bm25-dataset-v2"
        index_dir.mkdir()
        
        metadata_path = index_dir / "corpus_bm25.pkl"
        
        # Data structure as it would be saved on disk
        dummy_data = {
            "tf_matrix": MagicMockMatrix(shape=(10, 5)), # Mocking a sparse matrix shape
            "vectorizer": "mock_vectorizer",
            "citations": ["cit1"],
            "doc_lens": [10],
            "avgdl": 10.0,
            "idf": [1.0]
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(dummy_data, f)
            
        # Act
        idx = BM25Index()
        idx.load(str(index_dir))
        
        # Assert
        assert idx.citations == ["cit1"]
        assert idx.tf_matrix.shape == (10, 5)

class MagicMockMatrix:
    """Helper to mock sparse matrix shape attribute."""
    def __init__(self, shape):
        self.shape = shape
