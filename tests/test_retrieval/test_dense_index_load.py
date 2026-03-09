import pytest
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch
from omnilex.retrieval.dense_index import DenseIndex

class TestDenseIndexLoad:
    @patch('omnilex.retrieval.dense_index.SentenceTransformer')
    def test_load_from_directory_fails_currently(self, mock_st, tmp_path):
        """Verify the error message when metadata is missing in a directory."""
        index_dir = tmp_path / "empty-dataset"
        index_dir.mkdir()
        
        idx = DenseIndex()
        with pytest.raises(FileNotFoundError) as excinfo:
            idx.load(str(index_dir))
        
        assert "Metadados densos não encontrados" in str(excinfo.value)

    @patch('omnilex.retrieval.dense_index.SentenceTransformer')
    @patch('faiss.read_index')
    def test_load_from_directory_fixed(self, mock_faiss_read, mock_st_class, tmp_path):
        """Test loading Dense index by passing only the directory using instance method."""
        index_dir = tmp_path / "fixed-dataset"
        index_dir.mkdir()
        
        # Setup mocks
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        mock_model.get_sentence_embedding_dimension.return_value = 768
        
        mock_faiss_idx = MagicMock()
        mock_faiss_idx.d = 768 # Match dimension
        mock_faiss_read.return_value = mock_faiss_idx
        
        # Create dummy files
        metadata_path = index_dir / "corpus_dense.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({"citations": ["cit1"], "model_name": "dummy"}, f)
            
        index_file = index_dir / "corpus_dense.index"
        index_file.touch()
        
        # Act
        idx = DenseIndex()
        idx.load(str(index_dir))
        
        # Assert
        assert idx.citations == ["cit1"]
        assert idx.model_name == "dummy"
        assert mock_faiss_read.called

    @patch('omnilex.retrieval.dense_index.SentenceTransformer')
    @patch('faiss.read_index')
    def test_load_from_path_classmethod(self, mock_faiss_read, mock_st_class, tmp_path):
        """Test that the convenience classmethod still works."""
        index_dir = tmp_path / "class-dataset"
        index_dir.mkdir()
        
        # Setup mocks
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        mock_faiss_idx = MagicMock()
        mock_faiss_idx.d = 384
        mock_faiss_read.return_value = mock_faiss_idx
        
        metadata_path = index_dir / "corpus_dense.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({"citations": ["cit2"], "model_name": "dummy-small"}, f)
        
        (index_dir / "corpus_dense.index").touch()
        
        # Act
        idx = DenseIndex.load_from_path(str(index_dir))
        
        # Assert
        assert idx.citations == ["cit2"]
        assert idx.model_name == "dummy-small"
