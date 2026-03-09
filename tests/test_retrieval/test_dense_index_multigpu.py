import pytest
from unittest.mock import MagicMock, patch
from omnilex.retrieval.dense_index import DenseIndex

class TestDenseIndexMultiGPU:
    @patch('omnilex.retrieval.dense_index.SentenceTransformer')
    @patch('faiss.IndexFlatIP')
    def test_build_multi_gpu_calls(self, mock_faiss, mock_st_class):
        # Setup
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.start_multi_process_pool.return_value = {'pool': 'dummy'}
        mock_model.encode_multi_process.return_value = MagicMock() # Will return embeddings
        
        idx = DenseIndex(model_name="dummy")
        texts = ["text1", "text2"]
        citations = ["cit1", "cit2"]
        
        # Act
        idx.build_from_lists(texts, citations, multi_gpu=True)
        
        # Assert
        mock_model.start_multi_process_pool.assert_called_once()
        mock_model.encode_multi_process.assert_called_once()
        mock_model.stop_multi_process_pool.assert_called_once()
        assert idx.citations == citations

    def test_metadata_integrity_on_save(self, tmp_path):
        # Setup: Mock FAISS and ST to avoid heavy loading
        with patch('omnilex.retrieval.dense_index.SentenceTransformer'), \
             patch('faiss.write_index'), \
             patch('faiss.read_index'):
            
            idx = DenseIndex(model_name="dummy")
            idx.citations = ["cit1", "cit2", "cit3"]
            idx.index = MagicMock()
            idx.index.ntotal = 3
            
            save_path = tmp_path / "test_dense"
            idx.save(save_path)
            
            # Act: Load back
            loaded_idx = DenseIndex.load(save_path)
            
            # Assert
            assert len(loaded_idx.citations) == 3
            assert loaded_idx.citations == ["cit1", "cit2", "cit3"]
