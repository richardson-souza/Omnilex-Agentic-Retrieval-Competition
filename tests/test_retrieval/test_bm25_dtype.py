import pytest
import numpy as np
import scipy.sparse as sp
from unittest.mock import MagicMock
from omnilex.retrieval.bm25_index import BM25Index

class TestBM25DtypeIntegrity:
    def test_search_handles_int_to_float_casting(self):
        """Verify that search handles int64 to float32 casting without UFuncTypeError."""
        n_docs = 5
        n_vocab = 3
        idx = BM25Index()
        
        # Matrix with INT64
        data = np.array([1, 2, 3], dtype=np.int64)
        indices = [0, 1, 2]
        indptr = [0, 1, 2, 3]
        idx.tf_matrix = sp.csc_matrix((data, indices, indptr), shape=(n_docs, n_vocab))
        
        idx.doc_lens = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        idx.avgdl = 1.0
        idx.idf = np.array([1.0]*n_vocab, dtype=np.float32)
        idx.citations = [f"Doc{i}" for i in range(n_docs)]
        
        mock_vec = MagicMock()
        mock_vec.transform.return_value = sp.csr_matrix(([1], ([0], [0])), shape=(1, n_vocab))
        idx.vectorizer = mock_vec
        
        # Act: This should NOT raise TypeError anymore
        results = idx.search("query", top_k=3)
        
        # Assert
        assert len(results) > 0
        assert "citation" in results[0]
