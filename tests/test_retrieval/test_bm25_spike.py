import pytest
import numpy as np
import scipy.sparse as sp
from unittest.mock import MagicMock
from omnilex.retrieval.bm25_index import BM25Index


class TestBM25MemoryEfficiency:
    def test_search_logic_integrity(self):
        """Verify that term-by-term search returns correct top citations."""
        # Setup small controlled index
        n_docs = 3
        n_vocab = 5
        idx = BM25Index()

        # doc0: term0, doc1: term1, doc2: term2
        data = [1, 1, 1]
        indices = [0, 1, 2]
        indptr = [0, 1, 2, 3]
        idx.tf_matrix = sp.csr_matrix((data, indices, indptr), shape=(n_docs, n_vocab))

        idx.doc_lens = np.array([1, 1, 1])
        idx.avgdl = 1.0
        idx.idf = np.array([10.0, 5.0, 1.0, 0.0, 0.0])  # Term 0 is most informative
        idx.citations = ["Doc0", "Doc1", "Doc2"]

        # Mock vectorizer
        mock_vec = MagicMock()
        # Query has term 0 and term 2
        mock_vec.transform.return_value = sp.csr_matrix(
            ([1, 1], ([0, 0], [0, 2])), shape=(1, n_vocab)
        )
        idx.vectorizer = mock_vec

        # Act
        results = idx.search("term0 term2", top_k=3)

        # Assert
        # Doc0 should be first because it has Term 0 (IDF 10)
        # Doc2 should be second because it has Term 2 (IDF 1)
        assert results[0]["citation"] == "Doc0"
        assert results[1]["citation"] == "Doc2"
        assert len(results) == 2  # Doc1 has no query terms
