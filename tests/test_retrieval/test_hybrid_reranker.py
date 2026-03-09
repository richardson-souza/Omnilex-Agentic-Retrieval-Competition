import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from omnilex.retrieval.hybrid import HybridSearchEngine

class TestHybridReranker:
    def test_hybrid_query_with_reranking(self):
        """Test that the engine orchestrates RRF and then calls Reranker."""
        # Setup mocks
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [{"citation": "DocA"}, {"citation": "DocB"}]
        
        dense_mock = MagicMock()
        dense_mock.search.return_value = [{"citation": "DocC"}, {"citation": "DocA"}]
        
        reranker_mock = MagicMock()
        # Mock predict to return logits
        reranker_mock.predict.return_value = np.array([2.0, 0.5]) # Sigmoids will be ~0.88 and ~0.62
        
        text_lookup = {"DocA": "Texto A", "DocC": "Texto C"}
        
        # This should fail initially as HybridSearchEngine doesn't support reranker/text_lookup
        engine = HybridSearchEngine(
            bm25_index=bm25_mock,
            dense_index=dense_mock,
            reranker=reranker_mock,
            text_lookup=text_lookup
        )
        
        # Act
        results = engine.query("test query", top_k=2, top_k_rerank=2)
        
        # Assert
        assert len(results) == 2
        # Verify reranker was called with correct pairs
        # Top 2 from RRF would be DocA and DocC
        reranker_mock.predict.assert_called_once()
        # Scores should be probabilities (0 to 1)
        assert 0.0 <= results[0]["score"] <= 1.0
        assert results[0]["citation"] == "DocA" # Higher logit
