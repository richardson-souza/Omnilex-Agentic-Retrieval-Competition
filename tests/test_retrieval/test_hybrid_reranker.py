import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from omnilex.retrieval.hybrid import HybridSearchEngine

class TestHybridReranker:
    def test_hybrid_query_with_reranking_df_lookup(self):
        """Test that the engine orchestrates RRF and then calls Reranker using DataFrame lookup."""
        # Setup mocks
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [{"citation": "DocA"}, {"citation": "DocB"}]
        
        dense_mock = MagicMock()
        dense_mock.search.return_value = [{"citation": "DocC"}, {"citation": "DocA"}]
        
        reranker_mock = MagicMock()
        # Mock predict to return logits
        reranker_mock.predict.return_value = np.array([2.0, 0.5]) 
        
        # DataFrame lookup as requested for memory efficiency
        text_lookup = pd.DataFrame({
            "text": ["Texto A", "Texto C"]
        }, index=["DocA", "DocC"])
        
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
        reranker_mock.predict.assert_called_once()
        assert results[0]["citation"] == "DocA"
        assert 0.0 <= results[0]["score"] <= 1.0
