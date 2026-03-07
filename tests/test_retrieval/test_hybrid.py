import pytest
import numpy as np
from unittest.mock import MagicMock
from omnilex.retrieval.hybrid import HybridSearchEngine


def test_rrf_logic():
    """Test that RRF correctly fuses rankings."""
    # Mock BM25: Result A (rank 0), Result B (rank 1)
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = [
        {"citation": "DocA", "text": "text a"},
        {"citation": "DocB", "text": "text b"}
    ]
    
    # Mock Dense: Result C (rank 0), Result A (rank 1)
    dense_mock = MagicMock()
    dense_mock.search.return_value = [
        {"citation": "DocC", "text": "text c"},
        {"citation": "DocA", "text": "text a"}
    ]
    
    engine = HybridSearchEngine(bm25_mock, dense_mock, rrf_k=60)
    
    # Act
    results = engine.query("test query", top_k=3)
    
    # Assert
    # DocA: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252 (Rank 1)
    # DocC: 1/(60+1) = 0.01639 (Rank 2)
    # DocB: 1/(60+2) = 0.01613 (Rank 3)
    
    citations = [r["citation"] for r in results]
    assert citations == ["DocA", "DocC", "DocB"]
    
    # Check normalization: DocA should be 1.0, DocB should be 0.0
    assert results[0]["score"] == 1.0
    assert results[-1]["score"] == 0.0


def test_hybrid_empty_results():
    """Test that it handles empty results from both engines."""
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = []
    dense_mock = MagicMock()
    dense_mock.search.return_value = []
    
    engine = HybridSearchEngine(bm25_mock, dense_mock)
    results = engine.query("query", top_k=10)
    
    assert results == []


def test_hybrid_single_result():
    """Test normalization with a single result."""
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = [{"citation": "DocA"}]
    dense_mock = MagicMock()
    dense_mock.search.return_value = []
    
    engine = HybridSearchEngine(bm25_mock, dense_mock)
    results = engine.query("query", top_k=10)
    
    assert len(results) == 1
    assert results[0]["score"] == 1.0
