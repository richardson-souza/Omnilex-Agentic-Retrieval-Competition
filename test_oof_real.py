import pandas as pd
import pytest
import numpy as np
from omnilex.data.oof_generation import generate_oof_predictions
from omnilex.retrieval.bm25_index import BM25Index

def test_generate_oof_predictions_real_bm25():
    """Test OOF generation with a real BM25Index to check for dtype errors."""
    # Setup data with 2 folds
    data = {
        'query_id': ['q1', 'q2'],
        'query': ['text1', 'text2'],
        'fold': [0, 1]
    }
    df = pd.DataFrame(data)
    
    # Setup real BM25 index
    documents = [
        {'citation': 'C1', 'text': 'text1 match'},
        {'citation': 'C2', 'text': 'text2 match'},
        {'citation': 'C3', 'text': 'nothing here'}
    ]
    bm25 = BM25Index(documents=documents)
    
    # Act: This should NOT raise TypeError
    oof_df = generate_oof_predictions(df, bm25, top_k=2)
    
    # Assert
    assert isinstance(oof_df, pd.DataFrame)
    assert len(oof_df) > 0
    print("OOF generation with real BM25 passed!")

if __name__ == "__main__":
    test_generate_oof_predictions_real_bm25()
