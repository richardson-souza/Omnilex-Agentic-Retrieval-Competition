import pandas as pd
import pytest
import numpy as np
from omnilex.data.oof_generation import generate_oof_predictions

class MockSearchEngine:
    """Mock search engine for testing OOF generation."""
    def __init__(self, responses):
        self.responses = responses
        self.called_with = []

    def query(self, text, top_k=50):
        self.called_with.append(text)
        # Return a fixed response for testing
        return self.responses.get(text, [])

def test_generate_oof_predictions_basic():
    """Test that OOF predictions are generated correctly for each fold."""
    # Setup data with 2 folds
    data = {
        'query_id': ['q1', 'q2', 'q3', 'q4'],
        'query': ['text1', 'text2', 'text3', 'text4'],
        'fold': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Setup mock search engine responses
    # Scores are NOT normalized initially to test MinMax scaling
    responses = {
        'text1': [{'citation': 'C1', 'score': 10.0}, {'citation': 'C2', 'score': 5.0}],
        'text2': [{'citation': 'C3', 'score': 100.0}],
        'text3': [{'citation': 'C1', 'score': 1.0}],
        'text4': [] # No results
    }
    search_engine = MockSearchEngine(responses)
    
    # Act
    oof_df = generate_oof_predictions(df, search_engine, top_k=2)
    
    # Assert
    assert isinstance(oof_df, pd.DataFrame)
    assert set(oof_df.columns) == {'query_id', 'citation', 'score'}
    
    # q1 has 2 results, q2 has 1, q3 has 1, q4 has 0. Total = 4 rows.
    assert len(oof_df) == 4
    
    # Check if all validation queries were processed
    assert set(oof_df['query_id'].unique()) == {'q1', 'q2', 'q3'} # q4 has no results
    
    # Check normalization for q1: max=10, min=5 -> normalized: C1=1.0, C2=0.0
    q1_results = oof_df[oof_df['query_id'] == 'q1']
    assert q1_results[q1_results['citation'] == 'C1']['score'].iloc[0] == 1.0
    assert q1_results[q1_results['citation'] == 'C2']['score'].iloc[0] == 0.0
    
    # Check normalization for q2 (single result should be 1.0 or handled gracefully)
    q2_results = oof_df[oof_df['query_id'] == 'q2']
    assert q2_results['score'].iloc[0] == 1.0

def test_generate_oof_predictions_min_max_scaling():
    """Test MinMax scaling specifically."""
    data = {
        'query_id': ['q1'],
        'query': ['text1'],
        'fold': [0]
    }
    df = pd.DataFrame(data)
    
    # 3 results with different scores
    responses = {
        'text1': [
            {'citation': 'C1', 'score': 20.0},
            {'citation': 'C2', 'score': 15.0},
            {'citation': 'C3', 'score': 10.0}
        ]
    }
    search_engine = MockSearchEngine(responses)
    
    oof_df = generate_oof_predictions(df, search_engine)
    
    scores = oof_df['score'].tolist()
    assert max(scores) == 1.0
    assert min(scores) == 0.0
    assert 0.5 in scores # (15-10)/(20-10) = 0.5

def test_generate_oof_predictions_with_bm25_dtype():
    """Test OOF generation with real BM25Index to ensure no UFuncTypeError."""
    from omnilex.retrieval.bm25_index import BM25Index
    
    data = {
        'query_id': ['q1', 'q2'],
        'query': ['apple', 'banana'],
        'fold': [0, 1]
    }
    df = pd.DataFrame(data)
    
    docs = [
        {'citation': 'A1', 'text': 'apple fruit'},
        {'citation': 'B1', 'text': 'banana fruit'},
        {'citation': 'F1', 'text': 'fruit'}
    ]
    bm25 = BM25Index(documents=docs)
    
    # We need to wrap it because generate_oof_predictions calls .query
    class SearchWrapper:
        def __init__(self, engine):
            self.engine = engine
        def query(self, text, top_k=50):
            # BM25Index.search returns results with '_score' but oof_generation expects 'score'
            # Wait, let's check what oof_generation expects.
            # It expects 'score' if it uses candidates[i]['score']
            results = self.engine.search(text, top_k=top_k)
            for r in results:
                if '_score' in r:
                    r['score'] = r.pop('_score')
            return results

    wrapper = SearchWrapper(bm25)
    
    # Act: This should NOT raise UFuncTypeError
    oof_df = generate_oof_predictions(df, wrapper, top_k=10)
    
    # Assert
    assert isinstance(oof_df, pd.DataFrame)
    assert len(oof_df) > 0
