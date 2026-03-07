import pandas as pd
import numpy as np
import pytest
from omnilex.data.threshold_optimization import optimize_threshold, compute_macro_f1

def test_compute_macro_f1():
    """Test the macro F1 computation logic."""
    true_list = [['C1', 'C2'], ['C3']]
    pred_list = [['C1'], ['C3', 'C4']]
    
    # Query 1: TP=1, Pred=1, Gold=2 -> P=1.0, R=0.5 -> F1 = 2*1*0.5/(1.5) = 2/3
    # Query 2: TP=1, Pred=2, Gold=1 -> P=0.5, R=1.0 -> F1 = 2*0.5*1/(1.5) = 2/3
    # Mean F1 = 2/3 = 0.666...
    
    score = compute_macro_f1(true_list, pred_list)
    assert score == pytest.approx(2/3)

def test_optimize_threshold_basic():
    """Test that optimize_threshold finds the best threshold."""
    # Setup oof_df
    # q1: gold=['C1'], candidates=[(C1, 0.9), (C2, 0.5)]
    # q2: gold=['C3'], candidates=[(C3, 0.8), (C4, 0.2)]
    
    oof_data = {
        'query_id': ['q1', 'q1', 'q2', 'q2'],
        'citation': ['C1', 'C2', 'C3', 'C4'],
        'score': [0.9, 0.5, 0.8, 0.2]
    }
    oof_df = pd.DataFrame(oof_data)
    
    train_data = {
        'query_id': ['q1', 'q2'],
        'gold_citations': ['C1', 'C3']
    }
    train_df = pd.DataFrame(train_data)
    
    # If T=0.95: q1 predicts [], q2 predicts [] -> F1=0
    # If T=0.85: q1 predicts ['C1'], q2 predicts [] -> F1=(1+0)/2 = 0.5
    # If T=0.75: q1 predicts ['C1'], q2 predicts ['C3'] -> F1=1.0 (Optimal)
    # If T=0.4: q1 predicts ['C1', 'C2'], q2 predicts ['C3'] -> F1=(2/3 + 1)/2 = 0.833
    
    best_t = optimize_threshold(oof_df, train_df)
    
    # Best T should be between 0.51 and 0.8 (it covers C1 and C3 but not C2 or C4)
    assert 0.5 <= best_t <= 0.81

def test_optimize_threshold_speed():
    """Requirement: Grid Search must compute in less than 10 seconds."""
    import time
    
    # Create a larger dummy dataset (1000 queries, 50 candidates each)
    num_queries = 1000
    cands_per_query = 50
    
    oof_data = {
        'query_id': np.repeat([f'q{i}' for i in range(num_queries)], cands_per_query),
        'citation': [f'C{j}' for j in range(num_queries * cands_per_query)],
        'score': np.random.random(num_queries * cands_per_query)
    }
    oof_df = pd.DataFrame(oof_data)
    
    train_data = {
        'query_id': [f'q{i}' for i in range(num_queries)],
        'gold_citations': ['C1'] * num_queries
    }
    train_df = pd.DataFrame(train_data)
    
    start_time = time.time()
    _ = optimize_threshold(oof_df, train_df)
    duration = time.time() - start_time
    
    assert duration < 10.0, f"Optimization took too long: {duration:.2f}s"
