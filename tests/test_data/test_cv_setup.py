import pandas as pd
import pytest
import numpy as np
from omnilex.data.cv_setup import setup_cv

def test_setup_cv_basic():
    """Test that folds are correctly assigned."""
    # Create a dummy dataset with fold_bins
    # Need enough samples to satisfy n_splits=5 (at least 5 samples per bin)
    data = {
        'query_id': [f'q{i}' for i in range(50)],
        'fold_bin': ([0] * 10) + ([1] * 10) + ([2] * 20) + ([3] * 10)
    }
    df = pd.DataFrame(data)
    
    n_splits = 5
    # Act
    df_result = setup_cv(df, n_splits=n_splits, seed=42)
    
    # Assert
    assert 'fold' in df_result.columns
    assert df_result['fold'].isna().sum() == 0
    assert (df_result['fold'] == -1).sum() == 0
    
    # Check that we have exactly n_splits unique folds
    unique_folds = sorted(df_result['fold'].unique())
    assert unique_folds == list(range(n_splits))
    
    # Check each fold has samples
    fold_counts = df_result['fold'].value_counts()
    for count in fold_counts:
        assert count == 10 # 50 / 5 = 10 per fold

def test_setup_cv_stratification():
    """Test that fold_bin distribution is preserved in each fold."""
    data = {
        'query_id': [f'q{i}' for i in range(100)],
        'fold_bin': ([0] * 20) + ([1] * 30) + ([2] * 50)
    }
    df = pd.DataFrame(data)
    
    n_splits = 5
    df_result = setup_cv(df, n_splits=n_splits, seed=42)
    
    # Each fold should have 20 samples (100/5)
    # Distribution in each fold should be:
    # bin 0: 20/5 = 4
    # bin 1: 30/5 = 6
    # bin 2: 50/5 = 10
    
    for fold_idx in range(n_splits):
        fold_data = df_result[df_result['fold'] == fold_idx]
        assert len(fold_data) == 20
        assert (fold_data['fold_bin'] == 0).sum() == 4
        assert (fold_data['fold_bin'] == 1).sum() == 6
        assert (fold_data['fold_bin'] == 2).sum() == 10

def test_setup_cv_no_overlap():
    """Test that indices are unique across folds (implicitly tested by basic, but let's be explicit)."""
    # If a row belongs to fold 0, it shouldn't belong to any other fold.
    # In this implementation, 'fold' is a column, so this is naturally true.
    # The requirement asks for intersection of training and validation to be zero.
    # In fold i: val = df[df.fold == i], train = df[df.fold != i].
    
    data = {
        'query_id': [f'q{i}' for i in range(20)],
        'fold_bin': [0] * 20
    }
    df = pd.DataFrame(data)
    
    n_splits = 4
    df_result = setup_cv(df, n_splits=n_splits)
    
    for i in range(n_splits):
        val_indices = set(df_result[df_result['fold'] == i].index)
        train_indices = set(df_result[df_result['fold'] != i].index)
        
        assert val_indices.intersection(train_indices) == set()
        assert val_indices.union(train_indices) == set(df.index)
