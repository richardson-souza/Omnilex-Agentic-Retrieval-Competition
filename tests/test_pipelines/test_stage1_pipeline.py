import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from omnilex.pipelines.stage1 import run_stage1_pipeline

def test_run_stage1_pipeline_orchestration():
    """Test that all components are called in the correct sequence."""
    
    # 1. Setup mock data
    train_df = pd.DataFrame({
        'query_id': ['q1', 'q2'], 
        'query': ['text1', 'text2'],
        'gold_citations': ['C1', 'C2']
    })
    search_engine = MagicMock()
    
    # 2. Patch each component WHERE IT IS IMPORTED in stage1.py
    with patch('omnilex.pipelines.stage1.create_cardinality_bins') as mock_te, \
         patch('omnilex.pipelines.stage1.setup_cv') as mock_cv, \
         patch('omnilex.pipelines.stage1.generate_oof_predictions') as mock_oof, \
         patch('omnilex.pipelines.stage1.optimize_threshold') as mock_to:
        
        # Configure mocks to return expected outputs
        df_binned = train_df.copy()
        df_binned['fold_bin'] = [0, 1]
        mock_te.return_value = df_binned
        
        df_cv = df_binned.copy()
        df_cv['fold'] = [0, 1]
        mock_cv.return_value = df_cv
        
        oof_df = pd.DataFrame({'query_id': ['q1'], 'citation': ['C1'], 'score': [0.9]})
        mock_oof.return_value = oof_df
        
        mock_to.return_value = 0.85
        
        # 3. Act
        result = run_stage1_pipeline(train_df, search_engine, n_splits=2, seed=42, top_k=50)
        
        # 4. Assert sequence and arguments
        # TE called first
        mock_te.assert_called_once_with(train_df, n_splits=2)
        
        # CV called with TE output
        mock_cv.assert_called_once_with(df_binned, n_splits=2, seed=42)
        
        # OOF called with CV output
        mock_oof.assert_called_once_with(df_cv, search_engine, top_k=50)
        
        # TO called with OOF and CV output
        mock_to.assert_called_once_with(oof_df, df_cv, verbose=True)
        
        # Result should be the threshold
        assert result == 0.85
