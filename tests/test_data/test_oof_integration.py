import pandas as pd
import pytest
from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.tools import LawSearchTool
from omnilex.data.oof_generation import generate_oof_predictions

class SearchEngineAdapter:
    """Adapts LawSearchTool/CourtSearchTool to the expected .query interface."""
    def __init__(self, tool):
        self.tool = tool
        
    def query(self, text, top_k=50):
        # We need a method that returns [{'citation': ..., 'score': ...}, ...]
        # LawSearchTool has search_with_metadata which returns [{'citation': ..., 'text': ..., '_score': ...}, ...]
        results = self.tool.index.search(text, top_k=top_k, return_scores=True)
        return [{'citation': r['citation'], 'score': r['_score']} for r in results]

def test_oof_integration_with_real_bm25(sample_laws_corpus):
    """Integration test using the actual BM25 index."""
    # 1. Setup Index
    index = BM25Index(sample_laws_corpus)
    tool = LawSearchTool(index)
    engine = SearchEngineAdapter(tool)
    
    # 2. Setup Dummy Train Data with Folds
    data = {
        'query_id': ['train_1', 'train_2'],
        'query': ['Vertrag Abschluss', 'Zivilgesetzbuch Rechtsfragen'],
        'fold': [0, 1]
    }
    df = pd.DataFrame(data)
    
    # 3. Generate OOF
    oof_df = generate_oof_predictions(df, engine, top_k=2)
    
    # 4. Assert
    assert len(oof_df) > 0
    assert 'query_id' in oof_df.columns
    assert 'citation' in oof_df.columns
    assert 'score' in oof_df.columns
    
    # Check that scores are normalized
    for qid in oof_df['query_id'].unique():
        q_scores = oof_df[oof_df['query_id'] == qid]['score']
        if len(q_scores) > 1:
            assert q_scores.max() == 1.0
            assert q_scores.min() == 0.0
        else:
            assert q_scores.iloc[0] == 1.0
