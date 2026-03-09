import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.hybrid import HybridSearchEngine
from omnilex.evaluation.metrics import citation_f1

def main():
    processed_dir = "data/processed"
    
    # 1. Load sample query
    print("Loading sample query...")
    translated_df = pd.read_parquet(os.path.join(processed_dir, "train_cv_translated.parquet"))
    sample = translated_df.iloc[0]
    query_de = sample["query_de"]
    gold = sample["gold_citations"].split(";")
    
    print(f"\nQuery ID: {sample['query_id']}")
    print(f"Gold Citations: {gold}")

    # 2. Initialize Engine (First Stage ONLY)
    print("\nInitializing Hybrid Search Engine (First Stage)...")
    bm25_idx = BM25Index.load(os.path.join(processed_dir, "corpus_bm25.pkl"))
    dense_idx = DenseIndex.load(os.path.join(processed_dir, "corpus_dense"))
    
    engine = HybridSearchEngine(
        bm25_index=bm25_idx,
        dense_index=dense_idx
    )

    # 3. Search
    print("\nExecuting Hybrid Search (BM25 + Dense + RRF)...")
    results = engine.query(query_de, top_k=10)

    # 4. Display Results
    print("\n" + "="*50)
    print(f"{'RANK':<5} | {'CITATION':<25} | {'SCORE':<8}")
    print("-" * 50)
    
    predicted = []
    for i, res in enumerate(results):
        predicted.append(res['citation'])
        print(f"{i+1:<5} | {res['citation']:<25} | {res['score']:.4f}")
    
    # 5. Calculate Metrics
    metrics = citation_f1(predicted, gold)
    print("="*50)
    print(f"\nVALIDATION METRICS (Top-10):")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    # Check if any gold was found
    found = [c for c in predicted if c in gold]
    print(f"\nGold citations found in Top-10: {found}")

if __name__ == "__main__":
    main()
