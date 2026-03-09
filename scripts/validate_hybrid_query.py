import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.hybrid import HybridSearchEngine, build_text_lookup
from omnilex.evaluation.metrics import citation_f1


def main():
    processed_dir = "data/processed"
    raw_dir = "data/raw"

    # 1. Load sample query
    print("Loading sample query...")
    translated_df = pd.read_parquet(os.path.join(processed_dir, "train_cv_translated.parquet"))
    sample = translated_df.iloc[0]
    query_de = sample["query_de"]
    gold = sample["gold_citations"].split(";")

    print(f"\nQuery ID: {sample['query_id']}")
    print(f"Query (DE): {query_de[:200]}...")
    print(f"Gold Citations: {gold}")

    # 2. Initialize Engine
    print("\nInitializing Hybrid Search Engine...")
    bm25_idx = BM25Index.load(os.path.join(processed_dir, "corpus_bm25.pkl"))
    dense_idx = DenseIndex.load(os.path.join(processed_dir, "corpus_dense"))

    # Check if we can use reranker (optional for this quick validation)
    use_reranker = False
    reranker = None
    text_lookup = None

    try:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading Reranker on {device}...")
        reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)

        print("  Building text lookup (Lazy Loading)...")
        text_lookup = build_text_lookup(
            os.path.join(raw_dir, "laws_de.csv"), os.path.join(raw_dir, "court_considerations.csv")
        )
        use_reranker = True
    except Exception as e:
        print(f"  Reranker skipped: {e}")

    engine = HybridSearchEngine(
        bm25_index=bm25_idx, dense_index=dense_idx, reranker=reranker, text_lookup=text_lookup
    )

    # 3. Search
    print("\nExecuting Hybrid Search...")
    results = engine.query(query_de, top_k=10)

    # 4. Display Results
    print("\n" + "=" * 50)
    print(f"{'RANK':<5} | {'CITATION':<25} | {'SCORE':<8}")
    print("-" * 50)

    predicted = []
    for i, res in enumerate(results):
        predicted.append(res["citation"])
        print(f"{i + 1:<5} | {res['citation']:<25} | {res['score']:.4f}")

    # 5. Calculate Metrics
    metrics = citation_f1(predicted, gold)
    print("=" * 50)
    print("\nVALIDATION METRICS (Top-10):")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")

    # Check if any gold was found
    found = [c for c in predicted if c in gold]
    print(f"\nGold citations found in Top-10: {found}")


if __name__ == "__main__":
    main()
