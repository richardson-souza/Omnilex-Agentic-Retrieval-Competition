import os
import pandas as pd
from omnilex.pipelines.stage1 import run_stage1_pipeline


def main():
    # Configuration
    train_path = "data/raw/train.csv"
    output_dir = "data/processed"
    raw_data_dir = "data/raw"
    n_splits = 5
    seed = 42
    top_k = 100
    top_k_rerank = 50  # Refine top 50 with Cross-Encoder

    # Check resources
    try:
        import torch

        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    # Environmental Setup logic
    translated_path = os.path.join(output_dir, "train_cv_translated.parquet")
    skip_translation = not HAS_TORCH or os.path.exists(translated_path)

    bm25_path = os.path.join(output_dir, "corpus_bm25.pkl")
    dense_path = os.path.join(output_dir, "corpus_dense.index")
    skip_indexing = os.path.exists(bm25_path) and os.path.exists(dense_path)

    # REQ_10: Use Reranker?
    # Warning: Lazy loading full corpus requires ~4-8GB RAM + indices.
    use_reranker = HAS_TORCH

    # 1. Load training data
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    # 2. Run Pipeline
    print(f"Starting Stage 1-4 Pipeline...")
    print(f"  - Translation: {'Skipped' if skip_translation else 'Enabled'}")
    print(f"  - Indexing: {'Skipped' if skip_indexing else 'Enabled'}")
    print(f"  - Reranker: {'Enabled' if use_reranker else 'Disabled'}")

    optimal_t = run_stage1_pipeline(
        train_df,
        search_engine=None,
        n_splits=n_splits,
        seed=seed,
        top_k=top_k,
        top_k_rerank=top_k_rerank,
        output_dir=output_dir,
        raw_data_dir=raw_data_dir,
        skip_translation=skip_translation,
        skip_indexing=skip_indexing,
        use_reranker=use_reranker,
    )

    print("-" * 30)
    print("PIPELINE COMPLETE")
    print(f"Final Optimal Threshold (T*): {optimal_t:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
