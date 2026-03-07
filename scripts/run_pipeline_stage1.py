import os
import pandas as pd
from omnilex.pipelines.stage1 import run_stage1_pipeline


def main():
    # Configuration
    train_path = "data/raw/train.csv"
    output_dir = "data/processed"
    n_splits = 5
    seed = 42
    top_k = 100 # Increased for hybrid RRF robustness
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Environmental Setup
    try:
        import torch
        HAS_TORCH = True
        print(f"PyTorch detectou a GPU? {torch.cuda.is_available()}")
    except ImportError:
        HAS_TORCH = False

    # Check if we should skip translation (e.g., if already exists or no torch)
    translated_path = os.path.join(output_dir, "train_cv_translated.parquet")
    skip_translation = not HAS_TORCH or os.path.exists(translated_path)
    
    # Check if we should skip indexing (e.g., if already exists)
    bm25_path = os.path.join(output_dir, "corpus_bm25.pkl")
    dense_path = os.path.join(output_dir, "corpus_dense.index")
    skip_indexing = os.path.exists(bm25_path) and os.path.exists(dense_path)

    # 1. Load training data
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please ensure data is downloaded.")
        return

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    # 2. Run Pipeline
    # The pipeline now handles HybridSearchEngine initialization internally
    print(f"Starting Stage 1-3 Pipeline...")
    print(f"  - Translation: {'Skipped (using cache or no torch)' if skip_translation else 'Enabled'}")
    print(f"  - Indexing: {'Skipped (using cache)' if skip_indexing else 'Enabled'}")
    print(f"  - Output Dir: {output_dir}")

    optimal_t = run_stage1_pipeline(
        train_df,
        search_engine=None, # Will be initialized inside pipeline
        n_splits=n_splits,
        seed=seed,
        top_k=top_k,
        output_dir=output_dir,
        skip_translation=skip_translation,
        skip_indexing=skip_indexing,
        model_name=model_name
    )

    print("-" * 30)
    print("PIPELINE COMPLETE")
    print(f"Final Optimal Threshold (T*): {optimal_t:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
