import os
import pandas as pd
from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.tools import LawSearchTool
from omnilex.pipelines.stage1 import run_stage1_pipeline
from omnilex.citations.sample_data import SAMPLE_LAWS


class SearchEngineAdapter:
    """Adapts LawSearchTool to the expected .query interface for OOF generation."""

    def __init__(self, tool):
        self.tool = tool

    def query(self, text, top_k=50):
        # Perform real search using BM25
        results = self.tool.index.search(text, top_k=top_k, return_scores=True)
        return [
            {"citation": r.get("citation", "Unknown"), "score": r.get("_score", 0.0)}
            for r in results
        ]


def main():
    # Configuration
    train_path = "data/raw/train.csv"
    output_dir = "data/processed"
    n_splits = 5
    seed = 42
    top_k = 50

    # 1. Load training data
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please ensure data is downloaded.")
        return

    print(f"Loading data from {train_path}...")
    train_df = pd.read_csv(train_path)

    # 2. Setup Search Engine (using real BM25 with sample laws for demo or all if available)
    # In a real scenario, we would load the full corpus.
    # For Stage 1 orchestration demonstration, we'll use a mock or sample if full corpus is not indexed yet.
    print("Initializing BM25 Search Engine...")
    # NOTE: In a real run, we'd load documents from data/processed/ or similar.
    # Here we use SAMPLE_LAWS just to ensure the script can execute.
    index = BM25Index(SAMPLE_LAWS)
    tool = LawSearchTool(index)
    engine = SearchEngineAdapter(tool)

    # 3. Run Pipeline
    print("Starting Stage 1 Pipeline (TE -> CV -> OOF -> TO)...")
    print(f"Intermediate DataFrames will be saved to: {output_dir}")

    optimal_t = run_stage1_pipeline(
        train_df, engine, n_splits=n_splits, seed=seed, top_k=top_k, output_dir=output_dir
    )

    print("-" * 30)
    print("PIPELINE COMPLETE")
    print(f"Final Optimal Threshold: {optimal_t:.4f}")
    print(f"Processed files saved in {output_dir}:")
    print("  - train_binned.csv")
    print("  - train_cv.csv")
    print("  - oof_predictions.csv")
    print("-" * 30)


if __name__ == "__main__":
    main()
