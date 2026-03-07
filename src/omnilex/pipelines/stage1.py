import os
import pandas as pd
from ..data.target_engineering import create_cardinality_bins
from ..data.cv_setup import setup_cv
from ..data.oof_generation import generate_oof_predictions
from ..data.threshold_optimization import optimize_threshold


def run_stage1_pipeline(
    train_df: pd.DataFrame, 
    search_engine, 
    n_splits: int = 5, 
    seed: int = 42, 
    top_k: int = 50,
    output_dir: str = "data/processed"
):
    """
    Orchestrate Stage 1 of the competition pipeline.

    1. Target Engineering (Binning)
    2. CV Setup (Stratified K-Fold)
    3. OOF Inference (Retrieval)
    4. Threshold Optimization (Numerical Calibration)

    Args:
        train_df: Original training DataFrame.
        search_engine: Search engine instance for retrieval.
        n_splits: Number of folds for CV.
        seed: Random seed.
        top_k: Top K candidates to retrieve during OOF generation.
        output_dir: Directory to save the intermediate DataFrames.

    Returns:
        optimal_threshold (T*).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Fase 1: Preparação do CV
    # 1. Target Engineering
    print("Step 1: Creating cardinality bins...")
    train_df_binned = create_cardinality_bins(train_df, n_splits=n_splits)
    train_df_binned.to_csv(os.path.join(output_dir, "train_binned.csv"), index=False)

    # 2. CV Setup
    print("Step 2: Setting up CV folds...")
    train_df_cv = setup_cv(train_df_binned, n_splits=n_splits, seed=seed)
    train_df_cv.to_csv(os.path.join(output_dir, "train_cv.csv"), index=False)

    # Fase 2: Inferência OOF
    # 3. generate_oof_predictions
    print(f"Step 3: Generating OOF predictions (top_k={top_k})...")
    oof_df = generate_oof_predictions(train_df_cv, search_engine, top_k=top_k)
    oof_df.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    # Fase 3: Calibração Numérica
    # 4. optimize_threshold
    print("Step 4: Optimizing threshold...")
    optimal_threshold = optimize_threshold(oof_df, train_df_cv, verbose=True)

    return optimal_threshold
