import os
import pandas as pd
from ..data.target_engineering import create_cardinality_bins
from ..data.cv_setup import setup_cv
from ..data.oof_generation import generate_oof_predictions
from ..data.threshold_optimization import optimize_threshold
from ..data.translation import apply_translation_pipeline


def run_stage2_feature_engineering(
    df: pd.DataFrame,
    output_dir: str = "data/processed",
    filename_suffix: str = "train_cv_translated",
) -> pd.DataFrame:
    """
    Etapa 2.4: Persistência do Artefato (Checkpointing).
    Executa o pipeline de tradução e salva o resultado em Parquet.
    """
    # Executa o pipeline de tradução
    df_translated = apply_translation_pipeline(df, text_column="query")

    # Cache do artefato
    os.makedirs(output_dir, exist_ok=True)

    # Save as Parquet for performance and data integrity
    output_path = os.path.join(output_dir, f"{filename_suffix}.parquet")
    try:
        df_translated.to_parquet(output_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save as Parquet ({e}). Falling back to CSV.")
        output_path = os.path.join(output_dir, f"{filename_suffix}.csv")
        df_translated.to_csv(output_path, index=False)

    print(f"Feature Engineering concluída. Artefato salvo em: {output_path}")

    return df_translated


def run_stage1_pipeline(
    train_df: pd.DataFrame,
    search_engine,
    n_splits: int = 5,
    seed: int = 42,
    top_k: int = 50,
    output_dir: str = "data/processed",
    skip_translation: bool = False,
):
    """
    Orchestrate Stage 1 & 2 of the competition pipeline.

    Args:
        train_df: Original training DataFrame.
        search_engine: Search engine instance for retrieval.
        n_splits: Number of folds for CV.
        seed: Random seed.
        top_k: Top K candidates to retrieve during OOF generation.
        output_dir: Directory to save the intermediate DataFrames.
        skip_translation: If True, uses 'query' column. If False, translates and uses 'query_de'.

    Returns:
        optimal_threshold (T*).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to save DataFrames preferring Parquet
    def save_df(df, name):
        parquet_path = os.path.join(output_dir, f"{name}.parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            csv_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            return csv_path

    # Fase 1: Preparação do CV
    # 1. Target Engineering
    print("Step 1: Creating cardinality bins...")
    train_df_binned = create_cardinality_bins(train_df, n_splits=n_splits)
    path1 = save_df(train_df_binned, "train_binned")
    print(f"Saved to {path1}")

    # 2. CV Setup
    print("Step 2: Setting up CV folds...")
    train_df_cv = setup_cv(train_df_binned, n_splits=n_splits, seed=seed)
    path2 = save_df(train_df_cv, "train_cv")
    print(f"Saved to {path2}")

    # Fase 2: Feature Engineering (Tradução)
    query_col = "query"
    if not skip_translation:
        print("Step 2.5: Translating queries (Feature Engineering)...")
        train_df_cv = run_stage2_feature_engineering(train_df_cv, output_dir=output_dir)
        query_col = "query_de"

    # Fase 3: Inferência OOF (Translingual if translated)
    # 3. generate_oof_predictions
    print(f"Step 3: Generating OOF predictions (top_k={top_k}, using {query_col})...")

    df_for_oof = train_df_cv.copy()
    if query_col != "query":
        df_for_oof["query_original"] = df_for_oof["query"]
        df_for_oof["query"] = df_for_oof[query_col]

    oof_df = generate_oof_predictions(df_for_oof, search_engine, top_k=top_k)
    path3 = save_df(oof_df, "oof_predictions")
    print(f"Saved to {path3}")

    # Fase 4: Calibração Numérica
    # 4. optimize_threshold
    print("Step 4: Optimizing threshold...")
    optimal_threshold = optimize_threshold(oof_df, train_df_cv, verbose=True)

    return optimal_threshold
