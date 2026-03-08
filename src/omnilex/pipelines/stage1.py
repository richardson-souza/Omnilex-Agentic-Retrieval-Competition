import os
import pandas as pd
from ..data.target_engineering import create_cardinality_bins
from ..data.cv_setup import setup_cv
from ..data.oof_generation import generate_oof_predictions
from ..data.threshold_optimization import optimize_threshold
from ..data.translation import apply_translation_pipeline
from ..retrieval.bm25_index import BM25Index
from ..retrieval.dense_index import DenseIndex
from ..retrieval.hybrid import HybridSearchEngine


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
    search_engine=None,  # Optional if we load from files
    n_splits: int = 5,
    seed: int = 42,
    top_k: int = 100,
    output_dir: str = "data/processed",
    skip_translation: bool = False,
    skip_indexing: bool = True,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
):
    """
    Orchestrate Stage 1, 2 & 3 of the competition pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    print("Step 1: Creating cardinality bins...")
    train_df_binned = create_cardinality_bins(train_df, n_splits=n_splits)
    save_df(train_df_binned, "train_binned")

    print("Step 2: Setting up CV folds...")
    train_df_cv = setup_cv(train_df_binned, n_splits=n_splits, seed=seed)
    save_df(train_df_cv, "train_cv")

    # Fase 2: Feature Engineering (Tradução)
    query_col = "query"
    if not skip_translation:
        print("Step 2.5: Translating queries (Feature Engineering)...")
        train_df_cv = run_stage2_feature_engineering(train_df_cv, output_dir=output_dir)
        query_col = "query_de"
    elif os.path.exists(os.path.join(output_dir, "train_cv_translated.parquet")):
        print("Loading existing translated queries...")
        train_df_cv = pd.read_parquet(
            os.path.join(output_dir, "train_cv_translated.parquet")
        )
        query_col = "query_de"

    # Fase 3: Model Architecture (Retrieval Híbrido)
    if search_engine is None:
        print("Step 3: Initializing Hybrid Search Engine...")
        bm25_path = os.path.join(output_dir, "corpus_bm25.pkl")
        dense_prefix = os.path.join(output_dir, "corpus_dense")

        if not os.path.exists(bm25_path) or not os.path.exists(dense_prefix + ".index"):
            if skip_indexing:
                raise FileNotFoundError("Indices not found and skip_indexing is True.")
            # Build indices (import build_hybrid_index here to avoid circular imports if any)
            from ...utils.build_indices import build_hybrid_index

            build_hybrid_index(Path("data/raw"), Path(output_dir), model_name)

        print("  Loading BM25 index...")
        bm25_idx = BM25Index.load(bm25_path)
        print("  Loading Dense index...")
        dense_idx = DenseIndex.load(dense_prefix)
        search_engine = HybridSearchEngine(bm25_idx, dense_idx)

    # Fase 3B: Inferência OOF (Translingual)
    print(f"Step 3B: Generating OOF predictions (top_k={top_k}, using {query_col})...")
    df_for_oof = train_df_cv.copy()
    if query_col != "query":
        df_for_oof["query_original"] = df_for_oof["query"]
        df_for_oof["query"] = df_for_oof[query_col]

    oof_df = generate_oof_predictions(df_for_oof, search_engine, top_k=top_k)
    save_df(oof_df, "oof_predictions")

    # Fase 4: Calibração Numérica
    print("Step 4: Optimizing threshold...")
    optimal_threshold = optimize_threshold(oof_df, train_df_cv, verbose=True)

    return optimal_threshold
