import os
import pandas as pd
import torch
from ..data.target_engineering import create_cardinality_bins
from ..data.cv_setup import setup_cv
from ..data.oof_generation import generate_oof_predictions
from ..data.threshold_optimization import optimize_threshold
from ..data.translation import apply_translation_pipeline
from ..retrieval.bm25_index import BM25Index
from ..retrieval.dense_index import DenseIndex
from ..retrieval.hybrid import HybridSearchEngine, build_text_lookup


def run_stage2_feature_engineering(
    df: pd.DataFrame,
    output_dir: str = "data/processed",
    filename_suffix: str = "train_cv_translated",
) -> pd.DataFrame:
    """Etapa 2.4: Persistência do Artefato (Checkpointing)."""
    df_translated = apply_translation_pipeline(df, text_column="query")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename_suffix}.parquet")
    try:
        df_translated.to_parquet(output_path, index=False)
    except Exception:
        output_path = os.path.join(output_dir, f"{filename_suffix}.csv")
        df_translated.to_csv(output_path, index=False)
    print(f"Feature Engineering concluída. Artefato salvo em: {output_path}")
    return df_translated


def run_stage1_pipeline(
    train_df: pd.DataFrame,
    search_engine=None,
    n_splits: int = 5,
    seed: int = 42,
    top_k: int = 50,
    top_k_rerank: int = 50,
    output_dir: str = "data/processed",
    raw_data_dir: str = "data/raw",
    skip_translation: bool = False,
    skip_indexing: bool = True,
    use_reranker: bool = True,
    model_name: str = "intfloat/multilingual-e5-small",
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
):
    """Orchestrate Stage 1, 2, 3 & 4 of the competition pipeline."""
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
        train_df_cv = pd.read_parquet(os.path.join(output_dir, "train_cv_translated.parquet"))
        query_col = "query_de"

    # Fase 3 & 4: Model Architecture (Hybrid Retrieval + Reranking)
    if search_engine is None:
        print("Step 3 & 4: Initializing Hybrid Search Engine + Reranker...")
        bm25_path = os.path.join(output_dir, "corpus_bm25.pkl")
        dense_prefix = os.path.join(output_dir, "corpus_dense")

        # 1. Load First-Stage Indices
        print("  Loading BM25 index...")
        bm25_idx = BM25Index.load(bm25_path)
        print("  Loading Dense index...")
        dense_idx = DenseIndex.load(dense_prefix)

        # 2. Setup Reranker (Lazy Loading)
        reranker = None
        text_lookup = None
        if use_reranker:
            print(f"  Initializing Reranker: {reranker_model}...")
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            reranker = CrossEncoder(reranker_model, device=device, max_length=512)

            print("  Building text lookup for lazy loading...")
            laws_path = os.path.join(raw_data_dir, "laws_de.csv")
            courts_path = os.path.join(raw_data_dir, "court_considerations.csv")
            text_lookup = build_text_lookup(laws_path, courts_path)

        search_engine = HybridSearchEngine(
            bm25_index=bm25_idx,
            dense_index=dense_idx,
            reranker=reranker,
            text_lookup=text_lookup,
        )

    # Fase 3B & 4B: Inferência OOF
    print(f"Step 3B & 4B: Generating OOF predictions (top_k={top_k}, rerank={use_reranker})...")
    df_for_oof = train_df_cv.copy()
    if query_col != "query":
        df_for_oof["query"] = df_for_oof[query_col]

    # Note: top_k_rerank can be passed via kwargs or modified in generate_oof_predictions
    # For now, we assume generate_oof_predictions calls search_engine.query()
    oof_df = generate_oof_predictions(df_for_oof, search_engine, top_k=top_k)
    save_df(oof_df, "oof_predictions")

    # Fase 4: Calibração Numérica
    print("Step 4: Optimizing threshold...")
    optimal_threshold = optimize_threshold(oof_df, train_df_cv, verbose=True)

    return optimal_threshold
