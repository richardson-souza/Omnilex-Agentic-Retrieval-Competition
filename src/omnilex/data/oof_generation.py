import pandas as pd
import numpy as np


def generate_oof_predictions(
    df: pd.DataFrame, search_engine, top_k: int = 50
) -> pd.DataFrame:
    """
    Generate Out-Of-Fold predictions by running the search engine on validation sets.

    Args:
        df: DataFrame with 'fold', 'query_id', and 'query' columns.
        search_engine: An object with a .query(text, top_k) method.
        top_k: Number of candidates to retrieve per query.

    Returns:
        DataFrame with columns ['query_id', 'citation', 'score'].
        Scores are normalized per query using MinMax scaling.
    """
    oof_predictions = []

    # Iterate through each fold
    unique_folds = sorted(df["fold"].unique())

    for fold in unique_folds:
        val_queries = df[df["fold"] == fold]

        for _, row in val_queries.iterrows():
            query_id = row["query_id"]
            query_text = row["query"]

            # Retrieve candidates
            # search_engine.query returns a list of dicts: [{'citation': '...', 'score': ...}, ...]
            candidates = search_engine.query(query_text, top_k=top_k)

            if not candidates:
                continue

            # Perform MinMax scaling for this query's candidates
            scores = np.array([float(c["score"]) for c in candidates])

            if len(scores) > 1:
                min_score = scores.min()
                max_score = scores.max()

                if max_score > min_score:
                    norm_scores = (scores - min_score) / (max_score - min_score)
                else:
                    # All scores are equal
                    norm_scores = np.ones_like(scores)
            else:
                # Only one score, set to 1.0
                norm_scores = np.ones_like(scores)

            # Append results
            for idx, cand in enumerate(candidates):
                oof_predictions.append(
                    {
                        "query_id": query_id,
                        "citation": cand["citation"],
                        "score": float(norm_scores[idx]),
                    }
                )

    return pd.DataFrame(oof_predictions)
