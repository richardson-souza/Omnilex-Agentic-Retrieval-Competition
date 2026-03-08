import pandas as pd
import numpy as np


def compute_macro_f1(true_citations_list, pred_citations_list):
    """
    Computes Macro F1-Score for a list of ground truth and predicted citations.
    """
    f1_scores = []
    for y_true, y_pred in zip(true_citations_list, pred_citations_list):
        set_true = set(y_true)
        set_pred = set(y_pred)

        if len(set_true) == 0 and len(set_pred) == 0:
            f1_scores.append(1.0)
            continue
        if len(set_pred) == 0 or len(set_true) == 0:
            f1_scores.append(0.0)
            continue

        tp = len(set_true.intersection(set_pred))
        precision = tp / len(set_pred)
        recall = tp / len(set_true)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))

    return np.mean(f1_scores) if f1_scores else 0.0


def optimize_threshold(
    oof_df: pd.DataFrame, train_df: pd.DataFrame, verbose: bool = False
):
    """
    Perform Grid Search to find the optimal threshold for Macro F1.
    """
    # 1. Prepare ground truth
    truth_map = (
        train_df.set_index("query_id")["gold_citations"]
        .apply(
            lambda x: (
                [c.strip() for c in str(x).split(";") if c.strip()]
                if pd.notna(x) and str(x).strip() != ""
                else []
            )
        )
        .to_dict()
    )

    all_query_ids = list(truth_map.keys())
    y_true_all = [truth_map[qid] for qid in all_query_ids]

    # 2. Prepare OOF data for faster processing
    # Grouping by query_id once
    grouped_oof = (
        oof_df.groupby("query_id")
        .apply(lambda x: list(zip(x["citation"], x["score"])), include_groups=False)
        .to_dict()
    )

    thresholds = np.arange(0.1, 0.95, 0.01)

    best_t = 0.0
    best_f1 = -1.0

    for t in thresholds:
        y_pred_all = []
        for qid in all_query_ids:
            candidates = grouped_oof.get(qid, [])
            # Filter candidates by score
            # Optimization: since candidates are usually few (<100), list comprehension is fast
            preds = [c for c, s in candidates if s >= t]
            y_pred_all.append(preds)

        current_f1 = compute_macro_f1(y_true_all, y_pred_all)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_t = t

    if verbose:
        print(f"Optimal Threshold: {best_t:.3f} | Macro F1 OOF: {best_f1:.4f}")

    return best_t
