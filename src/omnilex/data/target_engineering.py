import pandas as pd
import numpy as np


def create_cardinality_bins(df: pd.DataFrame, n_splits: int = 1) -> pd.DataFrame:
    """
    Creates cardinality bins for the gold_citations column.
    Ensures each bin has at least n_splits samples by merging small bins.

    Bins (initial):
    - 0 citations: bin 0
    - 1 citation: bin 1
    - 2-3 citations: bin 2
    - 4+ citations: bin 3

    Args:
        df: Input DataFrame with gold_citations column.
        n_splits: Minimum number of samples required per bin.

    Returns:
        DataFrame with additional citation_count and fold_bin columns.
    """
    df = df.copy()

    # Create citation_count
    df["citation_count"] = df["gold_citations"].apply(
        lambda x: (
            len([c for c in str(x).split(";") if c.strip()])
            if pd.notna(x) and str(x).strip() != ""
            else 0
        )
    )

    # Initial binning
    conditions = [
        (df["citation_count"] == 0),
        (df["citation_count"] == 1),
        (df["citation_count"] >= 2) & (df["citation_count"] <= 3),
        (df["citation_count"] >= 4),
    ]
    choices = [0, 1, 2, 3]
    df["fold_bin"] = np.select(conditions, choices, default=3)

    if n_splits <= 1:
        return df

    # Merge small bins
    # We want to ensure each bin in df['fold_bin'] has >= n_splits
    unique_bins = sorted(df["fold_bin"].unique())

    # Simple merging strategy:
    # Iterate from left to right, if a bin is too small, merge it with the next one.
    # If the last bin is too small, merge it with the previous one.

    i = 0
    while i < len(unique_bins):
        current_bin = unique_bins[i]
        count = (df["fold_bin"] == current_bin).sum()

        if count < n_splits:
            if i < len(unique_bins) - 1:
                # Merge with next bin
                next_bin = unique_bins[i + 1]
                df.loc[df["fold_bin"] == current_bin, "fold_bin"] = next_bin
                unique_bins.pop(i)
                # Don't increment i, check the newly merged bin
            elif i > 0:
                # Last bin is small, merge with previous bin
                prev_bin = unique_bins[i - 1]
                df.loc[df["fold_bin"] == current_bin, "fold_bin"] = prev_bin
                unique_bins.pop(i)
                i -= 1  # Re-check previous bin just in case (though it should be fine)
            else:
                # Only one bin left and it's small? Nothing we can do but keep it.
                break
        else:
            i += 1

    return df
