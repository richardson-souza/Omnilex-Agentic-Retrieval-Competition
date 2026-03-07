import pandas as pd
from sklearn.model_selection import StratifiedKFold


def setup_cv(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Setup Stratified K-Fold cross validation.

    Assigns a 'fold' column to the dataframe (0 to n_splits-1).

    Args:
        df: Input DataFrame with 'fold_bin' column.
        n_splits: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with an additional 'fold' column.
    """
    df = df.copy()

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize fold column
    df["fold"] = -1

    # Split into folds based on fold_bin
    # X=df, y=df['fold_bin']
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["fold_bin"])):
        df.loc[df.index[val_idx], "fold"] = fold_idx

    return df
