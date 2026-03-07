import pandas as pd
import pytest
import numpy as np
from omnilex.data.target_engineering import create_cardinality_bins


def test_create_cardinality_bins_basic():
    """Test basic binning of citation counts."""
    data = {
        "query_id": ["q1", "q2", "q3", "q4", "q5"],
        "gold_citations": [
            "",  # 0 citations -> bin 0
            "Art. 1 ZGB",  # 1 citation -> bin 1
            "Art. 1;Art. 2",  # 2 citations -> bin 2
            "A;B;C",  # 3 citations -> bin 2
            "A;B;C;D",  # 4 citations -> bin 3
        ],
    }
    df = pd.DataFrame(data)

    # Act
    df_result = create_cardinality_bins(df)

    # Assert
    assert "citation_count" in df_result.columns
    assert "fold_bin" in df_result.columns

    expected_counts = [0, 1, 2, 3, 4]
    assert df_result["citation_count"].tolist() == expected_counts

    # Basic binning without n_splits check
    expected_bins = [0, 1, 2, 2, 3]
    assert df_result["fold_bin"].tolist() == expected_bins


def test_create_cardinality_bins_nan():
    """Test handling of NaN values in gold_citations."""
    data = {"query_id": ["q1"], "gold_citations": [np.nan]}
    df = pd.DataFrame(data)

    df_result = create_cardinality_bins(df)

    assert df_result["citation_count"].iloc[0] == 0
    assert df_result["fold_bin"].iloc[0] == 0


def test_create_cardinality_bins_min_size():
    """Test that bins with fewer than n_splits samples are merged."""
    # Suppose n_splits = 3
    data = {
        "query_id": [f"q{i}" for i in range(10)],
        "gold_citations": [
            "",
            "",
            "",
            "",
            "",  # 5 samples for bin 0
            "Art. 1 ZGB",  # 1 sample for bin 1 (should be merged)
            "A;B",
            "C;D",
            "E;F",  # 3 samples for bin 2
            "A;B;C;D",  # 1 sample for bin 3 (should be merged)
        ],
    }
    df = pd.DataFrame(data)

    # Act
    df_result = create_cardinality_bins(df, n_splits=3)

    # Assert
    # Check that each unique bin has at least 3 samples
    bin_counts = df_result["fold_bin"].value_counts()
    for b, count in bin_counts.items():
        assert count >= 3, f"Bin {b} has only {count} samples, which is < n_splits (3)"
