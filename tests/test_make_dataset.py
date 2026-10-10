"""
Unit and invariant tests for the Golub data processing pipeline.
Verifies shapes, label alignment, class distributions, and preprocessing bounds.
"""

import pytest
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    EXPRESSION_CLIP_MIN,
    EXPRESSION_CLIP_MAX
)


@pytest.fixture(scope="module")
def datasets():
    """Loads processed Parquet files once for the entire test session."""
    train_path = PROCESSED_DATA_DIR / "golub_train_38.parquet"
    test_path = PROCESSED_DATA_DIR / "golub_test_34.parquet"
    full_path = PROCESSED_DATA_DIR / "golub_full_72.parquet"

    assert train_path.exists(), "golub_train_38.parquet is missing. Run make_dataset first."
    assert test_path.exists(), "golub_test_34.parquet is missing. Run make_dataset first."
    assert full_path.exists(), "golub_full_72.parquet is missing. Run make_dataset first."

    return {
        "train": pd.read_parquet(train_path),
        "test": pd.read_parquet(test_path),
        "full": pd.read_parquet(full_path)
    }


def test_dataset_dimensions(datasets):
    """Verifies sample counts and probe counts match the assignment specification."""
    # 7,129 gene probes + 2 metadata columns ('label', 'subtype') = 7,131 columns
    expected_cols = 7131
    assert datasets["train"].shape == (38, expected_cols)
    assert datasets["test"].shape == (34, expected_cols)
    assert datasets["full"].shape == (72, expected_cols)


def test_no_sample_leakage(datasets):
    """Ensures train and test sets have completely disjoint patient IDs."""
    train_ids = set(datasets["train"].index)
    test_ids = set(datasets["test"].index)
    assert train_ids.isdisjoint(test_ids), "Data leakage: overlapping patient IDs found!"


def test_class_distributions(datasets):
    """
    Checks that the class distributions match the ground truth in Golub (1999):
    Train: 27 ALL (0), 11 AML (1)
    Test:  20 ALL (0), 14 AML (1)
    """
    train_counts = datasets["train"]["label"].value_counts().to_dict()
    test_counts = datasets["test"]["label"].value_counts().to_dict()

    assert train_counts[0] == 27, f"Expected 27 ALL in train, found {train_counts.get(0)}"
    assert train_counts[1] == 11, f"Expected 11 AML in train, found {train_counts.get(1)}"
    assert test_counts[0] == 20, f"Expected 20 ALL in test, found {test_counts.get(0)}"
    assert test_counts[1] == 14, f"Expected 14 AML in test, found {test_counts.get(1)}"


def test_data_sanitization_and_bounds(datasets):
    """Verifies that infinite values, NaNs, and out-of-bound intensities are cleaned."""
    feature_cols = [c for c in datasets["full"].columns if c not in ["label", "subtype"]]
    X = datasets["full"][feature_cols].values

    assert not np.isnan(X).any(), "Processed dataset contains NaN values."
    assert not np.isinf(X).any(), "Processed dataset contains Infinite values."
    assert X.min() >= EXPRESSION_CLIP_MIN, f"Values below minimum clipping threshold {EXPRESSION_CLIP_MIN} found."
    assert X.max() <= EXPRESSION_CLIP_MAX, f"Values above maximum clipping threshold {EXPRESSION_CLIP_MAX} found."