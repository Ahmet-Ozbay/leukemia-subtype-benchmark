"""
Invariant, unit, and regression test suite for multi-dataset preprocessing.
Verifies data contracts, quality controls, and harmonization for:
- BaseDatasetProcessor transformations (clipping, median imputation, schema validation)
- Golub et al. (1999) primary benchmark splits
- Armstrong et al. (2002) / GSE28497 external cohort
- MILE Study / GSE13159 multi-class expansion cohort
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.config import (
    EXPRESSION_CLIP_MAX,
    EXPRESSION_CLIP_MIN,
    LABEL_MAP,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.data.make_dataset import (
    DATASET_REGISTRY,
    BaseDatasetProcessor,
    process_dataset,
)


def _resolve_probe_path(name: str) -> Path:
    cand = PROBE_DATA_DIR / name
    if cand.exists():
        return cand
    return PROCESSED_DATA_DIR / name


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def golub_datasets() -> Dict[str, pd.DataFrame]:
    """Loads processed Golub Parquet files if present on disk."""
    train_path = _resolve_probe_path("golub_train_38.parquet")
    test_path = _resolve_probe_path("golub_test_34.parquet")
    full_path = _resolve_probe_path("golub_full_72.parquet")

    if not (train_path.exists() and test_path.exists() and full_path.exists()):
        pytest.skip("Golub processed artifacts missing. Run make_dataset first.")

    return {
        "train": pd.read_parquet(train_path),
        "test": pd.read_parquet(test_path),
        "full": pd.read_parquet(full_path),
    }


@pytest.fixture(scope="module")
def armstrong_dataset() -> pd.DataFrame:
    """Loads processed Armstrong Parquet file if present on disk."""
    armstrong_path = _resolve_probe_path("armstrong_full.parquet")
    if not armstrong_path.exists():
        pytest.skip("Armstrong processed artifact missing. Run make_dataset first.")

    return pd.read_parquet(armstrong_path)


@pytest.fixture(scope="module")
def mile_dataset() -> pd.DataFrame:
    """Loads processed MILE Parquet file if present on disk."""
    mile_path = _resolve_probe_path("mile_full.parquet")
    if not mile_path.exists():
        pytest.skip("MILE processed artifact missing. Run make_dataset first.")

    return pd.read_parquet(mile_path)


# ==============================================================================
# 1. Base Processor Unit Tests (Pure Logic, No Filesystem Dependency)
# ==============================================================================
class _DummyProcessor(BaseDatasetProcessor):
    """Concrete subclass for testing base processor functionality."""

    def name(self) -> str:
        return "dummy"

    def process(self) -> Dict[str, pd.DataFrame]:
        return {}


def test_clean_microarray_matrix_quality_controls():
    """
    Unit test: Verifies that clean_microarray_matrix quality controls:
    1. Clip raw expression intensities to [clip_min, clip_max].
    2. Replace inf, -inf, and NaNs with column-wise (probe-wise) medians.
    3. Coerce numeric strings to numeric float values.
    """
    processor = _DummyProcessor()

    raw_df = pd.DataFrame(
        {
            "probe_A": [10.0, 50.0, np.nan, 20000.0],
            "probe_B": [np.inf, 100.0, -np.inf, 500.0],
            "probe_C": ["1000", "2000", "3000", "4000"],
        }
    )

    cleaned = processor.clean_microarray_matrix(
        raw_df, clip_min=20.0, clip_max=16000.0
    )

    # probe_A: 10.0 clipped to 20.0, 20000.0 clipped to 16000.0
    # non-NaN values are [20.0, 50.0, 16000.0], median is 50.0 -> NaN imputed with 50.0
    assert cleaned["probe_A"].iloc[0] == 20.0
    assert cleaned["probe_A"].iloc[1] == 50.0
    assert cleaned["probe_A"].iloc[2] == 50.0
    assert cleaned["probe_A"].iloc[3] == 16000.0

    # probe_B: +/-inf become NaN; non-NaNs are [100.0, 500.0], median is 300.0
    assert cleaned["probe_B"].iloc[0] == 300.0
    assert cleaned["probe_B"].iloc[1] == 100.0
    assert cleaned["probe_B"].iloc[2] == 300.0
    assert cleaned["probe_B"].iloc[3] == 500.0

    # probe_C: string values successfully coerced to float
    assert np.allclose(cleaned["probe_C"].values, [1000.0, 2000.0, 3000.0, 4000.0])
    assert not cleaned.isna().any().any()


def test_validate_processed_matrix_invariants():
    """
    Unit test: Verifies that validate_processed_matrix enforces schema contracts:
    Raises AssertionError on missing columns, unlabeled instances, NaNs, Infs,
    or out-of-bound values.
    """
    processor = _DummyProcessor()

    valid_df = pd.DataFrame(
        {
            "gene_1": [100.0, 200.0],
            "gene_2": [50.0, 150.0],
            "label": [0, 1],
            "subtype": ["ALL", "AML"],
        }
    )

    # Valid schema passes without error
    processor.validate_processed_matrix(valid_df)

    # Missing label column
    with pytest.raises(AssertionError, match="missing mandatory 'label' column"):
        processor.validate_processed_matrix(valid_df.drop(columns=["label"]))

    # Missing subtype column
    with pytest.raises(AssertionError, match="missing mandatory 'subtype' column"):
        processor.validate_processed_matrix(valid_df.drop(columns=["subtype"]))

    # Unlabeled samples (NaN in label)
    unlabeled_df = valid_df.copy()
    unlabeled_df.loc[0, "label"] = np.nan
    with pytest.raises(AssertionError, match="Found unlabeled samples"):
        processor.validate_processed_matrix(unlabeled_df)

    # Feature matrix contains NaN
    nan_df = valid_df.copy()
    nan_df.loc[0, "gene_1"] = np.nan
    with pytest.raises(AssertionError, match="Sanitized feature space contains NaNs"):
        processor.validate_processed_matrix(nan_df)

    # Value below lower clipping threshold
    low_df = valid_df.copy()
    low_df.loc[0, "gene_1"] = EXPRESSION_CLIP_MIN - 1.0
    with pytest.raises(AssertionError, match="Values below minimum clipping threshold"):
        processor.validate_processed_matrix(low_df)

    # Value above upper clipping threshold
    high_df = valid_df.copy()
    high_df.loc[0, "gene_1"] = EXPRESSION_CLIP_MAX + 1.0
    with pytest.raises(AssertionError, match="Values above maximum clipping threshold"):
        processor.validate_processed_matrix(high_df)


# ==============================================================================
# 2. Architectural & Registry Invariant Tests
# ==============================================================================
def test_dataset_registry_conformance():
    """Verifies that all expected processors are registered and inherit from BaseDatasetProcessor."""
    expected_cohorts = {"golub", "armstrong", "mile"}
    assert expected_cohorts.issubset(set(DATASET_REGISTRY.keys())), (
        f"Missing expected registered cohorts: {expected_cohorts - set(DATASET_REGISTRY.keys())}"
    )

    for name, proc_cls in DATASET_REGISTRY.items():
        assert issubclass(
            proc_cls, BaseDatasetProcessor
        ), f"Processor '{name}' does not inherit from BaseDatasetProcessor."
        proc_instance = proc_cls()
        assert proc_instance.name() == name, (
            f"Processor.name() '{proc_instance.name()}' does not match registry key '{name}'."
        )


def test_process_dataset_unknown_cohort():
    """Verifies that requesting an unregistered dataset raises KeyError."""
    with pytest.raises(KeyError, match="Dataset 'non_existent_dataset' is not registered"):
        process_dataset("non_existent_dataset")


# ==============================================================================
# 3. Golub-Specific Invariant & Integrity Tests (Benchmark Core)
# ==============================================================================
def test_golub_dimensions(golub_datasets):
    """
    Verifies Golub sample counts and probe dimensionality:
    7,129 gene probes + 2 metadata columns ('label', 'subtype') = 7,131 columns.
    """
    expected_cols = 7131
    assert golub_datasets["train"].shape == (38, expected_cols)
    assert golub_datasets["test"].shape == (34, expected_cols)
    assert golub_datasets["full"].shape == (72, expected_cols)


def test_golub_no_sample_leakage(golub_datasets):
    """Ensures train and test partitions have completely disjoint patient IDs."""
    train_ids = set(golub_datasets["train"].index)
    test_ids = set(golub_datasets["test"].index)
    assert train_ids.isdisjoint(test_ids), "Data leakage: Overlapping patient IDs found!"
    assert len(train_ids) == 38, "Train set sample count mismatch."
    assert len(test_ids) == 34, "Test set sample count mismatch."


def test_golub_class_distributions(golub_datasets):
    """
    Validates Golub ground-truth class distribution:
    Train (38): 27 ALL (0), 11 AML (1)
    Test (34):  20 ALL (0), 14 AML (1)
    """
    train_counts = golub_datasets["train"]["label"].value_counts().to_dict()
    test_counts = golub_datasets["test"]["label"].value_counts().to_dict()

    assert train_counts[LABEL_MAP["ALL"]] == 27, f"Expected 27 ALL in train, found {train_counts.get(0)}"
    assert train_counts[LABEL_MAP["AML"]] == 11, f"Expected 11 AML in train, found {train_counts.get(1)}"
    assert test_counts[LABEL_MAP["ALL"]] == 20, f"Expected 20 ALL in test, found {test_counts.get(0)}"
    assert test_counts[LABEL_MAP["AML"]] == 14, f"Expected 14 AML in test, found {test_counts.get(1)}"


def test_golub_sanitization_and_bounds(golub_datasets):
    """Verifies that infinite values, NaNs, and out-of-bound intensities are cleaned."""
    feature_cols = [c for c in golub_datasets["full"].columns if c not in ["label", "subtype"]]
    X = golub_datasets["full"][feature_cols].values

    assert not np.isnan(X).any(), "Processed Golub dataset contains NaN values."
    assert not np.isinf(X).any(), "Processed Golub dataset contains Infinite values."
    assert X.min() >= EXPRESSION_CLIP_MIN, f"Values below minimum clipping bound {EXPRESSION_CLIP_MIN}."
    assert X.max() <= EXPRESSION_CLIP_MAX, f"Values above maximum clipping bound {EXPRESSION_CLIP_MAX}."


def test_golub_feature_alignment_across_splits(golub_datasets):
    """
    Critical ML benchmark invariant:
    Verifies that train, test, and full datasets have exactly identical feature columns
    in the exact same order (preventing silent feature permutation / misalignment).
    """
    train_feats = [c for c in golub_datasets["train"].columns if c not in ["label", "subtype"]]
    test_feats = [c for c in golub_datasets["test"].columns if c not in ["label", "subtype"]]
    full_feats = [c for c in golub_datasets["full"].columns if c not in ["label", "subtype"]]

    assert len(train_feats) == 7129, f"Expected 7129 feature columns, got {len(train_feats)}."
    assert train_feats == test_feats, "Train and test feature columns do not match in name or order!"
    assert train_feats == full_feats, "Train and full feature columns do not match in name or order!"


def test_golub_split_sample_fidelity(golub_datasets):
    """
    Verifies that feature matrices in train and test partitions match the full combined
    dataset exactly for corresponding sample IDs.
    """
    train_df = golub_datasets["train"]
    test_df = golub_datasets["test"]
    full_df = golub_datasets["full"]

    feature_cols = [c for c in full_df.columns if c not in ["label", "subtype"]]

    pd.testing.assert_frame_equal(
        train_df[feature_cols],
        full_df.loc[train_df.index, feature_cols],
        check_dtype=True,
    )
    pd.testing.assert_frame_equal(
        test_df[feature_cols],
        full_df.loc[test_df.index, feature_cols],
        check_dtype=True,
    )
    assert set(full_df.index) == set(train_df.index) | set(test_df.index)
    assert full_df.index.is_unique, "Duplicate patient IDs detected in full Golub dataset."


def test_golub_label_encoding_and_types(golub_datasets):
    """
    Verifies label encoding integrity and column types:
    - 'label' is an integer type (0 or 1).
    - 'subtype' strictly contains 'ALL' and 'AML'.
    - 'label' exactly mirrors 'subtype' mapped through LABEL_MAP.
    - All feature columns are numeric.
    """
    for split_name, df in golub_datasets.items():
        assert np.issubdtype(df["label"].dtype, np.integer), (
            f"'label' column in {split_name} must be integer type."
        )
        assert set(df["subtype"].unique()).issubset({"ALL", "AML"}), (
            f"Unexpected subtypes in {split_name}: {df['subtype'].unique()}"
        )
        expected_labels = df["subtype"].map(LABEL_MAP)
        assert (df["label"] == expected_labels).all(), (
            f"Label values in {split_name} do not match LABEL_MAP mapping."
        )


def test_golub_no_metadata_leaked_in_features(golub_datasets):
    """
    Verifies that raw microarray call flags ('P', 'A', 'M', 'Call')
    and annotation strings ('Gene Description') are never leaked into the feature space.
    """
    for split_name, df in golub_datasets.items():
        feature_cols = [c for c in df.columns if c not in ["label", "subtype"]]
        for col in feature_cols:
            col_lower = str(col).lower()
            assert "description" not in col_lower, (
                f"Annotation column '{col}' leaked into feature space in {split_name}."
            )
            assert not col_lower.startswith("call"), (
                f"Call flag column '{col}' leaked into feature space in {split_name}."
            )


# ==============================================================================
# 4. Armstrong et al. (GSE28497) Invariant Tests (Rubric A3 & F3)
# ==============================================================================
def test_armstrong_contract_and_sanitization(armstrong_dataset):
    """Verifies data hygiene and metadata alignment for the Armstrong cohort."""
    df = armstrong_dataset

    # Verify metadata columns
    assert "label" in df.columns, "Armstrong dataset missing 'label' column."
    assert "subtype" in df.columns, "Armstrong dataset missing 'subtype' column."
    assert not df["label"].isna().any(), "Armstrong dataset contains unassigned labels."

    # Verify binary target classes
    unique_labels = set(df["label"].unique())
    assert unique_labels.issubset({0, 1}), f"Non-binary labels detected: {unique_labels}"
    assert np.issubdtype(df["label"].dtype, np.integer), "'label' must be integer type."

    # Verify clinical sample count (GSE28497 contains 48-72 pediatric samples)
    assert len(df) >= 40, f"Unexpectedly small sample size: {len(df)} samples."

    # Verify high-dimensional microarray feature space
    feature_cols = [c for c in df.columns if c not in ["label", "subtype"]]
    assert len(feature_cols) >= 5000, f"Microarray probe count unexpectedly low: {len(feature_cols)}"

    # Verify expression value bounds and hygiene
    X = df[feature_cols].values
    assert not np.isnan(X).any(), "Armstrong expression matrix contains NaNs."
    assert not np.isinf(X).any(), "Armstrong expression matrix contains Infs."
    assert X.min() >= EXPRESSION_CLIP_MIN, f"Armstrong intensities below lower bound {EXPRESSION_CLIP_MIN}."
    assert X.max() <= EXPRESSION_CLIP_MAX, f"Armstrong intensities above upper bound {EXPRESSION_CLIP_MAX}."


def test_armstrong_unique_samples(armstrong_dataset):
    """Ensures GSM sample accession identifiers are unique."""
    assert armstrong_dataset.index.is_unique, "Duplicate sample IDs detected in Armstrong dataset."


# ==============================================================================
# 5. MILE Study (GSE13159) Invariant Tests (Bonus Cohort)
# ==============================================================================
def test_mile_contract_and_sanitization(mile_dataset):
    """Verifies data hygiene and binary target alignment for the MILE cohort."""
    df = mile_dataset

    # Verify metadata columns
    assert "label" in df.columns, "MILE dataset missing 'label' column."
    assert "subtype" in df.columns, "MILE dataset missing 'subtype' column."
    assert not df["label"].isna().any(), "MILE dataset contains unassigned labels."

    # Verify binary target classes
    unique_labels = set(df["label"].unique())
    assert unique_labels.issubset({0, 1}), f"Non-binary labels detected in MILE: {unique_labels}"
    assert np.issubdtype(df["label"].dtype, np.integer), "'label' must be integer type in MILE."

    # Verify expression value bounds and hygiene
    feature_cols = [c for c in df.columns if c not in ["label", "subtype"]]
    assert len(feature_cols) >= 5000, f"Probe count unexpectedly low in MILE: {len(feature_cols)}"

    X = df[feature_cols].values
    assert not np.isnan(X).any(), "MILE expression matrix contains NaNs."
    assert not np.isinf(X).any(), "MILE expression matrix contains Infs."
    assert X.min() >= EXPRESSION_CLIP_MIN, f"MILE intensities below {EXPRESSION_CLIP_MIN}."
    assert X.max() <= EXPRESSION_CLIP_MAX, f"MILE intensities above {EXPRESSION_CLIP_MAX}."


def test_mile_unique_samples(mile_dataset):
    """Ensures GSM accession identifiers are unique in MILE."""
    assert mile_dataset.index.is_unique, "Duplicate sample IDs detected in MILE dataset."


# ==============================================================================
# 6. Modular Architecture & Invariant Tests
# ==============================================================================
def test_modular_processors_package_exports():
    """Verifies that src.data.processors and src.data expose identical objects to make_dataset."""
    import src.data as data_pkg
    import src.data.make_dataset as make_dataset_mod
    import src.data.processors as proc_pkg

    symbols = [
        "BaseDatasetProcessor",
        "GolubProcessor",
        "ArmstrongProcessor",
        "MileProcessor",
        "DATASET_REGISTRY",
        "process_dataset",
        "process_all_available",
    ]

    for sym in symbols:
        mod_obj = getattr(make_dataset_mod, sym)
        proc_obj = getattr(proc_pkg, sym)
        pkg_obj = getattr(data_pkg, sym)

        assert mod_obj is proc_obj, f"Symbol '{sym}' differs between make_dataset and processors."
        assert mod_obj is pkg_obj, f"Symbol '{sym}' differs between make_dataset and src.data."


def test_geo_utils_locate_soft_missing(tmp_path):
    """Verifies locate_geo_soft_file raises informative FileNotFoundError when archive is missing."""
    from src.data.processors.geo_utils import locate_geo_soft_file

    empty_dir = tmp_path / "empty_cohort"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="GSE99999.*Run 'python -m src.data.download_data --dataset test'"):
        locate_geo_soft_file(
            search_dirs=[empty_dir],
            pattern="GSE99999*.soft*",
            cohort_name="GSE99999",
            download_arg="test",
        )