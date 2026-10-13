"""
Unit, invariant, and regression tests for probe-to-gene symbol mapping and harmonization.
Verifies:
1. Complete probe coverage for Golub (GPL80) and Armstrong (GPL96).
2. Ground-truth clinical biomarker alignment (CST3, CD33, CD19, MPO, MME/CD10, LYZ).
3. Zero-leakage invariant for ProbeCollapser transformer inside scikit-learn pipelines.
4. Correct aggregation behavior across max_variance, mean, and median strategies.
5. Schema and dimension invariants for harmonized cross-platform Parquet matrices.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ANNOTATIONS_DIR,
    GENE_DATA_DIR,
    HARMONIZED_DATA_DIR,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)
from src.data import load_dataset
from src.data.annotations import (
    KEY_BIOMARKERS,
    ProbeCollapser,
    collapse_dataset,
    extract_armstrong_annotations,
    extract_golub_annotations,
    generate_harmonization_report,
    get_shared_gene_symbols,
    harmonize_cohort_genes,
    load_annotation_mapping,
)


@pytest.fixture(scope="module")
def golub_mapping() -> pd.DataFrame:
    """Fixture providing Golub GPL80 annotation table."""
    return load_annotation_mapping("golub", annotations_dir=ANNOTATIONS_DIR)


@pytest.fixture(scope="module")
def armstrong_mapping() -> pd.DataFrame:
    """Fixture providing Armstrong GPL96 annotation table."""
    return load_annotation_mapping("armstrong", annotations_dir=ANNOTATIONS_DIR)


# ==============================================================================
# 1. Platform Annotation & Schema Invariants
# ==============================================================================
def test_golub_annotation_coverage_and_schema(golub_mapping):
    """
    Verifies that:
    - GPL80 table contains exactly 7,129 probe sets.
    - 100% of feature probes in processed Golub parquet are present.
    - Schema adheres to ['probe_id', 'gene_symbol', 'entrez_id', 'is_mapped', ...].
    - Affymetrix endogenous controls are flagged as control and unmapped.
    """
    assert len(golub_mapping) == 7129, f"Expected 7129 Golub probes, got {len(golub_mapping)}"

    expected_cols = {
        "probe_id",
        "gene_symbol",
        "entrez_id",
        "gene_title",
        "raw_symbol",
        "is_multi_mapped",
        "is_control",
        "is_mapped",
    }
    assert expected_cols.issubset(set(golub_mapping.columns))

    # Control probe validation
    controls = golub_mapping[golub_mapping["probe_id"].str.startswith("AFFX-")]
    assert len(controls) > 0
    assert controls["is_control"].all()
    assert not controls["is_mapped"].any()

    # Invariant with processed dataset if present
    golub_full_p = PROBE_DATA_DIR / "golub_full_72.parquet"
    if not golub_full_p.exists():
        golub_full_p = PROCESSED_DATA_DIR / "golub_full_72.parquet"
    if golub_full_p.exists():
        full_df = pd.read_parquet(golub_full_p)
        features = [c for c in full_df.columns if c not in ["label", "subtype"]]
        assert len(features) == 7129
        diff = set(features) - set(golub_mapping["probe_id"])
        assert len(diff) == 0, f"Unmapped feature probes found in Golub: {diff}"


def test_armstrong_annotation_coverage_and_schema(armstrong_mapping):
    """
    Verifies that:
    - GPL96 table contains exactly 22,283 probe sets.
    - 100% of feature probes in processed Armstrong parquet are present.
    - Control probes are flagged as control and unmapped.
    """
    assert len(armstrong_mapping) == 22283, f"Expected 22283 Armstrong probes, got {len(armstrong_mapping)}"

    controls = armstrong_mapping[armstrong_mapping["probe_id"].str.startswith("AFFX-")]
    assert len(controls) > 0
    assert controls["is_control"].all()
    assert not controls["is_mapped"].any()

    arm_full_p = PROBE_DATA_DIR / "armstrong_full.parquet"
    if not arm_full_p.exists():
        arm_full_p = PROCESSED_DATA_DIR / "armstrong_full.parquet"
    if arm_full_p.exists():
        full_df = pd.read_parquet(arm_full_p)
        features = [c for c in full_df.columns if c not in ["label", "subtype"]]
        assert len(features) == 22283
        diff = set(features) - set(armstrong_mapping["probe_id"])
        assert len(diff) == 0, f"Unmapped feature probes found in Armstrong: {diff}"


# ==============================================================================
# 2. Canonical Leukemia Biomarker Cross-Platform Alignment
# ==============================================================================
def test_key_biomarkers_mapped_in_both_cohorts(golub_mapping, armstrong_mapping):
    """
    Validates that core clinical biomarkers (CST3, CD33, CD19, MPO, FAH, MME/CD10, LYZ)
    have valid probe mappings in both platforms.
    """
    for bm in KEY_BIOMARKERS:
        sym = bm["gene_symbol"]
        g_probes = golub_mapping[golub_mapping["gene_symbol"] == sym]["probe_id"].tolist()
        a_probes = armstrong_mapping[armstrong_mapping["gene_symbol"] == sym]["probe_id"].tolist()

        assert len(g_probes) >= 1, f"Biomarker '{sym}' missing in Golub GPL80 mapping."
        assert len(a_probes) >= 1, f"Biomarker '{sym}' missing in Armstrong GPL96 mapping."

    # Verify specific known probe IDs from literature
    # Golub: CST3 is M27891_at, CD33 is M23197_at
    cst3_golub = golub_mapping[golub_mapping["gene_symbol"] == "CST3"]["probe_id"].tolist()
    assert "M27891_at" in cst3_golub

    cd33_golub = golub_mapping[golub_mapping["gene_symbol"] == "CD33"]["probe_id"].tolist()
    assert "M23197_at" in cd33_golub


def test_shared_gene_overlap(golub_mapping, armstrong_mapping):
    """
    Verifies that the shared HUGO gene intersection between Golub and Armstrong
    exceeds 5,000 genes (actual ~5,426 with primary symbol selection).
    """
    shared = get_shared_gene_symbols(golub_mapping, armstrong_mapping)
    assert len(shared) >= 5000, f"Expected >= 5000 shared genes, got {len(shared)}"
    assert all(isinstance(s, str) and len(s) > 0 for s in shared)


# ==============================================================================
# 3. Zero-Leakage ProbeCollapser Transformer Tests
# ==============================================================================
def test_probe_collapser_zero_leakage_guarantee():
    """
    Mathematical leakage test:
    Constructs a synthetic dataset with two candidate probes for gene 'GENE_X':
    - In Train: Probe_A has variance 100.0, Probe_B has variance 0.1.
    - In Test:  Probe_B has variance 10000.0, Probe_A has variance 0.0.

    When ProbeCollapser.fit() is called strictly on Train, it MUST select Probe_A.
    When transform() is called on Test, it MUST use Probe_A (completely ignoring Test variances).
    """
    mapping = pd.DataFrame(
        {
            "probe_id": ["probe_A", "probe_B", "probe_C"],
            "gene_symbol": ["GENE_X", "GENE_X", "GENE_Y"],
            "is_mapped": [True, True, True],
            "is_multi_mapped": [False, False, False],
        }
    )

    X_train = pd.DataFrame(
        {
            "probe_A": [10.0, 30.0],  # var = 200.0
            "probe_B": [5.0, 5.1],   # var = 0.005
            "probe_C": [1.0, 2.0],
        },
        index=["train_1", "train_2"],
    )

    X_test = pd.DataFrame(
        {
            "probe_A": [20.0, 20.0],   # var in test = 0.0
            "probe_B": [100.0, 1000.0], # huge var in test
            "probe_C": [3.0, 4.0],
        },
        index=["test_1", "test_2"],
    )

    collapser = ProbeCollapser(mapping_df=mapping, strategy="max_variance")
    collapser.fit(X_train)

    assert collapser.selected_probes_["GENE_X"] == "probe_A"

    transformed_test = collapser.transform(X_test)
    assert "GENE_X" in transformed_test.columns
    # Transformed test GENE_X values must match probe_A, NOT probe_B
    assert np.allclose(transformed_test["GENE_X"].values, [20.0, 20.0])


def test_probe_collapser_strategies():
    """Verifies max_variance, mean, and median aggregation mechanics."""
    mapping = pd.DataFrame(
        {
            "probe_id": ["p1", "p2", "p3"],
            "gene_symbol": ["GENE_A", "GENE_A", "GENE_B"],
            "is_mapped": [True, True, True],
        }
    )

    X = pd.DataFrame(
        {
            "p1": [10.0, 30.0],
            "p2": [20.0, 20.0],
            "p3": [5.0, 15.0],
        },
        index=["s1", "s2"],
    )

    # Strategy: Mean
    c_mean = ProbeCollapser(mapping_df=mapping, strategy="mean").fit(X)
    res_mean = c_mean.transform(X)
    # GENE_A mean for s1: (10 + 20)/2 = 15; for s2: (30 + 20)/2 = 25
    assert np.allclose(res_mean["GENE_A"].values, [15.0, 25.0])
    assert np.allclose(res_mean["GENE_B"].values, [5.0, 15.0])

    # Strategy: Median
    c_median = ProbeCollapser(mapping_df=mapping, strategy="median").fit(X)
    res_median = c_median.transform(X)
    assert np.allclose(res_median["GENE_A"].values, [15.0, 25.0])


def test_probe_collapser_pipeline_integration():
    """Verifies that ProbeCollapser integrates seamlessly in an sklearn Pipeline."""
    mapping = pd.DataFrame(
        {
            "probe_id": [f"probe_{i}" for i in range(10)],
            "gene_symbol": [f"GENE_{i // 2}" for i in range(10)],
            "is_mapped": [True] * 10,
        }
    )

    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(20, 10), columns=[f"probe_{i}" for i in range(10)])
    y_train = np.random.choice([0, 1], size=20)

    X_test = pd.DataFrame(np.random.randn(5, 10), columns=[f"probe_{i}" for i in range(10)])

    pipe = Pipeline(
        [
            ("collapse", ProbeCollapser(mapping_df=mapping, strategy="max_variance")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_collapse_dataset_preserves_metadata():
    """Verifies collapse_dataset retains label and subtype columns intact."""
    mapping = pd.DataFrame(
        {
            "probe_id": ["p1", "p2"],
            "gene_symbol": ["G1", "G2"],
            "is_mapped": [True, True],
        }
    )

    train_df = pd.DataFrame(
        {
            "p1": [10.0, 20.0],
            "p2": [30.0, 40.0],
            "label": [0, 1],
            "subtype": ["ALL", "AML"],
        },
        index=["p_1", "p_2"],
    )

    c_train, _ = collapse_dataset(train_df=train_df, mapping_df=mapping)
    assert "label" in c_train.columns
    assert "subtype" in c_train.columns
    assert list(c_train["label"]) == [0, 1]
    assert list(c_train["subtype"]) == ["ALL", "AML"]


# ==============================================================================
# 4. End-to-End Harmonization Invariants
# ==============================================================================
def test_harmonize_cohort_genes_live(golub_mapping, armstrong_mapping):
    """
    Tests end-to-end cohort harmonization on processed datasets if present on disk.
    Verifies that harmonized matrices have identical columns in identical order.
    """
    g_path = PROBE_DATA_DIR / "golub_full_72.parquet"
    if not g_path.exists():
        g_path = PROCESSED_DATA_DIR / "golub_full_72.parquet"

    a_path = PROBE_DATA_DIR / "armstrong_full.parquet"
    if not a_path.exists():
        a_path = PROCESSED_DATA_DIR / "armstrong_full.parquet"

    if not (g_path.exists() and a_path.exists()):
        pytest.skip("Processed Golub or Armstrong files not found on disk.")

    golub_df = pd.read_parquet(g_path)
    arm_df = pd.read_parquet(a_path)

    g_harm, a_harm = harmonize_cohort_genes(
        golub_df=golub_df,
        armstrong_df=arm_df,
        golub_mapping=golub_mapping,
        armstrong_mapping=armstrong_mapping,
        strategy="max_variance",
    )

    # Identical columns in exact identical order
    assert list(g_harm.columns) == list(a_harm.columns)

    # Sample counts preserved
    assert len(g_harm) == 72
    assert len(a_harm) == 103

    # Feature counts
    feature_cols = [c for c in g_harm.columns if c not in ["label", "subtype"]]
    assert len(feature_cols) >= 5000

    # No NaNs or Infs
    assert not g_harm[feature_cols].isna().any().any()
    assert not np.isinf(g_harm[feature_cols].to_numpy()).any()
    assert not a_harm[feature_cols].isna().any().any()
    assert not np.isinf(a_harm[feature_cols].to_numpy()).any()

    # Files must be written to HARMONIZED_DATA_DIR
    assert (HARMONIZED_DATA_DIR / "golub_harmonized_genes.parquet").exists()
    assert (HARMONIZED_DATA_DIR / "armstrong_harmonized_genes.parquet").exists()


def test_generate_harmonization_report_creates_tables(golub_mapping, armstrong_mapping):
    """Verifies that harmonization and biomarker summary tables are created."""
    tables = generate_harmonization_report(
        golub_mapping=golub_mapping,
        armstrong_mapping=armstrong_mapping,
        out_dir=TABLES_DIR,
    )

    assert "summary" in tables
    assert "biomarkers" in tables
    assert (TABLES_DIR / "gene_harmonization_summary.csv").exists()
    assert (TABLES_DIR / "gene_harmonization_summary.md").exists()
    assert (TABLES_DIR / "biomarker_probe_mapping.csv").exists()
    assert (TABLES_DIR / "biomarker_probe_mapping.md").exists()


def test_load_dataset_api():
    """
    Verifies that the unified load_dataset accessor retrieves matrices
    across probe, gene, and harmonized levels correctly.
    """
    # Probe level
    g_probes = load_dataset("golub", level="probes", split="full")
    assert g_probes.shape == (72, 7131)
    g_train = load_dataset("golub", level="probes", split="train")
    assert g_train.shape == (38, 7131)
    a_probes = load_dataset("armstrong", level="probes")
    assert a_probes.shape == (103, 22285)

    # Gene level
    g_genes = load_dataset("golub", level="genes", split="full")
    assert g_genes.shape[0] == 72
    assert "label" in g_genes.columns
    a_genes = load_dataset("armstrong", level="genes")
    assert a_genes.shape[0] == 103

    # Harmonized level
    g_harm = load_dataset("golub", level="harmonized")
    a_harm = load_dataset("armstrong", level="harmonized")
    assert list(g_harm.columns) == list(a_harm.columns)
    assert g_harm.shape == (72, 5428)
    assert a_harm.shape == (103, 5428)

    # Invalid arguments raise ValueError
    with pytest.raises(ValueError, match="Unknown cohort"):
        load_dataset("unknown_cohort")
    with pytest.raises(ValueError, match="Unknown level"):
        load_dataset("golub", level="invalid_level")
