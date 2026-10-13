"""
Unified dataset accessor for Leukemia Subtype Benchmark.
Provides clean, consistent ingestion for downstream EDA, feature engineering, and modeling pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    GENE_DATA_DIR,
    HARMONIZED_DATA_DIR,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def load_dataset(
    cohort: str,
    level: str = "probes",
    split: str = "full",
) -> pd.DataFrame:
    """
    Loads a standardized Parquet dataset for a specific cohort and feature abstraction level.

    Parameters:
        cohort: Target cohort: 'golub', 'armstrong', or 'mile'.
        level: Feature abstraction level:
            - 'probes': Raw-cleaned probe-level matrix (p >> n).
            - 'genes': Cohort-specific collapsed HUGO gene-level matrix.
            - 'harmonized': Cross-platform harmonized matrix on shared gene intersection.
        split: Partition to load (primarily for Golub):
            - 'full': Combined cohort (e.g. n=72 for Golub, n=103 for Armstrong).
            - 'train': Primary training partition (e.g. n=38 for Golub).
            - 'test': Primary independent test partition (e.g. n=34 for Golub).

    Returns:
        pd.DataFrame containing feature matrix with metadata ['label', 'subtype'].
    """
    cohort = cohort.lower().strip()
    level = level.lower().strip()
    split = split.lower().strip()

    if cohort not in ("golub", "armstrong", "mile"):
        raise ValueError(f"Unknown cohort '{cohort}'. Supported: 'golub', 'armstrong', 'mile'.")

    if level not in ("probes", "genes", "harmonized"):
        raise ValueError(f"Unknown level '{level}'. Supported: 'probes', 'genes', 'harmonized'.")

    if level == "harmonized":
        filename = f"{cohort}_harmonized_genes.parquet"
        search_dirs = [HARMONIZED_DATA_DIR, PROCESSED_DATA_DIR]
    elif level == "genes":
        if cohort == "golub":
            if split == "train":
                filename = "golub_train_genes.parquet"
            elif split == "test":
                filename = "golub_test_genes.parquet"
            else:
                filename = "golub_full_genes.parquet"
        else:
            filename = f"{cohort}_full_genes.parquet"
        search_dirs = [GENE_DATA_DIR, PROCESSED_DATA_DIR]
    elif level == "probes":
        if cohort == "golub":
            if split == "train":
                filename = "golub_train_38.parquet"
            elif split == "test":
                filename = "golub_test_34.parquet"
            else:
                filename = "golub_full_72.parquet"
        else:
            filename = f"{cohort}_full.parquet"
        search_dirs = [PROBE_DATA_DIR, PROCESSED_DATA_DIR]

    for d in search_dirs:
        filepath = d / filename
        if filepath.exists():
            return pd.read_parquet(filepath)

    dirs_str = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"Dataset file '{filename}' not found in candidate directories: {dirs_str}. "
        f"Ensure make_dataset / map_genes has been executed."
    )
