"""
Base dataset processor abstraction and microarray preprocessing contracts.
Provides template methods and common quality-control operations (clipping,
median imputation, schema invariant validation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    EXPRESSION_CLIP_MAX,
    EXPRESSION_CLIP_MIN,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


class BaseDatasetProcessor(ABC):
    """
    Abstract template for microarray cohort ingestion and standardization.
    Subclasses implement cohort-specific ingestion; the base handles sanitization,
    label integrity checks, and serialized output contracts.
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROBE_DATA_DIR):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def name(self) -> str:
        """Unique identifier key for the dataset."""
        pass

    @abstractmethod
    def process(self) -> Dict[str, pd.DataFrame]:
        """
        Ingests and harmonizes the cohort.
        Returns a dictionary mapping partition names to standardized DataFrames.
        """
        pass

    def clean_microarray_matrix(
        self,
        expr_df: pd.DataFrame,
        clip_min: float = EXPRESSION_CLIP_MIN,
        clip_max: float = EXPRESSION_CLIP_MAX,
        is_log2: bool = False,
    ) -> pd.DataFrame:
        """
        Applies standard biomedical microarray quality controls:
        1. Casts non-numeric tokens and infinite sensor values to NaN.
        2. If is_log2 is False (linear intensities, e.g. Golub), clips raw intensities
           to [clip_min, clip_max] to remove saturation/noise.
           If is_log2 is True (e.g. RMA pre-normalized), preserves log2 scale without linear clipping.
        3. Imputes missing values using probe-wise medians.
        """
        cleaned = expr_df.copy().replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        if not is_log2:
            cleaned = cleaned.clip(lower=clip_min, upper=clip_max)

        if cleaned.isna().sum().sum() > 0:
            cleaned = cleaned.fillna(cleaned.median())

        return cleaned

    def validate_processed_matrix(
        self,
        df: pd.DataFrame,
        is_log2: bool = False,
        meta_cols: tuple[str, ...] = (
            "label",
            "subtype",
            "tissue",
            "disease",
            "patient_id",
            "sample_id",
            "title",
        ),
    ) -> None:
        """Enforces runtime invariant checks before serialization."""
        assert "label" in df.columns, "Output DataFrame missing mandatory 'label' column."
        assert "subtype" in df.columns, "Output DataFrame missing mandatory 'subtype' column."
        assert not df["label"].isna().any(), "Found unlabeled samples in target column."

        feature_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feature_cols].to_numpy(dtype=float)
        assert not np.isnan(X).any(), "Sanitized feature space contains NaNs."
        assert not np.isinf(X).any(), "Sanitized feature space contains infinite values."

        if is_log2:
            assert X.min() >= 0.0, f"Values below minimum expected log2 expression (0.0): {X.min()}."
            assert X.max() <= 25.0, f"Values above maximum expected log2 expression (25.0): {X.max()}."
        else:
            assert (
                X.min() >= EXPRESSION_CLIP_MIN
            ), f"Values below minimum clipping threshold {EXPRESSION_CLIP_MIN}."
            assert (
                X.max() <= EXPRESSION_CLIP_MAX
            ), f"Values above maximum clipping threshold {EXPRESSION_CLIP_MAX}."
