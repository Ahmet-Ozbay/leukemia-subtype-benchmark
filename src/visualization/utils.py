"""
Internal visualization utilities and matrix parsing functions.
"""

from __future__ import annotations

from typing import Tuple
import pandas as pd


def get_feature_and_label_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates numeric gene/probe expression columns from metadata columns
    such as ['label', 'subtype', 'sample_id', 'tissue', 'disease', 'patient_id', 'title'].
    """
    meta_cols = [
        c
        for c in ["label", "subtype", "sample_id", "tissue", "disease", "patient_id", "title"]
        if c in df.columns
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].copy()
    y = df["label"].copy() if "label" in df.columns else pd.Series(dtype=object)
    return X, y

