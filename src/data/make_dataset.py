"""
Data preprocessing pipeline for the Golub et al. (1999) Leukemia dataset.
Parses, cleans, and structures raw microarray probes into patient-by-gene matrices.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LABEL_MAP,
    EXPRESSION_CLIP_MIN,
    EXPRESSION_CLIP_MAX
)


def load_actual_labels(filepath: Path = None) -> pd.DataFrame:
    """Loads and standardizes ground truth labels from actual.csv."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "actual.csv"
        
    if not filepath.exists():
        raise FileNotFoundError(f"Missing labels file: {filepath}")

    labels_df = pd.read_csv(filepath)
    labels_df.columns = [col.strip().lower() for col in labels_df.columns]
    
    id_col = [col for col in labels_df.columns if "patient" in col or "id" in col][0]
    cancer_col = [col for col in labels_df.columns if "cancer" in col or "type" in col][0]
    
    df = labels_df[[id_col, cancer_col]].copy()
    df.columns = ["patient_id", "subtype"]
    df["patient_id"] = df["patient_id"].astype(str)
    
    df["label"] = df["subtype"].str.strip().map(LABEL_MAP)
    if df["label"].isna().any():
        unknowns = df[df["label"].isna()]["subtype"].unique()
        raise ValueError(f"Unrecognized class labels: {unknowns}")
        
    return df.set_index("patient_id")


def parse_expression_file(filepath: Path) -> pd.DataFrame:
    """Parses raw expression CSV, drops call flags, and transposes to patient x gene format."""
    if not filepath.exists():
        raise FileNotFoundError(f"Expression file not found: {filepath}")

    df = pd.read_csv(filepath)

    probe_col = None
    for candidate in ["Gene Accession Number", "gene_accession_number", "Probe"]:
        if candidate in df.columns:
            probe_col = candidate
            break

    if probe_col is None:
        probe_col = df.columns[1]

    patient_cols = [
        col for col in df.columns 
        if col not in ["Gene Description", probe_col] and not col.lower().startswith("call")
    ]

    gene_matrix = df[[probe_col] + patient_cols].copy().set_index(probe_col)
    patient_matrix = gene_matrix.T
    patient_matrix.index = patient_matrix.index.astype(str)
    patient_matrix.index.name = "patient_id"

    return patient_matrix.apply(pd.to_numeric, errors="coerce")


def clean_microarray_matrix(
    expr_df: pd.DataFrame, 
    clip_min: float = EXPRESSION_CLIP_MIN, 
    clip_max: float = EXPRESSION_CLIP_MAX
) -> pd.DataFrame:
    """Clips extreme/negative intensities and imputes missing probe values."""
    cleaned = expr_df.copy().replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.clip(lower=clip_min, upper=clip_max)
    
    if cleaned.isna().sum().sum() > 0:
        cleaned = cleaned.fillna(cleaned.median())
        
    return cleaned


def process_golub_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates parsing, label alignment, cleaning, and Parquet serialization."""
    print("[*] Loading ground truth labels...")
    labels = load_actual_labels()

    print("[*] Parsing train and independent test CSVs...")
    train_raw = parse_expression_file(RAW_DATA_DIR / "data_set_ALL_AML_train.csv")
    test_raw = parse_expression_file(RAW_DATA_DIR / "data_set_ALL_AML_independent.csv")

    print("[*] Aligning patient IDs with ground truth labels...")
    train_df = train_raw.join(labels[["label", "subtype"]], how="inner")
    test_df = test_raw.join(labels[["label", "subtype"]], how="inner")

    combined_df = pd.concat([train_df, test_df], axis=0)

    print(f"    -> Train partition shape: {train_df.shape} (Expected: 38 samples)")
    print(f"    -> Test partition shape:  {test_df.shape} (Expected: 34 samples)")
    print(f"    -> Combined cohort shape: {combined_df.shape} (Expected: 72 samples)")

    meta_cols = ["label", "subtype"]
    feature_cols = [c for c in combined_df.columns if c not in meta_cols]

    print("[*] Applying threshold clipping and sanitization...")
    features_cleaned = clean_microarray_matrix(combined_df[feature_cols])
    final_combined = pd.concat([features_cleaned, combined_df[meta_cols]], axis=1)
    
    out_combined = PROCESSED_DATA_DIR / "golub_full_72.parquet"
    out_train = PROCESSED_DATA_DIR / "golub_train_38.parquet"
    out_test = PROCESSED_DATA_DIR / "golub_test_34.parquet"

    final_combined.to_parquet(out_combined)
    final_combined.loc[train_df.index].to_parquet(out_train)
    final_combined.loc[test_df.index].to_parquet(out_test)

    print(f"[+] Cleaned datasets successfully saved to:\n    - {out_combined}\n    - {out_train}\n    - {out_test}")
    return final_combined.loc[train_df.index], final_combined.loc[test_df.index]


if __name__ == "__main__":
    try:
        process_golub_dataset()
    except Exception as err:
        print(f"[!] Processing failed: {err}", file=sys.stderr)
        sys.exit(1)