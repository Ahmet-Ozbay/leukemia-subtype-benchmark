"""
Golub et al. (1999) microarray cohort processor.
Standardizes primary benchmark CSV tables into train/test/full Parquet matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import GOLUB_RAW_DIR, LABEL_MAP
from src.data.processors.base import BaseDatasetProcessor


class GolubProcessor(BaseDatasetProcessor):
    """Parses Golub CSVs, maps clinical ground truth, and outputs train/test/full splits."""

    def name(self) -> str:
        return "golub"

    def _locate_file(self, filename: str) -> Path:
        candidates = [
            GOLUB_RAW_DIR / filename,
            self.raw_dir / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not locate '{filename}' in {GOLUB_RAW_DIR} or {self.raw_dir}")

    def _load_labels(self) -> pd.DataFrame:
        actual_path = self._locate_file("actual.csv")
        labels_df = pd.read_csv(actual_path)
        labels_df.columns = [col.strip().lower() for col in labels_df.columns]

        id_col = [c for c in labels_df.columns if "patient" in c or "id" in c][0]
        cancer_col = [c for c in labels_df.columns if "cancer" in c or "type" in c][0]

        df = labels_df[[id_col, cancer_col]].copy()
        df.columns = ["patient_id", "subtype"]
        df["patient_id"] = df["patient_id"].astype(str)
        df["subtype"] = df["subtype"].str.strip()
        df["label"] = df["subtype"].map(LABEL_MAP)

        if df["label"].isna().any():
            unknowns = df[df["label"].isna()]["subtype"].unique()
            raise ValueError(f"Unrecognized class labels: {unknowns}")

        return df.set_index("patient_id")

    def _parse_expression_table(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath)

        probe_col = None
        for candidate in ["Gene Accession Number", "gene_accession_number", "Probe"]:
            if candidate in df.columns:
                probe_col = candidate
                break
        if probe_col is None:
            probe_col = df.columns[1]

        # Drop annotation text and microarray call flags (P/A/M)
        sample_cols = [
            c for c in df.columns
            if c not in ["Gene Description", probe_col] and not c.lower().startswith("call")
        ]

        patient_matrix = df[[probe_col] + sample_cols].set_index(probe_col).T
        patient_matrix.index = patient_matrix.index.astype(str)
        patient_matrix.index.name = "patient_id"
        return patient_matrix

    def process(self) -> Dict[str, pd.DataFrame]:
        print("[*] Processing Golub et al. (1999)...")
        labels = self._load_labels()

        train_raw = self._parse_expression_table(self._locate_file("data_set_ALL_AML_train.csv"))
        test_raw = self._parse_expression_table(self._locate_file("data_set_ALL_AML_independent.csv"))

        train_df = train_raw.join(labels[["label", "subtype"]], how="inner")
        test_df = test_raw.join(labels[["label", "subtype"]], how="inner")
        combined_df = pd.concat([train_df, test_df], axis=0)

        meta_cols = ["label", "subtype"]
        feature_cols = [c for c in combined_df.columns if c not in meta_cols]

        cleaned_features = self.clean_microarray_matrix(combined_df[feature_cols])
        sanitized_full = pd.concat([cleaned_features, combined_df[meta_cols]], axis=1)

        sanitized_train = sanitized_full.loc[train_df.index]
        sanitized_test = sanitized_full.loc[test_df.index]

        self.validate_processed_matrix(sanitized_full)

        # Output persistence compatible with test suite
        out_full = self.out_dir / "golub_full_72.parquet"
        out_train = self.out_dir / "golub_train_38.parquet"
        out_test = self.out_dir / "golub_test_34.parquet"

        sanitized_full.to_parquet(out_full)
        sanitized_train.to_parquet(out_train)
        sanitized_test.to_parquet(out_test)

        print(f"    [+] Saved: {out_full.name} (Shape: {sanitized_full.shape})")
        print(f"    [+] Saved: {out_train.name} (Shape: {sanitized_train.shape})")
        print(f"    [+] Saved: {out_test.name} (Shape: {sanitized_test.shape})")

        return {
            "golub_full": sanitized_full,
            "golub_train": sanitized_train,
            "golub_test": sanitized_test,
        }
