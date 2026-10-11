"""
Extensible data preprocessing pipeline for Leukemia Subtype Benchmark.
Standardizes heterogeneous cohorts (Golub CSVs, GEO SOFT archives) into unified
sample-by-gene Parquet matrices with strict binary ALL (0) vs. AML (1) labels.
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd

from src.config import (
    EXPRESSION_CLIP_MAX,
    EXPRESSION_CLIP_MIN,
    LABEL_MAP,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    GOLUB_RAW_DIR,
)


# ==============================================================================
# Base Extensible Processor
# ==============================================================================
class BaseDatasetProcessor(ABC):
    """
    Abstract template for microarray cohort ingestion and standardization.
    Subclasses implement cohort-specific ingestion; the base handles sanitization,
    label integrity checks, and serialized output contracts.
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR):
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
    ) -> pd.DataFrame:
        """
        Applies standard biomedical microarray quality controls:
        1. Casts non-numeric tokens and infinite sensor values to NaN.
        2. Clips raw intensities to [clip_min, clip_max] to remove saturation/noise.
        3. Imputes missing values using probe-wise medians.
        """
        cleaned = expr_df.copy().replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        cleaned = cleaned.clip(lower=clip_min, upper=clip_max)

        if cleaned.isna().sum().sum() > 0:
            cleaned = cleaned.fillna(cleaned.median())

        return cleaned

    def validate_processed_matrix(self, df: pd.DataFrame) -> None:
        """Enforces runtime invariant checks before serialization."""
        assert "label" in df.columns, "Output DataFrame missing mandatory 'label' column."
        assert "subtype" in df.columns, "Output DataFrame missing mandatory 'subtype' column."
        assert not df["label"].isna().any(), "Found unlabeled samples in target column."
        
        feature_cols = [c for c in df.columns if c not in ["label", "subtype"]]
        X = df[feature_cols].to_numpy()
        assert not np.isnan(X).any(), "Sanitized feature space contains NaNs."
        assert not np.isinf(X).any(), "Sanitized feature space contains infinite values."
        assert (
            X.min() >= EXPRESSION_CLIP_MIN
        ), f"Values below minimum clipping threshold {EXPRESSION_CLIP_MIN}."
        assert (
            X.max() <= EXPRESSION_CLIP_MAX
        ), f"Values above maximum clipping threshold {EXPRESSION_CLIP_MAX}."


# ==============================================================================
# Cohort 1: Golub et al. (1999)
# ==============================================================================
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


# ==============================================================================
# Cohort 2: Armstrong et al. (2002) / GSE28497
# ==============================================================================
class ArmstrongProcessor(BaseDatasetProcessor):
    """Parses NCBI GEO GSE28497 SOFT archive, isolates ALL vs. AML, cleans intensities."""

    def name(self) -> str:
        return "armstrong"

    def process(self) -> Dict[str, pd.DataFrame]:
        import GEOparse

        print("[*] Processing Armstrong et al. (GSE28497)...")
        geo_dir = self.raw_dir / "armstrong_2002"
        candidates = list(geo_dir.glob("GSE28497*.soft*")) if geo_dir.exists() else []

        if not candidates:
            # Check top-level raw dir as fallback
            candidates = list(self.raw_dir.glob("GSE28497*.soft*"))

        if not candidates:
            raise FileNotFoundError(
                "Armstrong GSE28497 SOFT archive not found in data/raw/armstrong_2002/. "
                "Run 'python -m src.data.download_data --dataset armstrong' first."
            )

        soft_path = candidates[0]
        print(f"    Parsing GEO archive: {soft_path.name}...")
        gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
        pheno = gse.phenotype_data

        # Locate classification column across common GEO metadata fields
        target_col = None
        for col in ["characteristics_ch1", "description", "title", "source_name_ch1"]:
            if col in pheno.columns:
                content = " ".join(pheno[col].dropna().astype(str)).upper()
                if "AML" in content and "ALL" in content:
                    target_col = col
                    break
        if target_col is None:
            target_col = "title"

        def map_subtype(text: str) -> Optional[str]:
            val = str(text).upper()
            if "AML" in val:
                return "AML"
            # Strict ALL matching; ignores ambiguous / non-target classes (e.g. MLL)
            if "ALL" in val and "MLL" not in val:
                return "ALL"
            return None

        pheno["subtype"] = pheno[target_col].apply(map_subtype)
        valid_samples = pheno[pheno["subtype"].notna()].copy()
        valid_samples["label"] = valid_samples["subtype"].map(LABEL_MAP)

        print(f"    Subtypes identified: {valid_samples['subtype'].value_counts().to_dict()}")

        # Extract sample expression series (probes x samples)
        expr_dict = {}
        for gsm_id in valid_samples.index:
            gsm = gse.gsms[gsm_id]
            series = gsm.table.set_index("ID_REF")["VALUE"]
            expr_dict[gsm_id] = pd.to_numeric(series, errors="coerce")

        raw_expr = pd.DataFrame(expr_dict).T
        raw_expr.index.name = "patient_id"

        cleaned_features = self.clean_microarray_matrix(raw_expr)
        sanitized_armstrong = pd.concat([cleaned_features, valid_samples[["label", "subtype"]]], axis=1)

        self.validate_processed_matrix(sanitized_armstrong)

        out_path = self.out_dir / "armstrong_full.parquet"
        sanitized_armstrong.to_parquet(out_path)
        print(f"    [+] Saved: {out_path.name} (Shape: {sanitized_armstrong.shape})")

        return {"armstrong_full": sanitized_armstrong}


# ==============================================================================
# Cohort 3: MILE Study (GSE13159) - Architecture Ready for Expansion
# ==============================================================================
class MileProcessor(BaseDatasetProcessor):
    """Template for GSE13159 integration (Bonus B1 / B4 expansion)."""

    def name(self) -> str:
        return "mile"

    def process(self) -> Dict[str, pd.DataFrame]:
        import GEOparse

        print("[*] Processing MILE Study (GSE13159)...")
        geo_dir = self.raw_dir / "mile_study"
        candidates = list(geo_dir.glob("GSE13159*.soft*")) if geo_dir.exists() else []

        if not candidates:
            raise FileNotFoundError(
                "MILE GSE13159 SOFT archive not found. "
                "Download it first via 'python -m src.data.download_data --dataset mile'."
            )

        print("    Parsing large GEO SOFT file...")
        gse = GEOparse.get_GEO(filepath=str(candidates[0]), silent=True)
        pheno = gse.phenotype_data

        # MILE contains >2000 samples across >15 subtypes. Subsetting strictly for ALL and AML:
        def filter_mile(text: str) -> Optional[str]:
            val = str(text).upper()
            if "ACUTE MYELOID LEUKEMIA" in val or "AML" in val:
                return "AML"
            if "ACUTE LYMPHOBLASTIC LEUKEMIA" in val or "ALL" in val:
                return "ALL"
            return None

        pheno["subtype"] = pheno["characteristics_ch1"].apply(filter_mile)
        valid = pheno[pheno["subtype"].notna()].copy()
        valid["label"] = valid["subtype"].map(LABEL_MAP)

        expr_dict = {}
        for gsm_id in valid.index:
            gsm = gse.gsms[gsm_id]
            expr_dict[gsm_id] = pd.to_numeric(gsm.table.set_index("ID_REF")["VALUE"], errors="coerce")

        raw_expr = pd.DataFrame(expr_dict).T
        raw_expr.index.name = "patient_id"

        cleaned_features = self.clean_microarray_matrix(raw_expr)
        sanitized = pd.concat([cleaned_features, valid[["label", "subtype"]]], axis=1)

        self.validate_processed_matrix(sanitized)
        out_path = self.out_dir / "mile_full.parquet"
        sanitized.to_parquet(out_path)
        print(f"    [+] Saved: {out_path.name} (Shape: {sanitized.shape})")

        return {"mile_full": sanitized}


# ==============================================================================
# Processor Registry & Orchestration
# ==============================================================================
DATASET_REGISTRY: Dict[str, Type[BaseDatasetProcessor]] = {
    "golub": GolubProcessor,
    "armstrong": ArmstrongProcessor,
    "mile": MileProcessor,
}


def process_dataset(name: str) -> Dict[str, pd.DataFrame]:
    """Instantiates and executes the processor registered for a given key."""
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' is not registered. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    processor = DATASET_REGISTRY[name]()
    return processor.process()


def process_all_available() -> None:
    """Processes all registered datasets whose raw dependencies exist on disk."""
    executed = 0
    for name, proc_cls in DATASET_REGISTRY.items():
        try:
            proc = proc_cls()
            proc.process()
            executed += 1
        except FileNotFoundError as e:
            print(f"[-] Skipping '{name}': {e}")
        except Exception as e:
            print(f"[!] Failed to process '{name}': {e}", file=sys.stderr)

    if executed == 0:
        print("[!] No datasets were processed. Download raw files first.")


def main():
    parser = argparse.ArgumentParser(
        description="Standardize leukemia microarray cohorts into clean Parquet tables."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="all",
        help="Target dataset to parse: 'golub', 'armstrong', 'mile', or 'all' (default).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      LEUKEMIA SUBTYPE BENCHMARK - DATA STANDARDIZATION")
    print("=" * 65)

    if args.dataset == "all":
        process_all_available()
    else:
        process_dataset(args.dataset)

    print("\n[✓] Data preprocessing sequence finished.")


if __name__ == "__main__":
    main()