"""
Agarwal et al. (2014) / GSE51082 cohort processor.
Parses NCBI GEO GSE51082 Series Matrix (Affymetrix HG-U133A / GPL96),
extracts all-leukemia exploratory cohort (n=139) and acute leukemia modeling cohort (n=66: 29 ALL, 37 AML),
and exports clean Parquet matrices.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import LABEL_MAP, PROBE_DATA_DIR, RAW_DATA_DIR
from src.data.processors.base import BaseDatasetProcessor


class GSE51082Processor(BaseDatasetProcessor):
    """
    Parses NCBI GEO GSE51082-GPL96 Series Matrix, isolates ALL (n=29) and AML (n=37)
    for supervised classification, and preserves all 139 samples for 6-class global EDA.
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROBE_DATA_DIR):
        super().__init__(raw_dir=raw_dir, out_dir=out_dir)

    def name(self) -> str:
        return "gse51082"

    def _locate_matrix_file(self) -> Path:
        """Locates GSE51082-GPL96 series matrix archive."""
        candidates = [
            self.raw_dir / "gse51082" / "GSE51082-GPL96_series_matrix.txt.gz",
            self.raw_dir / "GSE51082-GPL96_series_matrix.txt.gz",
        ]
        for cand in candidates:
            if cand.exists() and cand.stat().st_size > 1024:
                return cand

        # Fallback to any matching pattern
        pattern_matches = list((self.raw_dir / "gse51082").glob("*GPL96*.gz"))
        if pattern_matches:
            return pattern_matches[0]

        raise FileNotFoundError(
            "GSE51082-GPL96 series matrix not found in data/raw/gse51082/. "
            "Run 'python -m src.data.download_data --dataset gse51082' first."
        )

    def process(self) -> Dict[str, pd.DataFrame]:
        print("[*] Processing Agarwal et al. (GSE51082 / GPL96)...")
        matrix_path = self._locate_matrix_file()
        print(f"    Parsing GEO Series Matrix: {matrix_path.name}...")

        # Decompress and separate metadata header from expression table
        with gzip.open(matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        meta_dict = {}
        data_lines = []
        in_data = False

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("!series_matrix_table_begin"):
                in_data = True
                continue
            if line_str.startswith("!series_matrix_table_end"):
                break

            if in_data:
                data_lines.append(line)
            elif line_str.startswith("!Sample_"):
                parts = [p.strip('"') for p in line.rstrip("\r\n").split("\t")]
                key = parts[0]
                vals = parts[1:]
                if key not in meta_dict:
                    meta_dict[key] = vals
                else:
                    n_existing = len([k for k in meta_dict if k.startswith(key)])
                    meta_dict[f"{key}_{n_existing}"] = vals

        # 1. Parse Phenotype Metadata
        geo_accessions = meta_dict["!Sample_geo_accession"]
        titles = meta_dict.get("!Sample_title", [""] * len(geo_accessions))

        raw_tissue = meta_dict.get("!Sample_characteristics_ch1", [""] * len(geo_accessions))
        tissues = [t.replace("tissue: ", "").strip() for t in raw_tissue]

        raw_disease = meta_dict.get("!Sample_characteristics_ch1_1", [""] * len(geo_accessions))
        diseases = [d.replace("risk group: ", "").strip() for d in raw_disease]

        pheno = pd.DataFrame(
            {
                "sample_id": geo_accessions,
                "title": titles,
                "tissue": tissues,
                "disease": diseases,
            },
            index=geo_accessions,
        )
        pheno.index.name = "patient_id"

        # Map to ALL vs AML binary task:
        # PRE B-ALL (17) & T-ALL (12) -> ALL (Class 0)
        # AML (37) -> AML (Class 1)
        # Other leukemias (BCLL: 41, CML: 22, MDS: 10) -> Class -1 (Excluded from modeling)
        def assign_subtype(d: str) -> str:
            if d == "AML":
                return "AML"
            elif d in ["PRE B-ALL", "T-ALL"]:
                return "ALL"
            return d

        def assign_label(d: str) -> int:
            if d == "AML":
                return LABEL_MAP["AML"]  # 1
            elif d in ["PRE B-ALL", "T-ALL"]:
                return LABEL_MAP["ALL"]  # 0
            return -1

        pheno["subtype"] = pheno["disease"].apply(assign_subtype)
        pheno["label"] = pheno["disease"].apply(assign_label)

        # 2. Parse Expression Table (probes x samples)
        expr_raw = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t", index_col=0)
        expr_raw.columns = [c.strip('"') for c in expr_raw.columns]
        expr_samples_by_probes = expr_raw.T
        expr_samples_by_probes.index.name = "patient_id"

        # Match sample ordering with phenotype
        common_samples = [s for s in pheno.index if s in expr_samples_by_probes.index]
        expr_samples_by_probes = expr_samples_by_probes.loc[common_samples]
        pheno = pheno.loc[common_samples]

        # 3. Clean Microarray Matrix (Preserve RMA log2 scale)
        cleaned_expr = self.clean_microarray_matrix(expr_samples_by_probes, is_log2=True)

        # 4. Construct Artifact 1: Full EDA Cohort (n=139 across 6 classes)
        full_eda = pd.concat(
            [cleaned_expr, pheno[["label", "subtype", "tissue", "disease", "title"]]],
            axis=1,
        )
        self.validate_processed_matrix(full_eda, is_log2=True)
        out_eda = self.out_dir / "gse51082_full_eda.parquet"
        full_eda.to_parquet(out_eda)
        print(f"    [+] Saved: {out_eda.name} (Shape: {full_eda.shape}, 6 Classes: {pheno['disease'].value_counts().to_dict()})")

        # 5. Construct Artifact 2: Modeling Cohort (n=66: 29 ALL vs 37 AML)
        aml_all_mask = pheno["subtype"].isin(["ALL", "AML"])
        modeling_pheno = pheno[aml_all_mask].copy()
        modeling_expr = cleaned_expr.loc[modeling_pheno.index]

        all_aml = pd.concat(
            [modeling_expr, modeling_pheno[["label", "subtype", "tissue"]]],
            axis=1,
        )
        self.validate_processed_matrix(all_aml, is_log2=True)
        out_modeling = self.out_dir / "gse51082_all_aml.parquet"
        all_aml.to_parquet(out_modeling)
        print(f"    [+] Saved: {out_modeling.name} (Shape: {all_aml.shape}, Subtypes: {modeling_pheno['subtype'].value_counts().to_dict()})")

        return {
            "gse51082_full_eda": full_eda,
            "gse51082_all_aml": all_aml,
        }

