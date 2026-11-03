"""
Coustan-Smith et al. (2011) / GSE28497 cohort processor.
Parses NCBI GEO GSE28497 SOFT archive (Affymetrix HG-U133A / GPL96),
extracts pediatric ALL cohort (n=103) for cross-platform harmonization,
and exports clean Parquet matrix.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.config import LABEL_MAP
from src.data.processors.base import BaseDatasetProcessor
from src.data.processors.geo_utils import (
    extract_geo_expression_matrix,
    locate_geo_soft_file,
)


class CoustanSmithProcessor(BaseDatasetProcessor):
    """Parses NCBI GEO GSE28497 SOFT archive, isolates ALL cohort (n=103), cleans intensities."""

    def name(self) -> str:
        return "coustan_smith"

    def process(self) -> Dict[str, pd.DataFrame]:
        import GEOparse

        print("[*] Processing Coustan-Smith et al. (GSE28497)...")
        soft_path = locate_geo_soft_file(
            search_dirs=[self.raw_dir / "coustan_smith", self.raw_dir / "armstrong_2002", self.raw_dir],
            pattern="GSE28497*.soft*",
            cohort_name="Coustan-Smith GSE28497",
            download_arg="coustan_smith",
        )

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
            if any(k in val for k in ["NORMAL", "NONLEUKEMIC", "CONTROL"]):
                return None
            # Exclude ambiguous MLL translocations
            if "ALL" in val and "MLL" not in val:
                return "ALL"
            return None

        pheno["subtype"] = pheno[target_col].apply(map_subtype)
        valid_samples = pheno[pheno["subtype"].notna()].copy()
        valid_samples["label"] = valid_samples["subtype"].map(LABEL_MAP)

        print(f"    Subtypes identified: {valid_samples['subtype'].value_counts().to_dict()}")

        # Extract sample expression series (probes x samples)
        raw_expr = extract_geo_expression_matrix(gse, valid_samples.index)

        cleaned_features = self.clean_microarray_matrix(raw_expr)
        sanitized_cs = pd.concat([cleaned_features, valid_samples[["label", "subtype"]]], axis=1)

        self.validate_processed_matrix(sanitized_cs)

        out_path = self.out_dir / "coustan_smith_full.parquet"
        sanitized_cs.to_parquet(out_path)
        print(f"    [+] Saved: {out_path.name} (Shape: {sanitized_cs.shape})")

        return {"coustan_smith_full": sanitized_cs}

