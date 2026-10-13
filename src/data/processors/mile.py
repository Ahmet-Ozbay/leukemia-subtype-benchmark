"""
MILE Study (GSE13159) cohort processor.
Parses NCBI GEO GSE13159 SOFT archive, subsets ALL and AML cases from multi-class study,
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


class MileProcessor(BaseDatasetProcessor):
    """Template for GSE13159 integration (Bonus B1 / B4 expansion)."""

    def name(self) -> str:
        return "mile"

    def process(self) -> Dict[str, pd.DataFrame]:
        import GEOparse

        print("[*] Processing MILE Study (GSE13159)...")
        soft_path = locate_geo_soft_file(
            search_dirs=[self.raw_dir / "mile_study", self.raw_dir],
            pattern="GSE13159*.soft*",
            cohort_name="MILE GSE13159",
            download_arg="mile",
        )

        print("    Parsing large GEO SOFT file...")
        gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
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

        raw_expr = extract_geo_expression_matrix(gse, valid.index)

        cleaned_features = self.clean_microarray_matrix(raw_expr)
        sanitized = pd.concat([cleaned_features, valid[["label", "subtype"]]], axis=1)

        self.validate_processed_matrix(sanitized)
        out_path = self.out_dir / "mile_full.parquet"
        sanitized.to_parquet(out_path)
        print(f"    [+] Saved: {out_path.name} (Shape: {sanitized.shape})")

        return {"mile_full": sanitized}
