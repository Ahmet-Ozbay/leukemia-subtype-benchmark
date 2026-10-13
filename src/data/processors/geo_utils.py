"""
Shared NCBI GEO utilities for parsing SOFT archives and extracting expression data.
Used by GEO-based microarray processors (e.g., Armstrong et al., MILE study).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def locate_geo_soft_file(
    search_dirs: Sequence[Path],
    pattern: str,
    cohort_name: str,
    download_arg: str,
) -> Path:
    """
    Locates a GEO SOFT archive file across candidate directories.

    Raises:
        FileNotFoundError: If no archive file matching pattern is found.
    """
    for directory in search_dirs:
        if directory.exists():
            candidates = list(directory.glob(pattern))
            if candidates:
                return candidates[0]

    dirs_str = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"{cohort_name} SOFT archive not found in {dirs_str}. "
        f"Run 'python -m src.data.download_data --dataset {download_arg}' first."
    )


def extract_geo_expression_matrix(
    gse: any,
    sample_ids: Iterable[str],
) -> pd.DataFrame:
    """
    Extracts sample probe series from a GEOparse GSE object and constructs
    a unified sample-by-probe expression DataFrame with 'patient_id' index.
    """
    expr_dict = {}
    for gsm_id in sample_ids:
        gsm = gse.gsms[gsm_id]
        series = gsm.table.set_index("ID_REF")["VALUE"]
        expr_dict[gsm_id] = pd.to_numeric(series, errors="coerce")

    raw_expr = pd.DataFrame(expr_dict).T
    raw_expr.index.name = "patient_id"
    return raw_expr
