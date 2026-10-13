"""
Microarray platform annotation extraction, probe collapsing, and cohort harmonization package.
"""

from src.data.annotations.collapser import ProbeCollapser, collapse_dataset
from src.data.annotations.gpl_parser import (
    clean_symbol,
    extract_armstrong_annotations,
    extract_golub_annotations,
    load_annotation_mapping,
    parse_gpl80_annotations,
    parse_gpl96_annotations,
)
from src.data.annotations.harmonizer import (
    KEY_BIOMARKERS,
    generate_harmonization_report,
    get_shared_gene_symbols,
    harmonize_cohort_genes,
)

__all__ = [
    "clean_symbol",
    "parse_gpl80_annotations",
    "parse_gpl96_annotations",
    "extract_golub_annotations",
    "extract_armstrong_annotations",
    "load_annotation_mapping",
    "ProbeCollapser",
    "collapse_dataset",
    "get_shared_gene_symbols",
    "harmonize_cohort_genes",
    "generate_harmonization_report",
    "KEY_BIOMARKERS",
]
