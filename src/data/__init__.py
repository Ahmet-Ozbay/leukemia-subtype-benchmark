"""
Data acquisition, preprocessing, and cohort standardization pipelines.
"""

from src.data.annotations import (
    KEY_BIOMARKERS,
    ProbeCollapser,
    collapse_dataset,
    extract_coustan_smith_annotations,
    extract_golub_annotations,
    extract_gse51082_annotations,
    generate_harmonization_report,
    get_shared_gene_symbols,
    harmonize_cohort_genes,
    load_annotation_mapping,
)
from src.data.io import load_dataset
from src.data.processors import (
    DATASET_REGISTRY,
    BaseDatasetProcessor,
    CoustanSmithProcessor,
    GolubProcessor,
    GSE51082Processor,
    MileProcessor,
    process_all_available,
    process_dataset,
)

__all__ = [
    "BaseDatasetProcessor",
    "GolubProcessor",
    "GSE51082Processor",
    "CoustanSmithProcessor",
    "MileProcessor",
    "DATASET_REGISTRY",
    "process_dataset",
    "process_all_available",
    "extract_golub_annotations",
    "extract_gse51082_annotations",
    "extract_coustan_smith_annotations",
    "load_annotation_mapping",
    "ProbeCollapser",
    "collapse_dataset",
    "get_shared_gene_symbols",
    "harmonize_cohort_genes",
    "generate_harmonization_report",
    "KEY_BIOMARKERS",
    "load_dataset",
]
