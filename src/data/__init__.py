"""
Data acquisition, preprocessing, and cohort standardization pipelines.
"""

from src.data.processors import (
    DATASET_REGISTRY,
    ArmstrongProcessor,
    BaseDatasetProcessor,
    GolubProcessor,
    MileProcessor,
    process_all_available,
    process_dataset,
)

__all__ = [
    "BaseDatasetProcessor",
    "GolubProcessor",
    "ArmstrongProcessor",
    "MileProcessor",
    "DATASET_REGISTRY",
    "process_dataset",
    "process_all_available",
]
