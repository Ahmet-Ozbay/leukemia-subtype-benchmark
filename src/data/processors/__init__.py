"""
Microarray dataset processors and harmonization pipelines.
"""

from src.data.processors.base import BaseDatasetProcessor
from src.data.processors.coustan_smith import CoustanSmithProcessor
from src.data.processors.golub import GolubProcessor
from src.data.processors.mile import MileProcessor
from src.data.processors.registry import (
    DATASET_REGISTRY,
    process_all_available,
    process_dataset,
)

__all__ = [
    "BaseDatasetProcessor",
    "GolubProcessor",
    "CoustanSmithProcessor",
    "MileProcessor",
    "DATASET_REGISTRY",
    "process_dataset",
    "process_all_available",
]
