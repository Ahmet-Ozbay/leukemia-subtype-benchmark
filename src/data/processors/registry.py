"""
Dataset processor registry and pipeline execution orchestrators.
Maps dataset identifier strings to processor implementations and manages batch execution.
"""

from __future__ import annotations

import sys
from typing import Dict, Type

import pandas as pd

from src.data.processors.armstrong import ArmstrongProcessor
from src.data.processors.base import BaseDatasetProcessor
from src.data.processors.golub import GolubProcessor
from src.data.processors.mile import MileProcessor

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
