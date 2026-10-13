"""
Extensible data preprocessing pipeline for Leukemia Subtype Benchmark.
Standardizes heterogeneous cohorts (Golub CSVs, GEO SOFT archives) into unified
sample-by-gene Parquet matrices with strict binary ALL (0) vs. AML (1) labels.

Acts as the primary CLI entrypoint and orchestrator facade for src.data.processors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Enable direct script execution: python src/data/make_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    "main",
]


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