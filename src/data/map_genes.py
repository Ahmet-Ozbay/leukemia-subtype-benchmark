"""
CLI orchestrator for probe-to-gene symbol mapping and cross-platform cohort harmonization.

Executes:
1. Extraction and persistence of platform annotations (GPL80 for Golub, GPL96 for Armstrong).
2. Cross-platform overlap calculation and biomarker tracking tables (Rubric A3 & F5).
3. (Optional --collapse) Zero-leakage aggregation into gene-level matrices.
4. (Optional --harmonize) Alignment to the shared ~5,958 HUGO gene intersection.

Usage:
    python -m src.data.map_genes
    python -m src.data.map_genes --collapse --harmonize --strategy max_variance
    python -m src.data.map_genes --drop-multi-mapped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Enable direct script execution: python src/data/map_genes.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    ANNOTATIONS_DIR,
    GENE_DATA_DIR,
    HARMONIZED_DATA_DIR,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)
from src.data.annotations import (
    collapse_dataset,
    extract_armstrong_annotations,
    extract_golub_annotations,
    generate_harmonization_report,
    harmonize_cohort_genes,
)


def run_mapping_pipeline(
    target: str = "all",
    strategy: str = "max_variance",
    drop_multi_mapped: bool = False,
    do_collapse: bool = False,
    do_harmonize: bool = False,
) -> None:
    print("\n" + "=" * 65)
    print("  LEUKEMIA SUBTYPE BENCHMARK - PROBE-TO-GENE MAPPING & HARMONIZATION")
    print("=" * 65)

    golub_mapping = None
    armstrong_mapping = None

    # 1. Extraction of annotations
    if target in ("golub", "all"):
        print("\n[*] Step 1A: Extracting Golub et al. (Hu6800 / GPL80) annotations...")
        golub_mapping = extract_golub_annotations(
            out_dir=ANNOTATIONS_DIR,
            drop_multi_mapped=drop_multi_mapped,
        )

    if target in ("armstrong", "all"):
        print("\n[*] Step 1B: Extracting Armstrong et al. (HG-U133A / GPL96) annotations...")
        armstrong_mapping = extract_armstrong_annotations(
            out_dir=ANNOTATIONS_DIR,
            drop_multi_mapped=drop_multi_mapped,
        )

    # 2. Harmonization report and biomarker tracking
    if golub_mapping is not None and armstrong_mapping is not None:
        print("\n[*] Step 2: Generating cross-platform harmonization & biomarker reports...")
        generate_harmonization_report(
            golub_mapping=golub_mapping,
            armstrong_mapping=armstrong_mapping,
            out_dir=TABLES_DIR,
        )

    def _locate_probe_file(name: str) -> Path:
        cand = PROBE_DATA_DIR / name
        if cand.exists():
            return cand
        return PROCESSED_DATA_DIR / name

    # 3. Optional Gene-level Collapsing (Zero-Leakage)
    if do_collapse:
        print(f"\n[*] Step 3: Collapsing probe-level matrices to gene level (Strategy: {strategy})...")
        if target in ("golub", "all"):
            train_p = _locate_probe_file("golub_train_38.parquet")
            test_p = _locate_probe_file("golub_test_34.parquet")
            full_p = _locate_probe_file("golub_full_72.parquet")

            if train_p.exists() and test_p.exists():
                print("    Collapsing Golub train/test splits with train-only variance fitting (Zero Leakage)...")
                train_df = pd.read_parquet(train_p)
                test_df = pd.read_parquet(test_p)
                c_train, c_test = collapse_dataset(
                    train_df=train_df,
                    test_df=test_df,
                    mapping_df=golub_mapping,
                    strategy=strategy,
                    drop_multi_mapped=drop_multi_mapped,
                )
                c_train.to_parquet(GENE_DATA_DIR / "golub_train_genes.parquet")
                c_test.to_parquet(GENE_DATA_DIR / "golub_test_genes.parquet")
                print(f"    [+] Saved: {GENE_DATA_DIR.name}/golub_train_genes.parquet (Shape: {c_train.shape})")
                print(f"    [+] Saved: {GENE_DATA_DIR.name}/golub_test_genes.parquet (Shape: {c_test.shape})")

            if full_p.exists():
                full_df = pd.read_parquet(full_p)
                c_full, _ = collapse_dataset(
                    train_df=full_df,
                    mapping_df=golub_mapping,
                    strategy=strategy,
                    drop_multi_mapped=drop_multi_mapped,
                )
                c_full.to_parquet(GENE_DATA_DIR / "golub_full_genes.parquet")
                print(f"    [+] Saved: {GENE_DATA_DIR.name}/golub_full_genes.parquet (Shape: {c_full.shape})")

        if target in ("armstrong", "all"):
            arm_p = _locate_probe_file("armstrong_full.parquet")
            if arm_p.exists():
                arm_df = pd.read_parquet(arm_p)
                c_arm, _ = collapse_dataset(
                    train_df=arm_df,
                    mapping_df=armstrong_mapping,
                    strategy=strategy,
                    drop_multi_mapped=drop_multi_mapped,
                )
                c_arm.to_parquet(GENE_DATA_DIR / "armstrong_full_genes.parquet")
                print(f"    [+] Saved: {GENE_DATA_DIR.name}/armstrong_full_genes.parquet (Shape: {c_arm.shape})")

    # 4. Optional Cross-Platform Harmonization
    if do_harmonize:
        print("\n[*] Step 4: Harmonizing cohorts on shared HUGO gene symbols...")
        golub_full_p = _locate_probe_file("golub_full_72.parquet")
        arm_full_p = _locate_probe_file("armstrong_full.parquet")

        if not (golub_full_p.exists() and arm_full_p.exists()):
            print("[!] Cannot harmonize: ensure both golub_full_72.parquet and armstrong_full.parquet exist on disk.")
        else:
            golub_df = pd.read_parquet(golub_full_p)
            armstrong_df = pd.read_parquet(arm_full_p)
            harmonize_cohort_genes(
                golub_df=golub_df,
                armstrong_df=armstrong_df,
                golub_mapping=golub_mapping,
                armstrong_mapping=armstrong_mapping,
                strategy=strategy,
                drop_multi_mapped=drop_multi_mapped,
                out_dir=HARMONIZED_DATA_DIR,
            )

    print("\n[✓] Probe-to-gene mapping & harmonization pipeline finished successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Probe-to-Gene symbol mapping and cross-platform cohort harmonization."
    )
    parser.add_argument(
        "--dataset",
        choices=["golub", "armstrong", "all"],
        default="all",
        help="Target cohort for annotation extraction: 'golub', 'armstrong', or 'all' (default).",
    )
    parser.add_argument(
        "--strategy",
        choices=["max_variance", "mean", "median"],
        default="max_variance",
        help="Aggregation strategy for multi-probe genes: 'max_variance' (default), 'mean', or 'median'.",
    )
    parser.add_argument(
        "--drop-multi-mapped",
        action="store_true",
        help="Drop probes mapped to multiple genes (e.g. 'GENE1 /// GENE2') instead of taking the primary symbol.",
    )
    parser.add_argument(
        "--collapse",
        action="store_true",
        help="Collapse processed probe matrices into gene-level matrices.",
    )
    parser.add_argument(
        "--harmonize",
        action="store_true",
        help="Harmonize Golub and Armstrong into identical shared-gene feature matrices.",
    )
    args = parser.parse_args()

    run_mapping_pipeline(
        target=args.dataset,
        strategy=args.strategy,
        drop_multi_mapped=args.drop_multi_mapped,
        do_collapse=args.collapse,
        do_harmonize=args.harmonize,
    )


if __name__ == "__main__":
    main()
