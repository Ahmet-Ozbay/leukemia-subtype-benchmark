"""
Automated dataset acquisition pipeline for Leukemia Subtype Benchmark.
Downloads Golub (Kaggle), Armstrong et al. (GEO GSE28497), and MILE (GEO GSE13159).
Fulfills Rubric F9 for reproducible execution without committing raw data.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from src.config import RAW_DATA_DIR

def ensure_kaggle_token() -> bool:
    """Verifies KAGGLE_API_TOKEN exists, delegating to setup script if missing."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    print("\n[!] Kaggle API token not detected in environment or .env.")
    proceed = input("    Would you like to configure it now? [Y/n]: ").strip().lower()
    if proceed in ["n", "no"]:
        print("[!] Download aborted: Kaggle authentication is required.", file=sys.stderr)
        return False

    from scripts.setup_kaggle_auth import prompt_kaggle_token
    return prompt_kaggle_token()


def download_golub(
    dataset_slug: str = "crawford/gene-expression",
    label: str = "golub",
    dest_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> None:
    """Downloads and stages Golub et al. (1999) dataset into data/raw/<label>/."""
    expected_files = [
        "data_set_ALL_AML_train.csv",
        "data_set_ALL_AML_independent.csv",
        "actual.csv",
    ]

    target_dir = dest_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    if not force and all((target_dir / f).exists() for f in expected_files):
        print(f"[✓] {label} ({dataset_slug}) is already present in data/raw/{label}/. (Use --force to redownload)")
        return

    print("\n" + "-" * 60)
    print(f"  Fetching {label} ({dataset_slug}) via kagglehub...")
    print("-" * 60)

    if not ensure_kaggle_token():
        return

    try:
        import kagglehub

        print(f"[*] Fetching archive '{dataset_slug}'...")
        cache_dir = Path(kagglehub.dataset_download(dataset_slug))

        target_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for filename in expected_files:
            src = cache_dir / filename
            if src.exists():
                shutil.copy2(src, target_dir / filename)
                print(f"    [+] Staged: data/raw/{label}/{filename}")
            else:
                missing.append(filename)

        if missing:
            print(f"[!] Warning: Missing expected file(s): {missing}", file=sys.stderr)
        else:
            print(f"[✓] Successfully retrieved {label} ({dataset_slug}) into data/raw/{label}/")

    except Exception as err:
        print(f"[!] Failed to acquire Golub dataset: {err}", file=sys.stderr)
        print("    Run 'python scripts/setup_kaggle_auth.py' to reconfigure your key.")


def download_geo(geo_id: str, label: str, dest_dir: Path, force: bool = False) -> None:
    """Downloads and caches an NCBI GEO Series Matrix using GEOparse."""
    import GEOparse

    target_dir = dest_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(target_dir.glob(f"{geo_id}*"))
    if not force and len(existing_files) > 0:
        print(f"[✓] {label} ({geo_id}) is already present in data/raw/{label}/. (Use --force to redownload)")
        return

    print("\n" + "-" * 60)
    print(f"  Fetching {label} ({geo_id}) from NCBI GEO...")
    print("-" * 60)

    try:
        GEOparse.get_GEO(geo=geo_id, destdir=str(target_dir), silent=False)
        print(f"[✓] Successfully retrieved {label} ({geo_id}) into data/raw/{label}/")
    except Exception as err:
        print(f"[!] Failed to download {geo_id}: {err}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Automated dataset downloader for Leukemia Subtype Benchmark."
    )
    parser.add_argument(
        "--dataset",
        choices=["golub", "armstrong", "mile", "all"],
        default="all",
        help="Target dataset: 'golub', 'armstrong' (GSE28497), 'mile' (GSE13159), or 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if raw files already exist.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LEUKEMIA SUBTYPE BENCHMARK - DATA ACQUISITION PIPELINE")
    print("=" * 60)

    if args.dataset in ["golub", "all"]:
        download_golub("crawford/gene-expression", "golub", RAW_DATA_DIR, force=args.force)

    if args.dataset in ["armstrong", "all"]:
        download_geo("GSE28497", "armstrong_2002", RAW_DATA_DIR, force=args.force)

    if args.dataset in ["mile", "all"]:
        download_geo("GSE13159", "mile_study", RAW_DATA_DIR, force=args.force)

    print("\n[✓] Pipeline execution finished.")


if __name__ == "__main__":
    main()