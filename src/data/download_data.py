"""
Automated dataset acquisition pipeline for Leukemia Subtype Benchmark.
Downloads Golub (Kaggle), Coustan-Smith et al. (GEO GSE28497), and MILE (GEO GSE13159).
Fulfills Rubric F9 for reproducible execution without committing raw data.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Enable direct script execution: python src/data/download_data.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR


def ensure_kaggle_token() -> bool:
    """Verifies KAGGLE_API_TOKEN exists, delegating to setup script if missing."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    print("\n[!] Kaggle API token not detected in environment or .env.", file=sys.stderr)
    if not sys.stdin.isatty():
        print(
            "[!] Non-interactive environment detected. Please set KAGGLE_API_TOKEN in .env or environment.",
            file=sys.stderr,
        )
        return False

    try:
        proceed = input("    Would you like to configure it now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Input aborted.", file=sys.stderr)
        return False

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
) -> bool:
    """Downloads and stages Golub et al. (1999) dataset into dest_dir/<label>/."""
    expected_files = [
        "data_set_ALL_AML_train.csv",
        "data_set_ALL_AML_independent.csv",
        "actual.csv",
    ]

    target_dir = dest_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check for complete, non-empty existing files
    if not force and all(
        (target_dir / f).exists() and (target_dir / f).stat().st_size > 0
        for f in expected_files
    ):
        print(f"[OK] {label} ({dataset_slug}) is already present in data/raw/{label}/. (Use --force to redownload)")
        return True

    print("\n" + "-" * 60)
    print(f"  Fetching {label} ({dataset_slug}) via kagglehub...")
    print("-" * 60)

    if not ensure_kaggle_token():
        return False

    try:
        import kagglehub

        print(f"[*] Fetching archive '{dataset_slug}'...")
        cache_dir = Path(kagglehub.dataset_download(dataset_slug))

        target_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for filename in expected_files:
            src = cache_dir / filename
            if src.exists() and src.stat().st_size > 0:
                shutil.copy2(src, target_dir / filename)
                print(f"    [+] Staged: data/raw/{label}/{filename}")
            else:
                missing.append(filename)

        if missing:
            print(f"[!] Warning: Missing or empty expected file(s): {missing}", file=sys.stderr)
            return False

        print(f"[OK] Successfully retrieved {label} ({dataset_slug}) into data/raw/{label}/")
        return True

    except Exception as err:
        print(f"[!] Failed to acquire Golub dataset: {err}", file=sys.stderr)
        print("    Run 'python scripts/setup_kaggle_auth.py' to reconfigure your key.")
        return False


def download_gse51082(dest_dir: Path = RAW_DATA_DIR, force: bool = False) -> bool:
    """Downloads and caches GSE51082-GPL96 (HG-U133A) Series Matrix from NCBI GEO."""
    target_dir = dest_dir / "gse51082"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "GSE51082-GPL96_series_matrix.txt.gz"

    if not force and target_file.exists() and target_file.stat().st_size > 1024:
        print(f"[OK] gse51082 is already present in {target_file}. (Use --force to redownload)")
        return True

    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51082/matrix/GSE51082-GPL96_series_matrix.txt.gz"
    print("\n" + "-" * 60)
    print("  Fetching GSE51082 (GPL96 Series Matrix) from NCBI GEO FTP...")
    print("-" * 60)

    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target_file, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)

        print(f"[OK] Successfully retrieved GSE51082 into {target_file} ({target_file.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as err:
        print(f"[!] Failed to download GSE51082: {err}", file=sys.stderr)
        return False


def download_geo(geo_id: str, label: str, dest_dir: Path, force: bool = False) -> bool:
    """Downloads and caches an NCBI GEO Series Matrix / SOFT archive using GEOparse."""
    import GEOparse

    target_dir = dest_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)

    if force:
        # GEOparse reuses local file if present on disk. Purge cache on force redownload.
        for existing in target_dir.glob(f"{geo_id}*"):
            if existing.is_file():
                existing.unlink()

    # Valid completed archives are non-empty (>1KB) and not temporary files
    valid_files = [
        f for f in target_dir.glob(f"{geo_id}*")
        if f.is_file() and not f.name.endswith(".tmp") and f.stat().st_size > 1024
    ]

    if not force and len(valid_files) > 0:
        print(f"[OK] {label} ({geo_id}) is already present in data/raw/{label}/. (Use --force to redownload)")
        return True

    print("\n" + "-" * 60)
    print(f"  Fetching {label} ({geo_id}) from NCBI GEO...")
    print("-" * 60)

    try:
        GEOparse.get_GEO(geo=geo_id, destdir=str(target_dir), silent=False)
        print(f"[OK] Successfully retrieved {label} ({geo_id}) into data/raw/{label}/")
        return True
    except Exception as err:
        print(f"[!] Failed to download {geo_id}: {err}", file=sys.stderr)
        return False


DOWNLOADERS = {
    "golub": lambda force: download_golub("crawford/gene-expression", "golub", RAW_DATA_DIR, force=force),
    "gse51082": lambda force: download_gse51082(RAW_DATA_DIR, force=force),
    "mile": lambda force: download_geo("GSE13159", "mile_study", RAW_DATA_DIR, force=force),
    "coustan_smith": lambda force: download_geo("GSE28497", "coustan_smith", RAW_DATA_DIR, force=force),
}


def main():
    parser = argparse.ArgumentParser(
        description="Automated dataset downloader for Leukemia Subtype Benchmark."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        default="all",
        help="Target dataset: 'golub', 'gse51082' (Agarwal et al. 2014), 'mile' (GSE13159), 'coustan_smith' (deprecated), or 'all' (default).",
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

    targets = list(DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]
    failed = []

    for name in targets:
        success = DOWNLOADERS[name](force=args.force)
        if not success:
            failed.append(name)

    if failed:
        print(f"\n[!] Pipeline execution finished with errors. Failed cohorts: {failed}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()