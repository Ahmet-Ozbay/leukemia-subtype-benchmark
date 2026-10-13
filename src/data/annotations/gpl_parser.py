"""
NCBI GEO Platform (GPL) annotation parser for Leukemia Subtype Benchmark.
Extracts, cleans, and standardizes probe-to-gene mappings for:
- Affymetrix HuGeneFL / Hu6800 (GEO GPL80) -> Golub cohort
- Affymetrix HG-U133A (GEO GPL96) -> Armstrong GSE28497 cohort
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    ANNOTATIONS_DIR,
    EXTERNAL_DATA_DIR,
    PROBE_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def clean_symbol(val: any) -> str:
    """Sanitizes raw symbol string, removing whitespace and nan representations."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "", "---"):
        return ""
    return s


def parse_gpl80_annotations(
    gpl_path: Optional[Path] = None,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Parses GPL80 (HuGeneFL / Hu6800) annotation table.
    Downloads from NCBI GEO if local file not found in data/external/.
    """
    import GEOparse

    if gpl_path is None:
        gpl_path = EXTERNAL_DATA_DIR / "GPL80.txt"

    if not gpl_path.exists():
        print(f"[*] GPL80 annotation not found at {gpl_path}. Fetching from NCBI GEO...")
        EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        GEOparse.get_GEO("GPL80", destdir=str(EXTERNAL_DATA_DIR), silent=True)

    table = GEOparse.parse_GPL(str(gpl_path)).table

    records = []
    for _, row in table.iterrows():
        probe_id = str(row["ID"]).strip()
        raw_sym = clean_symbol(row.get("Gene Symbol", ""))
        is_multi = "///" in raw_sym
        is_ctrl = probe_id.startswith("AFFX-") or probe_id == "hum_alu_at"

        if drop_multi_mapped and is_multi:
            gene_symbol = ""
        else:
            gene_symbol = raw_sym.split("///")[0].strip() if raw_sym else ""

        raw_entrez = clean_symbol(row.get("ENTREZ_GENE_ID", ""))
        entrez_id = raw_entrez.split("///")[0].strip() if raw_entrez else ""
        gene_title = clean_symbol(row.get("Gene Title", ""))

        is_mapped = bool(gene_symbol and not is_ctrl)

        records.append(
            {
                "probe_id": probe_id,
                "gene_symbol": gene_symbol,
                "entrez_id": entrez_id,
                "gene_title": gene_title,
                "raw_symbol": raw_sym,
                "is_multi_mapped": is_multi,
                "is_control": is_ctrl,
                "is_mapped": is_mapped,
            }
        )

    df = pd.DataFrame(records)
    return df


def parse_gpl96_annotations(
    soft_path: Optional[Path] = None,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Parses GPL96 (HG-U133A) annotation table from GSE28497 SOFT archive or external cache.
    """
    import GEOparse

    cached_gpl = EXTERNAL_DATA_DIR / "GPL96.txt"
    if cached_gpl.exists():
        table = GEOparse.parse_GPL(str(cached_gpl)).table
    else:
        if soft_path is None:
            candidates = list((RAW_DATA_DIR / "armstrong_2002").glob("GSE28497*.soft*"))
            if candidates:
                soft_path = candidates[0]

        if soft_path is not None and soft_path.exists():
            print(f"[*] Extracting GPL96 from {soft_path.name}...")
            gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
            if "GPL96" in gse.gpls:
                table = gse.gpls["GPL96"].table
            else:
                raise KeyError(f"GPL96 not found in {soft_path}. Available GPLs: {list(gse.gpls.keys())}")
        else:
            print("[*] Fetching GPL96 from NCBI GEO...")
            EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            gpl = GEOparse.get_GEO("GPL96", destdir=str(EXTERNAL_DATA_DIR), silent=True)
            table = gpl.table

    records = []
    for _, row in table.iterrows():
        probe_id = str(row["ID"]).strip()
        raw_sym = clean_symbol(row.get("Gene Symbol", ""))
        is_multi = "///" in raw_sym
        is_ctrl = probe_id.startswith("AFFX-")

        if drop_multi_mapped and is_multi:
            gene_symbol = ""
        else:
            gene_symbol = raw_sym.split("///")[0].strip() if raw_sym else ""

        raw_entrez = clean_symbol(row.get("ENTREZ_GENE_ID", ""))
        entrez_id = raw_entrez.split("///")[0].strip() if raw_entrez else ""
        gene_title = clean_symbol(row.get("Gene Title", ""))

        is_mapped = bool(gene_symbol and not is_ctrl)

        records.append(
            {
                "probe_id": probe_id,
                "gene_symbol": gene_symbol,
                "entrez_id": entrez_id,
                "gene_title": gene_title,
                "raw_symbol": raw_sym,
                "is_multi_mapped": is_multi,
                "is_control": is_ctrl,
                "is_mapped": is_mapped,
            }
        )

    df = pd.DataFrame(records)
    return df


def extract_golub_annotations(
    out_dir: Path = ANNOTATIONS_DIR,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Extracts Golub (GPL80) probe mapping and persists to Parquet and CSV.
    Validates that all feature probes in processed Golub parquet are covered.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_gpl80_annotations(drop_multi_mapped=drop_multi_mapped)

    # Validate against processed dataset if available
    golub_path = PROBE_DATA_DIR / "golub_full_72.parquet"
    if not golub_path.exists():
        golub_path = PROCESSED_DATA_DIR / "golub_full_72.parquet"
    if golub_path.exists():
        golub_df = pd.read_parquet(golub_path)
        feature_probes = [c for c in golub_df.columns if c not in ["label", "subtype"]]
        annotated_probes = set(df["probe_id"])
        missing = set(feature_probes) - annotated_probes
        assert len(missing) == 0, f"Golub features missing from GPL80 annotation: {len(missing)} probes"

    parquet_path = out_dir / "golub_probe_to_gene.parquet"
    csv_path = out_dir / "golub_probe_to_gene.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"    [+] Saved Golub annotations: {parquet_path.name} ({len(df)} probes, {df['is_mapped'].sum()} mapped)")
    return df


def extract_armstrong_annotations(
    out_dir: Path = ANNOTATIONS_DIR,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Extracts Armstrong (GPL96) probe mapping and persists to Parquet and CSV.
    Validates that all feature probes in processed Armstrong parquet are covered.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_gpl96_annotations(drop_multi_mapped=drop_multi_mapped)

    # Validate against processed dataset if available
    armstrong_path = PROBE_DATA_DIR / "armstrong_full.parquet"
    if not armstrong_path.exists():
        armstrong_path = PROCESSED_DATA_DIR / "armstrong_full.parquet"
    if armstrong_path.exists():
        arm_df = pd.read_parquet(armstrong_path)
        feature_probes = [c for c in arm_df.columns if c not in ["label", "subtype"]]
        annotated_probes = set(df["probe_id"])
        missing = set(feature_probes) - annotated_probes
        assert len(missing) == 0, f"Armstrong features missing from GPL96 annotation: {len(missing)} probes"

    parquet_path = out_dir / "armstrong_probe_to_gene.parquet"
    csv_path = out_dir / "armstrong_probe_to_gene.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"    [+] Saved Armstrong annotations: {parquet_path.name} ({len(df)} probes, {df['is_mapped'].sum()} mapped)")
    return df


def load_annotation_mapping(
    cohort: str,
    annotations_dir: Path = ANNOTATIONS_DIR,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Loads cached probe-to-gene mapping table for a cohort, or parses it if not yet cached.
    """
    parquet_path = annotations_dir / f"{cohort}_probe_to_gene.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if drop_multi_mapped:
            df = df.copy()
            df.loc[df["is_multi_mapped"], "gene_symbol"] = ""
            df.loc[df["is_multi_mapped"], "is_mapped"] = False
        return df

    if cohort == "golub":
        return extract_golub_annotations(out_dir=annotations_dir, drop_multi_mapped=drop_multi_mapped)
    elif cohort == "armstrong":
        return extract_armstrong_annotations(out_dir=annotations_dir, drop_multi_mapped=drop_multi_mapped)
    else:
        raise ValueError(f"Unknown cohort '{cohort}'. Supported: 'golub', 'armstrong'.")
