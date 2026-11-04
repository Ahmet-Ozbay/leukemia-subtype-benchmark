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


def extract_primary_symbol_and_entrez(raw_sym: str, raw_entrez: str) -> tuple[str, str]:
    """
    Extracts primary gene symbol and corresponding Entrez ID.
    If multiple symbols are present (separated by '///'), prioritizes official
    HUGO symbols over uncharacterized LOC... identifiers.
    """
    if not raw_sym:
        return "", ""
    parts = [p.strip() for p in raw_sym.split("///") if p.strip()]
    entrez_parts = [p.strip() for p in raw_entrez.split("///") if p.strip()] if raw_entrez else []

    canonical_indices = [i for i, p in enumerate(parts) if not p.upper().startswith("LOC")]
    if canonical_indices:
        idx = canonical_indices[0]
        sym = parts[idx]
        eid = entrez_parts[idx] if idx < len(entrez_parts) else (entrez_parts[0] if entrez_parts else "")
    elif parts:
        sym = parts[0]
        eid = entrez_parts[0] if entrez_parts else ""
    else:
        sym = ""
        eid = ""
    return sym, eid


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

        raw_entrez = clean_symbol(row.get("ENTREZ_GENE_ID", ""))
        gene_title = clean_symbol(row.get("Gene Title", ""))

        if drop_multi_mapped and is_multi:
            gene_symbol = ""
            entrez_id = ""
        else:
            gene_symbol, entrez_id = extract_primary_symbol_and_entrez(raw_sym, raw_entrez)

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
    Parses GPL96 (HG-U133A) annotation table from external cache or NCBI GEO.
    """
    import GEOparse

    cached_gpl = EXTERNAL_DATA_DIR / "GPL96.txt"
    if cached_gpl.exists():
        table = GEOparse.parse_GPL(str(cached_gpl)).table
    else:
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

        raw_entrez = clean_symbol(row.get("ENTREZ_GENE_ID", ""))
        gene_title = clean_symbol(row.get("Gene Title", ""))

        if drop_multi_mapped and is_multi:
            gene_symbol = ""
            entrez_id = ""
        else:
            gene_symbol, entrez_id = extract_primary_symbol_and_entrez(raw_sym, raw_entrez)

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


def extract_gse51082_annotations(
    out_dir: Path = ANNOTATIONS_DIR,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Extracts GSE51082 (GPL96 / HG-U133A) probe mapping and persists to Parquet and CSV.
    Validates that all feature probes in processed GSE51082 parquet are covered.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_gpl96_annotations(drop_multi_mapped=drop_multi_mapped)

    # Validate against processed dataset if available
    gse_path = PROBE_DATA_DIR / "gse51082_all_aml.parquet"
    if not gse_path.exists():
        gse_path = PROBE_DATA_DIR / "gse51082_full_eda.parquet"
    if gse_path.exists():
        gse_df = pd.read_parquet(gse_path)
        meta_cols = {"label", "subtype", "tissue", "disease", "patient_id", "sample_id", "title"}
        feature_probes = [c for c in gse_df.columns if c not in meta_cols]
        annotated_probes = set(df["probe_id"])
        missing = set(feature_probes) - annotated_probes
        assert len(missing) == 0, f"GSE51082 features missing from GPL96 annotation: {len(missing)} probes"

    parquet_path = out_dir / "gse51082_probe_to_gene.parquet"
    csv_path = out_dir / "gse51082_probe_to_gene.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"    [+] Saved GSE51082 annotations: {parquet_path.name} ({len(df)} probes, {df['is_mapped'].sum()} mapped)")
    return df


def extract_coustan_smith_annotations(
    out_dir: Path = ANNOTATIONS_DIR,
    drop_multi_mapped: bool = False,
) -> pd.DataFrame:
    """
    Extracts Coustan-Smith (GPL96) probe mapping and persists to Parquet and CSV.
    Deprecated: Kept for backward compatibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_gpl96_annotations(drop_multi_mapped=drop_multi_mapped)

    parquet_path = out_dir / "coustan_smith_probe_to_gene.parquet"
    csv_path = out_dir / "coustan_smith_probe_to_gene.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"    [+] Saved Coustan-Smith annotations: {parquet_path.name} ({len(df)} probes, {df['is_mapped'].sum()} mapped)")
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
    elif cohort == "gse51082":
        return extract_gse51082_annotations(out_dir=annotations_dir, drop_multi_mapped=drop_multi_mapped)
    elif cohort == "coustan_smith":
        return extract_coustan_smith_annotations(out_dir=annotations_dir, drop_multi_mapped=drop_multi_mapped)
    else:
        raise ValueError(f"Unknown cohort '{cohort}'. Supported: 'golub', 'gse51082', 'coustan_smith'.")
