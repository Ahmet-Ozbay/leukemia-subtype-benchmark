"""
Cross-platform gene harmonization and biomarker reporting for Golub and Armstrong cohorts.
Aligns feature spaces to common HUGO gene symbols and outputs verification tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import HARMONIZED_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from src.features.collapser import collapse_dataset


KEY_BIOMARKERS = [
    {
        "gene_symbol": "CST3",
        "name": "Cystatin C",
        "clinical_role": "Classic AML-selective marker (Golub et al. 1999 top predictor)",
        "subtype_association": "AML",
    },
    {
        "gene_symbol": "CD33",
        "name": "Myeloid cell surface antigen CD33",
        "clinical_role": "Myeloid differentiation antigen, AML therapeutic target (Gemtuzumab)",
        "subtype_association": "AML",
    },
    {
        "gene_symbol": "CD19",
        "name": "B-lymphocyte antigen CD19",
        "clinical_role": "Definitive B-lineage surface marker, ALL diagnostic hallmark",
        "subtype_association": "ALL",
    },
    {
        "gene_symbol": "MPO",
        "name": "Myeloperoxidase",
        "clinical_role": "Specific enzyme marker of myeloid lineage, AML diagnostic standard",
        "subtype_association": "AML",
    },
    {
        "gene_symbol": "FAH",
        "name": "Fumarylacetoacetate hydrolase",
        "clinical_role": "AML discriminator identified in primary Golub predictor gene set",
        "subtype_association": "AML",
    },
    {
        "gene_symbol": "MME",
        "name": "Membrane metalloendopeptidase (CD10 / CALLA)",
        "clinical_role": "Common Acute Lymphoblastic Leukemia Antigen (CALLA), key ALL marker",
        "subtype_association": "ALL",
    },
    {
        "gene_symbol": "LYZ",
        "name": "Lysozyme",
        "clinical_role": "Monocytic/granulocytic marker elevated in AML subtype FAB M4/M5",
        "subtype_association": "AML",
    },
]


def df_to_markdown(df: pd.DataFrame) -> str:
    """Renders a pandas DataFrame as a GitHub-flavored Markdown table without tabulate dependency."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(val).replace("|", "\\|") for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def get_shared_gene_symbols(
    golub_mapping: pd.DataFrame,
    armstrong_mapping: pd.DataFrame,
) -> List[str]:
    """Calculates the intersection of valid HUGO gene symbols across both cohorts."""
    g_syms = set(golub_mapping[golub_mapping["is_mapped"]]["gene_symbol"].dropna()) - {""}
    a_syms = set(armstrong_mapping[armstrong_mapping["is_mapped"]]["gene_symbol"].dropna()) - {""}
    shared = sorted(list(g_syms.intersection(a_syms)))
    return shared


def generate_harmonization_report(
    golub_mapping: pd.DataFrame,
    armstrong_mapping: pd.DataFrame,
    out_dir: Path = TABLES_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Generates and persists side-by-side harmonization specification tables (Rubric A3 & F5)
    and canonical leukemia biomarker mapping tables.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_genes = get_shared_gene_symbols(golub_mapping, armstrong_mapping)
    g_mapped = golub_mapping[golub_mapping["is_mapped"]]
    a_mapped = armstrong_mapping[armstrong_mapping["is_mapped"]]

    g_syms_count = g_mapped["gene_symbol"].nunique()
    a_syms_count = a_mapped["gene_symbol"].nunique()
    g_cov = (len(shared_genes) / g_syms_count) * 100 if g_syms_count else 0.0
    a_cov = (len(shared_genes) / a_syms_count) * 100 if a_syms_count else 0.0

    # 1. Summary comparison table
    summary_data = [
        {
            "Metric": "Cohort / Publication",
            "Golub et al. (1999)": "Golub et al. (1999)",
            "Armstrong et al. (2002)": "Armstrong et al. (GSE28497)",
        },
        {
            "Metric": "GEO Platform ID",
            "Golub et al. (1999)": "GPL80",
            "Armstrong et al. (2002)": "GPL96",
        },
        {
            "Metric": "Array Platform Name",
            "Golub et al. (1999)": "Affymetrix HuGeneFL (Hu6800)",
            "Armstrong et al. (2002)": "Affymetrix Human Genome U133A",
        },
        {
            "Metric": "Total Probe Sets (p)",
            "Golub et al. (1999)": str(len(golub_mapping)),
            "Armstrong et al. (2002)": str(len(armstrong_mapping)),
        },
        {
            "Metric": "Mapped Probes to HUGO Symbols",
            "Golub et al. (1999)": f"{len(g_mapped)} ({len(g_mapped)/len(golub_mapping):.1%})",
            "Armstrong et al. (2002)": f"{len(a_mapped)} ({len(a_mapped)/len(armstrong_mapping):.1%})",
        },
        {
            "Metric": "Control & Unmapped Probes",
            "Golub et al. (1999)": str(len(golub_mapping) - len(g_mapped)),
            "Armstrong et al. (2002)": str(len(armstrong_mapping) - len(a_mapped)),
        },
        {
            "Metric": "Unique HUGO Gene Symbols",
            "Golub et al. (1999)": str(g_syms_count),
            "Armstrong et al. (2002)": str(a_syms_count),
        },
        {
            "Metric": "Cross-Platform Overlapping Genes",
            "Golub et al. (1999)": f"{len(shared_genes)} ({g_cov:.1f}% of Golub genes)",
            "Armstrong et al. (2002)": f"{len(shared_genes)} ({a_cov:.1f}% of Armstrong genes)",
        },
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_csv = out_dir / "gene_harmonization_summary.csv"
    summary_md = out_dir / "gene_harmonization_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Cross-Platform Microarray Cohort Harmonization Summary\n\n")
        f.write(df_to_markdown(summary_df))
        f.write("\n")

    # 2. Canonical biomarker tracking table
    biomarker_rows = []
    for bm in KEY_BIOMARKERS:
        sym = bm["gene_symbol"]
        g_probes = golub_mapping[golub_mapping["gene_symbol"] == sym]["probe_id"].tolist()
        a_probes = armstrong_mapping[armstrong_mapping["gene_symbol"] == sym]["probe_id"].tolist()

        biomarker_rows.append(
            {
                "Gene Symbol": sym,
                "Full Name": bm["name"],
                "Diagnostic Role": bm["clinical_role"],
                "Subtype": bm["subtype_association"],
                "Golub (Hu6800 / GPL80) Probes": ", ".join(g_probes) if g_probes else "N/A",
                "Armstrong (HG-U133A / GPL96) Probes": ", ".join(a_probes) if a_probes else "N/A",
                "Cross-Platform Match": "Yes" if (g_probes and a_probes) else "No",
            }
        )

    biomarker_df = pd.DataFrame(biomarker_rows)
    biomarker_csv = out_dir / "biomarker_probe_mapping.csv"
    biomarker_md = out_dir / "biomarker_probe_mapping.md"
    biomarker_df.to_csv(biomarker_csv, index=False)
    with open(biomarker_md, "w", encoding="utf-8") as f:
        f.write("# Canonical Leukemia Subtype Biomarker Cross-Platform Mapping\n\n")
        f.write(df_to_markdown(biomarker_df))
        f.write("\n")

    print(f"    [+] Saved harmonization summary: {summary_csv.name} and {summary_md.name}")
    print(f"    [+] Saved biomarker validation:  {biomarker_csv.name} and {biomarker_md.name}")

    return {"summary": summary_df, "biomarkers": biomarker_df}


def harmonize_cohort_genes(
    golub_df: pd.DataFrame,
    armstrong_df: pd.DataFrame,
    golub_mapping: pd.DataFrame,
    armstrong_mapping: pd.DataFrame,
    strategy: str = "max_variance",
    drop_multi_mapped: bool = False,
    out_dir: Path = HARMONIZED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapses both cohorts to gene symbols using zero-leakage aggregation and harmonizes
    feature space strictly to the common gene intersection.
    """
    shared_genes = get_shared_gene_symbols(golub_mapping, armstrong_mapping)
    print(f"[*] Harmonizing cohorts on {len(shared_genes)} shared HUGO gene symbols (Strategy: {strategy})...")

    # Collapse Golub
    golub_collapsed, _ = collapse_dataset(
        train_df=golub_df,
        mapping_df=golub_mapping,
        strategy=strategy,
        drop_multi_mapped=drop_multi_mapped,
    )

    # Collapse Armstrong
    armstrong_collapsed, _ = collapse_dataset(
        train_df=armstrong_df,
        mapping_df=armstrong_mapping,
        strategy=strategy,
        drop_multi_mapped=drop_multi_mapped,
    )

    meta_cols = ["label", "subtype"]

    # Subset strictly to shared genes with identical column ordering
    golub_harmonized = pd.concat([golub_collapsed[shared_genes], golub_collapsed[meta_cols]], axis=1)
    armstrong_harmonized = pd.concat([armstrong_collapsed[shared_genes], armstrong_collapsed[meta_cols]], axis=1)

    # Invariant assertions
    assert list(golub_harmonized.columns) == list(armstrong_harmonized.columns), "Harmonized feature columns do not match!"
    assert not golub_harmonized.isna().any().any(), "Harmonized Golub matrix contains NaNs!"
    assert not armstrong_harmonized.isna().any().any(), "Harmonized Armstrong matrix contains NaNs!"

    # Export
    out_dir.mkdir(parents=True, exist_ok=True)
    golub_out = out_dir / "golub_harmonized_genes.parquet"
    armstrong_out = out_dir / "armstrong_harmonized_genes.parquet"

    golub_harmonized.to_parquet(golub_out)
    armstrong_harmonized.to_parquet(armstrong_out)

    print(f"    [+] Saved: {golub_out.name} (Shape: {golub_harmonized.shape})")
    print(f"    [+] Saved: {armstrong_out.name} (Shape: {armstrong_harmonized.shape})")

    return golub_harmonized, armstrong_harmonized
