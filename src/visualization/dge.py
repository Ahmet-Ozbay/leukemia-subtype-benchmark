"""
Figure 3: Differential Gene Expression (DGE) & Hierarchical Clustermap.

Performs parametric (Welch's t-test) and non-parametric (Mann-Whitney U) differential
expression analysis with Benjamini-Hochberg FDR correction. Generates a two-way
hierarchical clustered heatmap (Euclidean distance, Ward linkage) highlighting
canonical acute leukemia biomarkers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from src.config import FIGURES_DIR
from src.data.io import load_dataset
from src.visualization.style import apply_style, save_figure
from src.visualization.utils import get_feature_and_label_matrices

CANONICAL_BIOMARKERS: Set[str] = {"CST3", "CD33", "MPO", "LYZ", "FAH", "CD19", "MME"}


def compute_differential_expression(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Computes univariate differential gene expression statistics between
    AML (Class 1) and ALL (Class 0).

    Calculates:
    - Welch's two-sample unequal-variance t-statistic and p-value.
    - Two-sided Mann-Whitney U statistic and p-value.
    - Benjamini-Hochberg False Discovery Rate (FDR) adjustments for both tests.

    Parameters:
        X: Feature DataFrame of log2-transformed expression values (samples x genes).
        y: Binary target labels (0 = ALL, 1 = AML).

    Returns:
        pd.DataFrame sorted by p_welch ascending, containing:
        ['gene', 't_stat', 'p_welch', 'fdr_welch', 'u_stat', 'p_mwu', 'fdr_mwu']
    """
    all_mask = (y == 0).values
    aml_mask = (y == 1).values

    stats_list: List[Dict[str, Union[str, float]]] = []
    for gene in X.columns:
        all_vals = X.loc[all_mask, gene].to_numpy(dtype=float)
        aml_vals = X.loc[aml_mask, gene].to_numpy(dtype=float)

        # Handle zero-variance edge cases gracefully
        if np.all(all_vals == aml_vals[0]) and np.all(aml_vals == aml_vals[0]):
            t_stat, p_welch = 0.0, 1.0
            u_stat, p_mwu = 0.0, 1.0
        else:
            t_stat, p_welch = stats.ttest_ind(aml_vals, all_vals, equal_var=False, nan_policy="omit")
            try:
                u_stat, p_mwu = stats.mannwhitneyu(aml_vals, all_vals, alternative="two-sided")
            except ValueError:
                u_stat, p_mwu = 0.0, 1.0

        stats_list.append({
            "gene": str(gene),
            "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "p_welch": float(p_welch) if not np.isnan(p_welch) else 1.0,
            "u_stat": float(u_stat),
            "p_mwu": float(p_mwu) if not np.isnan(p_mwu) else 1.0,
        })

    res_df = pd.DataFrame(stats_list)
    n_genes = len(res_df)

    if n_genes == 0:
        return pd.DataFrame(columns=["gene", "t_stat", "p_welch", "fdr_welch", "u_stat", "p_mwu", "fdr_mwu"])

    # Benjamini-Hochberg step-up FDR calculation
    for p_col, fdr_col in [("p_welch", "fdr_welch"), ("p_mwu", "fdr_mwu")]:
        res_df = res_df.sort_values(p_col).reset_index(drop=True)
        res_df[fdr_col] = (res_df[p_col] * n_genes) / (res_df.index + 1)
        res_df[fdr_col] = np.minimum.accumulate(res_df[fdr_col].iloc[::-1])[::-1]
        res_df[fdr_col] = res_df[fdr_col].clip(upper=1.0)

    res_df = res_df.sort_values("p_welch").reset_index(drop=True)
    return res_df


def plot_top_genes_heatmap(
    df: pd.DataFrame,
    top_n: int = 40,
    style_cfg: Optional[Dict] = None,
) -> sns.matrix.ClusterGrid:
    """
    Renders two-way hierarchical clustering heatmap (clustermap) of top differentially
    expressed genes between ALL and AML.

    Parameters:
        df: Expression DataFrame with metadata columns.
        top_n: Number of top discriminative genes to display.
        style_cfg: Optional style configuration dictionary from apply_style.

    Returns:
        sns.matrix.ClusterGrid: The rendered clustermap grid.
    """
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    X_raw, y = get_feature_and_label_matrices(df)
    X_log = np.log2(np.maximum(X_raw, 0) + 1.0)

    res_df = compute_differential_expression(X_log, y)

    # Strictly data-driven selection of top N genes passing FDR < 0.001
    fdr_sig = res_df[res_df["fdr_welch"] < 0.001]
    if len(fdr_sig) >= min(5, top_n):
        top_genes = fdr_sig.head(top_n)["gene"].tolist()
    else:
        # Fallback for synthetic/smaller test matrices
        top_genes = res_df.head(top_n)["gene"].tolist()

    # Extract expression sub-matrix and compute z-score per gene across samples
    X_sub = X_log[top_genes].T  # genes x samples
    X_sub_z = X_sub.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

    palette = style_cfg["palette"]
    col_colors = pd.Series(
        [palette.get(val, "#333333") for val in y],
        index=X_sub_z.columns,
        name="Subtype",
    )

    # Format row labels: star canonical biomarkers
    new_index = [
        f"{g} *" if g in CANONICAL_BIOMARKERS else str(g)
        for g in X_sub_z.index
    ]
    X_sub_z.index = new_index

    cmap = style_cfg.get("heatmap_cmap", "mako")

    g = sns.clustermap(
        X_sub_z,
        col_colors=col_colors,
        cmap=cmap,
        metric="euclidean",
        method="ward",
        figsize=(7.2, 8.5),
        linewidths=0.0,
        cbar_pos=(0.02, 0.85, 0.03, 0.12),
        cbar_kws={"label": "Z-Score Expression"},
        yticklabels=True,
        xticklabels=False,
        tree_kws={"linewidths": 0.8},
    )

    g.fig.suptitle(
        f"Hierarchical Clustering of Top {len(top_genes)} Discriminative Genes (* Canonical Biomarkers)",
        fontweight="bold",
        fontsize=9.5,
        y=0.99,
    )

    legend_elements = [
        Patch(facecolor=palette.get(0, "#333333"), edgecolor="black", label="ALL (Class 0)"),
        Patch(facecolor=palette.get(1, "#999999"), edgecolor="black", label="AML (Class 1)"),
    ]
    if hasattr(g, "ax_col_dendrogram") and g.ax_col_dendrogram is not None:
        g.ax_col_dendrogram.legend(
            handles=legend_elements,
            loc="center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=2,
            fontsize=8,
            frameon=True,
        )

    g.ax_heatmap.set_ylabel("HUGO Gene Symbol", fontsize=8.5)
    g.ax_heatmap.set_xlabel(f"Patient Samples (n = {len(df)})", fontsize=8.5)
    g.ax_heatmap.tick_params(axis="y", labelsize=7.5, pad=2)

    return g


def generate_figure_3(
    style_cfg: Optional[Dict] = None,
    top_n: int = 40,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Union[Path, str] = FIGURES_DIR,
) -> Path:
    """
    Runner for Figure 3: loads Golub gene-level data, renders Figure 3,
    and persists 300 DPI PNG and vector PDF.
    """
    print(f"[*] Generating Figure 3: Top {top_n} Genes Heatmap & Clustermap...")
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    df = load_dataset("golub", level="genes", split="full")
    g = plot_top_genes_heatmap(df, top_n=top_n, style_cfg=style_cfg)
    png_path, _ = save_figure(
        g,
        "fig3_top_genes_heatmap",
        out_dir=out_dir,
        dpi=dpi,
        save_pdf=save_pdf,
        close=True,
    )
    print(f"    [+] Saved Figure 3 to: {png_path}")
    return png_path

