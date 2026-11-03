"""
Publication-Grade Exploratory Data Analysis (EDA) & Visualization Engine
for Leukemia Subtype Benchmark (ALL vs. AML).

Generates IEEE-compliant, 300+ DPI figures referenced in MÖAT - Ara Rapor:
- Figure 1: fig1_expression_distributions.png (Raw vs. log2(x+1) density distributions)
- Figure 2: fig2_dimensionality_reduction.png (Scree plot, 2D PCA, t-SNE, UMAP)
- Figure 3: fig3_top_genes_heatmap.png (Top 40 differentially expressed genes clustermap)
- Figure 4: fig4_batch_effect_diagnostics.png (Golub train/test shift & cross-platform batch effect)
- Figure 5: fig5_tissue_confounding_and_landscape.png (Tissue diagnostic & 6-class global landscape)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.data import load_dataset
from src.visualization.style import apply_style


def get_feature_and_label_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates numeric gene/probe expression columns from metadata ['label', 'subtype', etc.]."""
    meta_cols = [
        c
        for c in ["label", "subtype", "sample_id", "tissue", "disease", "patient_id", "title"]
        if c in df.columns
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].copy()
    y = df["label"].copy()
    return X, y


# ==============================================================================
# Figure 1: Expression Distributions (Raw vs. Log2(x + 1))
# ==============================================================================
def plot_figure_1_distributions(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Renders Figure 1: Density distributions comparing raw microarray fluorescence
    intensity against log2(x + 1) transformed values for Golub and Coustan-Smith cohorts.
    """
    print("[*] Generating Figure 1: Expression Distributions...")

    golub_df = load_dataset("golub", level="probes", split="full")
    gse_df = load_dataset("gse51082", level="probes", split="all_aml")

    X_golub_raw, _ = get_feature_and_label_matrices(golub_df)
    X_gse_raw, _ = get_feature_and_label_matrices(gse_df)

    # Compute exact full-cohort statistics for reporting
    full_g_raw = X_golub_raw.values.ravel()
    full_gse_rma = X_gse_raw.values.ravel()
    median_g = np.median(full_g_raw)
    skew_g = stats.skew(full_g_raw)
    median_gse = np.median(full_gse_rma)
    skew_gse = stats.skew(full_gse_rma)

    full_g_log = np.log2(full_g_raw + 1.0)
    std_g_log = np.std(full_g_log)
    skew_g_log = stats.skew(full_g_log)
    std_gse = np.std(full_gse_rma)

    # Sample representative values (subset of 50,000 measurements for smooth KDE calculation)
    np.random.seed(42)
    sample_g_raw = np.random.choice(full_g_raw, size=50000, replace=False)
    sample_g_log = np.log2(sample_g_raw + 1.0)

    sample_gse_rma = np.random.choice(full_gse_rma, size=50000, replace=False)
    sample_gse_std = (sample_gse_rma - np.mean(full_gse_rma)) / std_gse

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey="row")

    palette = style_cfg["palette"]
    gse_color = palette.get("GSE51082", palette.get("Coustan-Smith", "#888888"))

    # Golub Raw
    sns.histplot(
        sample_g_raw,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color=palette["Golub"],
        edgecolor=palette["Golub"],
        stat="density",
        alpha=0.3,
    )
    axes[0, 0].set_title("(a) Golub (Hu6800) - Raw Intensity", fontweight="bold")
    axes[0, 0].set_xlabel("Raw Fluorescence Intensity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_xlim(0, 16500)
    axes[0, 0].annotate(
        f"Skewness: {skew_g:.2f}\nMedian: {median_g:.1f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # GSE51082 RMA (Pre-normalized log2)
    sns.histplot(
        sample_gse_rma,
        bins=50,
        kde=True,
        ax=axes[0, 1],
        color=gse_color,
        edgecolor=gse_color,
        stat="density",
        alpha=0.3,
    )
    axes[0, 1].set_title(r"(b) Agarwal / GSE51082 - RMA $\log_2$ Intensity", fontweight="bold")
    axes[0, 1].set_xlabel(r"RMA $\log_2$ Normalized Intensity")
    axes[0, 1].set_xlim(3, 15)
    axes[0, 1].annotate(
        f"Skewness: {skew_gse:.2f}\nMedian: {median_gse:.1f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Golub Log2
    sns.histplot(
        sample_g_log,
        bins=50,
        kde=True,
        ax=axes[1, 0],
        color=palette["Golub"],
        edgecolor=palette["Golub"],
        stat="density",
        alpha=0.3,
    )
    axes[1, 0].set_title(r"(c) Golub - $\log_2(x + 1)$ Transform", fontweight="bold")
    axes[1, 0].set_xlabel(r"$\log_2(\text{Intensity} + 1)$")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(4, 15)
    axes[1, 0].annotate(
        f"Skewness: {skew_g_log:.2f}\nStd: {std_g_log:.2f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # GSE51082 Standardized
    sns.histplot(
        sample_gse_std,
        bins=50,
        kde=True,
        ax=axes[1, 1],
        color=gse_color,
        edgecolor=gse_color,
        stat="density",
        alpha=0.3,
    )
    axes[1, 1].set_title(r"(d) Agarwal / GSE51082 - Standardized (Z-Score)", fontweight="bold")
    axes[1, 1].set_xlabel("Standardized Z-Score")
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].annotate(
        f"Skewness: {skew_gse:.2f}\nStd: {1.00:.2f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    png_path = out_dir / "fig1_expression_distributions.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig1_expression_distributions.pdf")
    plt.close()

    print(f"    [+] Saved Figure 1 to: {png_path}")
    return png_path


# ==============================================================================
# Figure 2: Dimensionality Reduction (PCA Scree, 2D PCA, t-SNE, UMAP)
# ==============================================================================
def plot_figure_2_dimensionality_reduction(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Renders Figure 2: 4-panel dimensionality reduction analysis on Golub cohort:
    (a) PCA scree plot & cumulative explained variance
    (b) 2D PCA projection
    (c) 2D t-SNE projection
    (d) 2D UMAP projection
    """
    print("[*] Generating Figure 2: Dimensionality Reduction (PCA / t-SNE / UMAP)...")

    df = load_dataset("golub", level="genes", split="full")
    X_raw, y = get_feature_and_label_matrices(df)

    # Standardize log-transformed features
    X_log = np.log2(X_raw + 1.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 1. Full PCA for scree plot
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(explained_var)

    # 2. 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)

    # 3. 2D t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, n_iter_without_progress=300)
    X_tsne = tsne.fit_transform(X_scaled)

    # 4. 2D UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # --- Panel A: Scree Plot ---
    n_bars = 15
    ax_scree = axes[0, 0]
    ax_cum = ax_scree.twinx()

    x_idx = np.arange(1, n_bars + 1)
    ax_scree.bar(
        x_idx,
        explained_var[:n_bars],
        color="#555555" if style_cfg["style"] == "ticks" else palette[0],
        alpha=0.65,
        label="Component Variance (%)",
    )
    ax_cum.plot(
        x_idx,
        cum_var[:n_bars],
        color="#111111" if style_cfg["style"] == "ticks" else palette[1],
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Cumulative Variance (%)",
    )
    ax_scree.set_title("(a) PCA Scree Plot", fontweight="bold")
    ax_scree.set_xlabel("Principal Component (PC)")
    ax_scree.set_ylabel("Component Variance (%)")
    ax_cum.set_ylabel("Cumulative Variance (%)")
    ax_scree.set_xticks(x_idx[::2])
    ax_cum.grid(False)

    # Combine legends
    lines_1, labels_1 = ax_scree.get_legend_handles_labels()
    lines_2, labels_2 = ax_cum.get_legend_handles_labels()
    ax_scree.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=7)

    # Helper function for 2D projections
    def render_projection(ax, coords, title, xlabel, ylabel):
        for label_val, name in [(0, "ALL (Class 0)"), (1, "AML (Class 1)")]:
            mask = (y == label_val)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=palette[label_val],
                marker=markers[label_val],
                label=name,
                alpha=0.85,
                s=36,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=7.5)

    # --- Panel B: PCA 2D ---
    render_projection(
        axes[0, 1],
        X_pca,
        f"(b) 2D PCA (PC1: {explained_var[0]:.1f}%, PC2: {explained_var[1]:.1f}%)",
        f"PC1 ({explained_var[0]:.1f}%)",
        f"PC2 ({explained_var[1]:.1f}%)",
    )

    # --- Panel C: t-SNE 2D ---
    render_projection(
        axes[1, 0],
        X_tsne,
        "(c) 2D t-SNE (Perplexity=15)",
        "t-SNE Dimension 1",
        "t-SNE Dimension 2",
    )

    # --- Panel D: UMAP 2D ---
    render_projection(
        axes[1, 1],
        X_umap,
        "(d) 2D UMAP (Cosine Metric)",
        "UMAP Dimension 1",
        "UMAP Dimension 2",
    )

    plt.tight_layout()
    png_path = out_dir / "fig2_dimensionality_reduction.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig2_dimensionality_reduction.pdf")
    plt.close()

    print(f"    [+] Saved Figure 2 to: {png_path}")
    return png_path


# ==============================================================================
# Figure 3: Top Differentially Expressed Genes Clustermap
# ==============================================================================
def plot_figure_3_heatmap(
    style_cfg: Dict,
    top_n: int = 40,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Renders Figure 3: Two-way hierarchical clustering heatmap (clustermap)
    of the top 40 differentially expressed genes between ALL and AML.
    Highlights canonical biomarkers (CST3, CD33, MPO, LYZ, FAH, CD19, MME).
    """
    print(f"[*] Generating Figure 3: Top {top_n} Genes Heatmap & Clustermap...")

    df = load_dataset("golub", level="genes", split="full")
    X_raw, y = get_feature_and_label_matrices(df)
    X_log = np.log2(X_raw + 1.0)

    # Identify differentially expressed genes using both Welch's t-test and Mann-Whitney U test
    all_mask = (y == 0).values
    aml_mask = (y == 1).values

    stats_list = []
    for gene in X_log.columns:
        all_vals = X_log.loc[all_mask, gene]
        aml_vals = X_log.loc[aml_mask, gene]
        t_stat, p_welch = stats.ttest_ind(aml_vals, all_vals, equal_var=False)
        u_stat, p_mwu = stats.mannwhitneyu(aml_vals, all_vals, alternative="two-sided")
        stats_list.append({
            "gene": gene,
            "t_stat": t_stat,
            "p_welch": p_welch,
            "u_stat": u_stat,
            "p_mwu": p_mwu,
        })

    res_df = pd.DataFrame(stats_list)
    n_genes = len(res_df)

    # Benjamini-Hochberg FDR correction on both statistical tests
    for p_col, fdr_col in [("p_welch", "fdr_welch"), ("p_mwu", "fdr_mwu")]:
        res_df = res_df.sort_values(p_col).reset_index(drop=True)
        res_df[fdr_col] = res_df[p_col] * n_genes / (res_df.index + 1)
        res_df[fdr_col] = np.minimum.accumulate(res_df[fdr_col].iloc[::-1])[::-1]

    # Strictly data-driven selection of top N genes passing FDR < 0.001
    res_df = res_df.sort_values("p_welch").reset_index(drop=True)
    fdr_sig = res_df[res_df["fdr_welch"] < 0.001]
    top_genes = fdr_sig.head(top_n)["gene"].tolist()

    # Canonical biomarkers for biological interpretation
    canonical_biomarkers = {"CST3", "CD33", "MPO", "LYZ", "FAH", "CD19", "MME"}

    # Extract expression sub-matrix and compute z-score per gene across samples
    X_sub = X_log[top_genes].T  # genes x samples
    X_sub_z = X_sub.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

    # Label colors for samples (column sidebar)
    palette = style_cfg["palette"]
    col_colors = pd.Series(
        [palette[val] for val in y],
        index=X_sub_z.columns,
        name="Subtype",
    )

    # Format row labels: bold/star canonical biomarkers
    new_index = []
    for g in X_sub_z.index:
        if g in canonical_biomarkers:
            new_index.append(f"{g} *")
        else:
            new_index.append(g)
    X_sub_z.index = new_index

    cmap = style_cfg["heatmap_cmap"]

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
        f"Hierarchical Clustering of Top {top_n} Discriminative Genes (* Canonical Biomarkers)",
        fontweight="bold",
        fontsize=9.5,
        y=0.99,
    )

    # Add custom legend for sample class sidebar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette[0], edgecolor="black", label="ALL (Class 0)"),
        Patch(facecolor=palette[1], edgecolor="black", label="AML (Class 1)"),
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    g.ax_heatmap.set_ylabel("HUGO Gene Symbol", fontsize=8.5)
    g.ax_heatmap.set_xlabel("Patient Samples (n = 72)", fontsize=8.5)

    png_path = out_dir / "fig3_top_genes_heatmap.png"
    g.savefig(png_path, dpi=dpi)
    if save_pdf:
        g.savefig(out_dir / "fig3_top_genes_heatmap.pdf")
    plt.close()

    print(f"    [+] Saved Figure 3 to: {png_path}")
    return png_path


# ==============================================================================
# Figure 4: Batch Effect Diagnostics (Within-Cohort & Cross-Platform)
# ==============================================================================
def plot_figure_4_batch_effects(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Renders Figure 4: Comprehensive 4-panel diagnostic figure:
    (a) Golub train (n=38) vs. independent test (n=34) in joint PCA space (covariate shift).
    (b) Shared 5,426 HUGO gene space between Golub (Hu6800) and GSE51082 (GPL96) cross-platform PCA.
    (c) Tissue origin confounding diagnostics across cohorts (BM vs. PB with Fisher exact test).
    (d) 6-Class global leukemia transcriptomic landscape on GSE51082 (n=139 across 22,283 probes).
    """
    print("[*] Generating Figure 4: Batch Effect, Tissue Confounding & Landscape Diagnostics...")

    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    # --- Top-Left: Golub Train vs. Test Covariate Shift (a) ---
    df_train = load_dataset("golub", level="genes", split="train")
    df_test = load_dataset("golub", level="genes", split="test")

    X_train_raw, y_train = get_feature_and_label_matrices(df_train)
    X_test_raw, y_test = get_feature_and_label_matrices(df_test)

    X_train_log = np.log2(X_train_raw + 1.0)
    X_test_log = np.log2(X_test_raw + 1.0)

    scaler_g = StandardScaler()
    X_train_s = scaler_g.fit_transform(X_train_log)
    X_test_s = scaler_g.transform(X_test_log)

    pca_split = PCA(n_components=2, random_state=42)
    X_train_pca = pca_split.fit_transform(X_train_s)
    X_test_pca = pca_split.transform(X_test_s)

    ax_a = axes[0, 0]
    ax_a.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=palette["train"],
        marker=markers["train"],
        label="Training Set (n = 38)",
        alpha=0.85,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_a.scatter(
        X_test_pca[:, 0],
        X_test_pca[:, 1],
        c=palette["test"],
        marker=markers["test"],
        label="Independent Test (n = 34)",
        alpha=0.85,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_a.set_title("(a) Golub Train vs. Independent Test Shift", fontweight="bold", fontsize=8)
    ax_a.set_xlabel(f"PC1 ({pca_split.explained_variance_ratio_[0]*100:.1f}%)", fontsize=7.5)
    ax_a.set_ylabel(f"PC2 ({pca_split.explained_variance_ratio_[1]*100:.1f}%)", fontsize=7.5)
    ax_a.legend(loc="upper right", fontsize=6.5)

    # --- Top-Right: Cross-Platform Harmonization on 5,426 shared genes (b) ---
    df_harm_g = load_dataset("golub", level="harmonized")
    try:
        df_harm_sec = load_dataset("gse51082", level="harmonized")
        sec_name = "Agarwal (GSE51082)"
        sec_color = palette.get("GSE51082", palette.get("Coustan-Smith", "#888888"))
        sec_marker = markers.get("GSE51082", markers.get("Coustan-Smith", "D"))
    except FileNotFoundError:
        df_harm_sec = load_dataset("coustan_smith", level="harmonized")
        sec_name = "Coustan-Smith (GSE28497)"
        sec_color = palette.get("Coustan-Smith", "#888888")
        sec_marker = markers.get("Coustan-Smith", "D")

    X_g_raw, _ = get_feature_and_label_matrices(df_harm_g)
    X_sec_raw, _ = get_feature_and_label_matrices(df_harm_sec)

    X_g_s = StandardScaler().fit_transform(np.log2(X_g_raw + 1.0))
    X_sec_s = StandardScaler().fit_transform(X_sec_raw)

    X_combined = np.vstack([X_g_s, X_sec_s])
    pca_cross = PCA(n_components=2, random_state=42)
    X_cross_pca = pca_cross.fit_transform(X_combined)

    n_g = len(df_harm_g)
    n_sec = len(df_harm_sec)

    ax_b = axes[0, 1]
    ax_b.scatter(
        X_cross_pca[:n_g, 0],
        X_cross_pca[:n_g, 1],
        c=palette["Golub"],
        marker=markers["Golub"],
        label=f"Golub (Hu6800, n = {n_g})",
        alpha=0.8,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_b.scatter(
        X_cross_pca[n_g:, 0],
        X_cross_pca[n_g:, 1],
        c=sec_color,
        marker=sec_marker,
        label=f"{sec_name} (GPL96, n = {n_sec})",
        alpha=0.8,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_b.set_title("(b) Cross-Platform Harmonization (5,426 Genes)", fontweight="bold", fontsize=8)
    ax_b.set_xlabel(f"PC1 ({pca_cross.explained_variance_ratio_[0]*100:.1f}%)", fontsize=7.5)
    ax_b.set_ylabel(f"PC2 ({pca_cross.explained_variance_ratio_[1]*100:.1f}%)", fontsize=7.5)
    ax_b.legend(loc="upper right", fontsize=6.5)

    # --- Bottom-Left: Tissue Confounding Diagnostic (c) ---
    ax_c = axes[1, 0]
    groups = [
        ("AML (n=37)", 30, 7, 0),
        ("ALL (n=29)", 25, 4, 0),
        ("CML (n=22)*", 16, 6, 0),
        ("MDS (n=10)*", 10, 0, 0),
        ("BCLL (n=41)*", 2, 38, 1),
    ]

    y_pos = np.arange(len(groups))
    bm_pcts = []
    pb_pcts = []
    ln_pcts = []

    for name, bm, pb, ln in groups:
        total = bm + pb + ln
        bm_pcts.append((bm / total) * 100.0)
        pb_pcts.append((pb / total) * 100.0)
        ln_pcts.append((ln / total) * 100.0)

    bm_color = "#2b2b2b"
    pb_color = "#888888"
    ln_color = "#e0e0e0"

    ax_c.barh(y_pos, bm_pcts, color=bm_color, edgecolor="black", linewidth=0.5, height=0.55, label="Bone Marrow (BM)")
    ax_c.barh(y_pos, pb_pcts, left=bm_pcts, color=pb_color, edgecolor="black", linewidth=0.5, height=0.55, label="Peripheral Blood (PB)")
    ax_c.barh(y_pos, ln_pcts, left=np.array(bm_pcts) + np.array(pb_pcts), color=ln_color, edgecolor="black", linewidth=0.5, height=0.55, label="Lymph Node (LN)")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([g[0] for g in groups], fontsize=7, fontweight="medium")
    ax_c.set_xlabel("Tissue Proportion (%)", fontsize=7.5)
    ax_c.set_xlim(0, 100)
    ax_c.set_title("(c) Tissue Confounding Diagnostic", fontweight="bold", fontsize=8)
    ax_c.legend(loc="lower left", fontsize=6.0, framealpha=0.9)

    table_model = [[25, 4], [30, 7]]
    fisher_res = stats.fisher_exact(table_model)

    stats_text = (
        f"ALL vs. AML Modeling:\n"
        f"  BM: 86.2% vs. 81.1%\n"
        f"  Fisher's Exact: p={fisher_res.pvalue:.3f} (n.s.)\n"
        f"*Excluded: BCLL (92.7% PB), MDS (100% BM)"
    )
    ax_c.text(
        0.04,
        0.58,
        stats_text,
        transform=ax_c.transAxes,
        fontsize=5.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#666666", alpha=0.92),
    )

    # --- Bottom-Right: 6-Class Global Landscape PCA (d) ---
    ax_d = axes[1, 1]
    df_eda = load_dataset("gse51082", level="probes", split="full_eda")
    meta_cols = ["label", "subtype", "sample_id", "tissue", "disease", "patient_id", "title"]
    probe_cols = [c for c in df_eda.columns if c not in meta_cols]

    X_eda = df_eda[probe_cols].to_numpy(dtype=float)
    X_eda_s = StandardScaler().fit_transform(X_eda)

    pca_6cl = PCA(n_components=2, random_state=42)
    X_eda_pca = pca_6cl.fit_transform(X_eda_s)

    pc1_var = pca_6cl.explained_variance_ratio_[0] * 100.0
    pc2_var = pca_6cl.explained_variance_ratio_[1] * 100.0

    class_styles = {
        "AML": {"label": "AML (n=37, Model)", "marker": "o", "color": "#111111", "size": 30, "alpha": 0.85},
        "PRE B-ALL": {"label": "PRE B-ALL (n=17, Model)", "marker": "s", "color": "#444444", "size": 30, "alpha": 0.85},
        "T-ALL": {"label": "T-ALL (n=12, Model)", "marker": "D", "color": "#777777", "size": 30, "alpha": 0.85},
        "BCLL": {"label": "BCLL (n=41, Excl.)", "marker": "v", "color": "#aaaaaa", "size": 26, "alpha": 0.75},
        "CML": {"label": "CML (n=22, Excl.)", "marker": "^", "color": "#cccccc", "size": 26, "alpha": 0.75},
        "MDS": {"label": "MDS (n=10, Excl.)", "marker": "X", "color": "#333333", "size": 36, "alpha": 0.85},
    }

    if style_cfg.get("context") != "grayscale" and "ALL" in style_cfg["palette"] and style_cfg["palette"]["ALL"] != "#333333":
        class_styles["AML"]["color"] = "#d62728"
        class_styles["PRE B-ALL"]["color"] = "#1f77b4"
        class_styles["T-ALL"]["color"] = "#2ca02c"
        class_styles["BCLL"]["color"] = "#9467bd"
        class_styles["CML"]["color"] = "#ff7f0e"
        class_styles["MDS"]["color"] = "#8c564b"

    for disease_key, cfg in class_styles.items():
        mask = (df_eda["disease"] == disease_key).values
        ax_d.scatter(
            X_eda_pca[mask, 0],
            X_eda_pca[mask, 1],
            c=cfg["color"],
            marker=cfg["marker"],
            label=cfg["label"],
            s=cfg["size"],
            alpha=cfg["alpha"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax_d.annotate(
        "MDS -> AML continuity",
        xy=(50, -5),
        xytext=(45, -45),
        fontsize=5.5,
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#333333"),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888888", alpha=0.9),
    )

    ax_d.set_title("(d) 6-Class Global Landscape (n=139)", fontweight="bold", fontsize=8)
    ax_d.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=7.5)
    ax_d.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=7.5)
    ax_d.legend(loc="upper left", fontsize=5.5, framealpha=0.9, ncol=1)

    plt.tight_layout()
    png_path = out_dir / "fig4_batch_effect_diagnostics.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig4_batch_effect_diagnostics.pdf")
    plt.close()

    print(f"    [+] Saved Figure 4 to: {png_path}")
    return png_path

    plt.tight_layout()
    png_path = out_dir / "fig4_batch_effect_diagnostics.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig4_batch_effect_diagnostics.pdf")
    plt.close()

    print(f"    [+] Saved Figure 4 to: {png_path}")
    return png_path


# ==============================================================================
# Figure 5: Tissue Confounding Diagnostics & 6-Class Global Landscape
# ==============================================================================
def plot_figure_5_tissue_and_landscape(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Renders Figure 5:
    (a) Tissue confounding diagnostics: Cross-tabulation of tissue origin (Bone Marrow vs.
        Peripheral Blood) across ALL, AML, and excluded classes with Fisher's exact test.
    (b) 6-Class global leukemia transcriptomic landscape (PCA projection on GSE51082, n=139).
    """
    print("[*] Generating Figure 5: Tissue Confounding & 6-Class Global Landscape...")

    df_eda = load_dataset("gse51082", level="probes", split="full_eda")
    meta_cols = ["label", "subtype", "sample_id", "tissue", "disease", "patient_id", "title"]
    probe_cols = [c for c in df_eda.columns if c not in meta_cols]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # --- Left: Tissue Distribution Cross-Tabulation (a) ---
    ax_left = axes[0]

    # Define cohorts and their tissue counts
    # ALL combines PRE B-ALL (16 BM, 1 PB) and T-ALL (9 BM, 3 PB) => 25 BM, 4 PB
    # AML: 30 BM, 7 PB
    # Excluded: BCLL (2 BM, 38 PB, 1 LN), CML (16 BM, 6 PB), MDS (10 BM, 0 PB)
    groups = [
        ("AML (n=37)", 30, 7, 0),
        ("ALL (n=29)", 25, 4, 0),
        ("CML (n=22)*", 16, 6, 0),
        ("MDS (n=10)*", 10, 0, 0),
        ("BCLL (n=41)*", 2, 38, 1),
    ]

    y_pos = np.arange(len(groups))
    bm_pcts = []
    pb_pcts = []
    ln_pcts = []

    for name, bm, pb, ln in groups:
        total = bm + pb + ln
        bm_pcts.append((bm / total) * 100.0)
        pb_pcts.append((pb / total) * 100.0)
        ln_pcts.append((ln / total) * 100.0)

    bm_color = "#2b2b2b"
    pb_color = "#888888"
    ln_color = "#e0e0e0"

    ax_left.barh(y_pos, bm_pcts, color=bm_color, edgecolor="black", linewidth=0.6, height=0.55, label="Bone Marrow (BM)")
    ax_left.barh(y_pos, pb_pcts, left=bm_pcts, color=pb_color, edgecolor="black", linewidth=0.6, height=0.55, label="Peripheral Blood (PB)")
    ax_left.barh(y_pos, ln_pcts, left=np.array(bm_pcts) + np.array(pb_pcts), color=ln_color, edgecolor="black", linewidth=0.6, height=0.55, label="Lymph Node (LN)")

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels([g[0] for g in groups], fontsize=7.5, fontweight="medium")
    ax_left.set_xlabel("Tissue Origin Proportion (%)", fontsize=8)
    ax_left.set_xlim(0, 100)
    ax_left.set_title("(a) Tissue Confounding Diagnostic", fontweight="bold", fontsize=8.5)
    ax_left.legend(loc="lower left", fontsize=6.8, framealpha=0.9)

    # Compute 2x2 Fisher exact test for modeling cohort (ALL vs. AML)
    table_model = [[25, 4], [30, 7]]
    fisher_res = stats.fisher_exact(table_model)

    stats_text = (
        f"Modeling Cohort (ALL vs. AML):\n"
        f"  BM: 86.2% vs. 81.1%\n"
        f"  Fisher's Exact: OR={fisher_res.statistic:.2f}, p={fisher_res.pvalue:.3f} (n.s.)\n"
        f"  -> No tissue confounding in ALL/AML\n"
        f"*Excluded: BCLL (92.7% PB), MDS (100% BM)"
    )
    ax_left.text(
        0.04,
        0.58,
        stats_text,
        transform=ax_left.transAxes,
        fontsize=6.0,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#666666", alpha=0.92),
    )

    # --- Right: 6-Class Global Leukemia Landscape PCA (b) ---
    ax_right = axes[1]

    X = df_eda[probe_cols].to_numpy(dtype=float)
    X_s = StandardScaler().fit_transform(X)

    pca_6cl = PCA(n_components=2, random_state=42)
    X_pca = pca_6cl.fit_transform(X_s)

    pc1_var = pca_6cl.explained_variance_ratio_[0] * 100.0
    pc2_var = pca_6cl.explained_variance_ratio_[1] * 100.0

    class_styles = {
        "AML": {"label": "AML (n=37, Modeling)", "marker": "o", "color": "#111111", "size": 36, "alpha": 0.85},
        "PRE B-ALL": {"label": "PRE B-ALL (n=17, Modeling)", "marker": "s", "color": "#444444", "size": 36, "alpha": 0.85},
        "T-ALL": {"label": "T-ALL (n=12, Modeling)", "marker": "D", "color": "#777777", "size": 36, "alpha": 0.85},
        "BCLL": {"label": "BCLL (n=41, Excluded)", "marker": "v", "color": "#aaaaaa", "size": 30, "alpha": 0.75},
        "CML": {"label": "CML (n=22, Excluded)", "marker": "^", "color": "#cccccc", "size": 30, "alpha": 0.75},
        "MDS": {"label": "MDS (n=10, Excluded)", "marker": "X", "color": "#333333", "size": 42, "alpha": 0.85},
    }

    if style_cfg.get("context") != "grayscale" and "ALL" in style_cfg["palette"] and style_cfg["palette"]["ALL"] != "#333333":
        class_styles["AML"]["color"] = "#d62728"
        class_styles["PRE B-ALL"]["color"] = "#1f77b4"
        class_styles["T-ALL"]["color"] = "#2ca02c"
        class_styles["BCLL"]["color"] = "#9467bd"
        class_styles["CML"]["color"] = "#ff7f0e"
        class_styles["MDS"]["color"] = "#8c564b"

    for disease_key, cfg in class_styles.items():
        mask = (df_eda["disease"] == disease_key).values
        ax_right.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=cfg["color"],
            marker=cfg["marker"],
            label=cfg["label"],
            s=cfg["size"],
            alpha=cfg["alpha"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax_right.annotate(
        "Myeloid continuity\n(MDS -> AML)",
        xy=(50, -5),
        xytext=(55, -45),
        fontsize=6.0,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888888", alpha=0.9),
    )

    ax_right.set_title("(b) 6-Class Global Landscape (n=139)", fontweight="bold", fontsize=8.5)
    ax_right.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=8)
    ax_right.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=8)
    ax_right.legend(loc="upper left", fontsize=5.8, framealpha=0.9, ncol=1)

    plt.tight_layout()
    png_path = out_dir / "fig5_tissue_confounding_and_landscape.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig5_tissue_confounding_and_landscape.pdf")
    plt.close()

    print(f"    [+] Saved Figure 5 to: {png_path}")
    return png_path


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-grade EDA figures for Leukemia Subtype Benchmark."
    )
    parser.add_argument(
        "--style",
        choices=["grayscale", "academic", "vibrant"],
        default="grayscale",
        help="Visual style preset (default: 'grayscale' for IEEE publication).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in DPI (default: 300).",
    )
    parser.add_argument(
        "--figure",
        choices=["all", "fig1", "fig2", "fig3", "fig4", "fig5"],
        default="all",
        help="Target figure to render: 'all' (default), or individual figure id.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable parallel generation of vector PDF formats.",
    )
    args = parser.parse_args()

    style_cfg = apply_style(args.style)
    save_pdf = not args.no_pdf
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("      LEUKEMIA SUBTYPE BENCHMARK - PUBLICATION EDA ENGINE")
    print(f"      Preset: {args.style} | DPI: {args.dpi} | Format: PNG {'+ PDF' if save_pdf else ''}")
    print("=" * 65)

    if args.figure in ["all", "fig1"]:
        plot_figure_1_distributions(style_cfg, dpi=args.dpi, save_pdf=save_pdf)
    if args.figure in ["all", "fig2"]:
        plot_figure_2_dimensionality_reduction(style_cfg, dpi=args.dpi, save_pdf=save_pdf)
    if args.figure in ["all", "fig3"]:
        plot_figure_3_heatmap(style_cfg, top_n=40, dpi=args.dpi, save_pdf=save_pdf)
    if args.figure in ["all", "fig4"]:
        plot_figure_4_batch_effects(style_cfg, dpi=args.dpi, save_pdf=save_pdf)
    if args.figure in ["all", "fig5"]:
        plot_figure_5_tissue_and_landscape(style_cfg, dpi=args.dpi, save_pdf=save_pdf)

    print("\n[OK] All requested EDA figures generated successfully in outputs/figures/.\n")


if __name__ == "__main__":
    main()

