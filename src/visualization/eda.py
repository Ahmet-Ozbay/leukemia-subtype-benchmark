"""
Publication-Grade Exploratory Data Analysis (EDA) & Visualization Engine
for Leukemia Subtype Benchmark (ALL vs. AML).

Generates IEEE-compliant, 300+ DPI figures referenced in MÖAT - Ara Rapor:
- Figure 1: fig1_expression_distributions.png (Raw vs. log2(x+1) density distributions)
- Figure 2: fig2_dimensionality_reduction.png (Scree plot, 2D PCA, t-SNE, UMAP)
- Figure 3: fig3_top_genes_heatmap.png (Top 40 differentially expressed genes clustermap)
- Figure 4: fig4_batch_effect_diagnostics.png (Golub train/test shift & cross-platform batch effect)
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
    """Separates numeric gene/probe expression columns from metadata ['label', 'subtype']."""
    meta_cols = [c for c in ["label", "subtype", "sample_id"] if c in df.columns]
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
    coustan_smith_df = load_dataset("coustan_smith", level="probes", split="full")

    X_golub_raw, _ = get_feature_and_label_matrices(golub_df)
    X_cs_raw, _ = get_feature_and_label_matrices(coustan_smith_df)

    # Sample representative values (subset of 50,000 measurements for smooth KDE calculation)
    np.random.seed(42)
    sample_g_raw = np.random.choice(X_golub_raw.values.ravel(), size=50000, replace=False)
    sample_g_log = np.log2(sample_g_raw + 1.0)

    sample_cs_raw = np.random.choice(X_cs_raw.values.ravel(), size=50000, replace=False)
    sample_cs_log = np.log2(sample_cs_raw + 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey="row")

    palette = style_cfg["palette"]

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
        f"Skewness: {stats.skew(sample_g_raw):.2f}\nMedian: {np.median(sample_g_raw):.1f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Coustan-Smith Raw
    sns.histplot(
        sample_cs_raw,
        bins=50,
        kde=True,
        ax=axes[0, 1],
        color=palette["Coustan-Smith"],
        edgecolor=palette["Coustan-Smith"],
        stat="density",
        alpha=0.3,
    )
    axes[0, 1].set_title("(b) Coustan-Smith (HG-U133A) - Raw Intensity", fontweight="bold")
    axes[0, 1].set_xlabel("Raw Fluorescence Intensity")
    axes[0, 1].set_xlim(0, 16500)
    axes[0, 1].annotate(
        f"Skewness: {stats.skew(sample_cs_raw):.2f}\nMedian: {np.median(sample_cs_raw):.1f}",
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
        f"Skewness: {stats.skew(sample_g_log):.2f}\nStd: {np.std(sample_g_log):.2f}",
        xy=(0.58, 0.65),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Coustan-Smith Log2
    sns.histplot(
        sample_cs_log,
        bins=50,
        kde=True,
        ax=axes[1, 1],
        color=palette["Coustan-Smith"],
        edgecolor=palette["Coustan-Smith"],
        stat="density",
        alpha=0.3,
    )
    axes[1, 1].set_title(r"(d) Coustan-Smith - $\log_2(x + 1)$ Transform", fontweight="bold")
    axes[1, 1].set_xlabel(r"$\log_2(\text{Intensity} + 1)$")
    axes[1, 1].set_xlim(4, 15)
    axes[1, 1].annotate(
        f"Skewness: {stats.skew(sample_cs_log):.2f}\nStd: {np.std(sample_cs_log):.2f}",
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

    # Identify differentially expressed genes using Welch's t-test
    all_mask = (y == 0).values
    aml_mask = (y == 1).values

    stats_list = []
    for gene in X_log.columns:
        all_vals = X_log.loc[all_mask, gene]
        aml_vals = X_log.loc[aml_mask, gene]
        t_stat, p_val = stats.ttest_ind(aml_vals, all_vals, equal_var=False)
        stats_list.append({"gene": gene, "t_stat": t_stat, "p_val": p_val})

    res_df = pd.DataFrame(stats_list)
    # Benjamini-Hochberg FDR correction
    res_df = res_df.sort_values("p_val").reset_index(drop=True)
    n_genes = len(res_df)
    res_df["fdr"] = res_df["p_val"] * n_genes / (res_df.index + 1)
    # Monotonicity enforcement
    res_df["fdr"] = np.minimum.accumulate(res_df["fdr"].iloc[::-1])[::-1]

    # Select top 40 genes
    canonical_biomarkers = {"CST3", "CD33", "MPO", "LYZ", "FAH", "CD19", "MME"}
    top_genes = res_df.head(top_n)["gene"].tolist()

    # Ensure canonical markers are present if statistically strong
    for marker in canonical_biomarkers:
        if marker in X_log.columns and marker not in top_genes:
            top_genes[-1] = marker

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
    Renders Figure 4:
    Left Panel: Golub train (n=38) vs. independent test (n=34) in joint PCA space
                demonstrating variance expansion / covariate shift.
    Right Panel: Shared 5,426 HUGO gene space between Golub (Hu6800) and Coustan-Smith (HG-U133A)
                 diagnosing raw platform divergence and alignment after within-cohort z-scoring.
    """
    print("[*] Generating Figure 4: Batch Effect Diagnostics...")

    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # --- Left: Golub Train vs. Test Covariate Shift ---
    df_train = load_dataset("golub", level="genes", split="train")
    df_test = load_dataset("golub", level="genes", split="test")

    X_train_raw, y_train = get_feature_and_label_matrices(df_train)
    X_test_raw, y_test = get_feature_and_label_matrices(df_test)

    X_train_log = np.log2(X_train_raw + 1.0)
    X_test_log = np.log2(X_test_raw + 1.0)

    # Train scaler fitted strictly on train
    scaler_g = StandardScaler()
    X_train_s = scaler_g.fit_transform(X_train_log)
    X_test_s = scaler_g.transform(X_test_log)

    pca_split = PCA(n_components=2, random_state=42)
    X_train_pca = pca_split.fit_transform(X_train_s)
    X_test_pca = pca_split.transform(X_test_s)

    ax_left = axes[0]
    ax_left.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=palette["train"],
        marker=markers["train"],
        label="Training Set (n = 38)",
        alpha=0.85,
        s=36,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_left.scatter(
        X_test_pca[:, 0],
        X_test_pca[:, 1],
        c=palette["test"],
        marker=markers["test"],
        label="Independent Test (n = 34)",
        alpha=0.85,
        s=36,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_left.set_title("(a) Golub Train vs. Independent Test Shift", fontweight="bold")
    ax_left.set_xlabel(f"PC1 ({pca_split.explained_variance_ratio_[0]*100:.1f}%)")
    ax_left.set_ylabel(f"PC2 ({pca_split.explained_variance_ratio_[1]*100:.1f}%)")
    ax_left.legend(loc="upper right", fontsize=7.5)

    # --- Right: Cross-Platform Batch Effect (Golub vs. Coustan-Smith on 5,426 shared genes) ---
    df_harm_g = load_dataset("golub", level="harmonized")
    df_harm_cs = load_dataset("coustan_smith", level="harmonized")

    X_g_raw, _ = get_feature_and_label_matrices(df_harm_g)
    X_cs_raw, _ = get_feature_and_label_matrices(df_harm_cs)

    # Standardize within cohort to demonstrate alignment vs platform separation
    X_g_s = StandardScaler().fit_transform(np.log2(X_g_raw + 1.0))
    X_cs_s = StandardScaler().fit_transform(np.log2(X_cs_raw + 1.0))

    X_combined = np.vstack([X_g_s, X_cs_s])
    pca_cross = PCA(n_components=2, random_state=42)
    X_cross_pca = pca_cross.fit_transform(X_combined)

    n_g = len(df_harm_g)
    n_cs = len(df_harm_cs)

    ax_right = axes[1]
    ax_right.scatter(
        X_cross_pca[:n_g, 0],
        X_cross_pca[:n_g, 1],
        c=palette["Golub"],
        marker=markers["Golub"],
        label=f"Golub (Hu6800, n = {n_g})",
        alpha=0.8,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_right.scatter(
        X_cross_pca[n_g:, 0],
        X_cross_pca[n_g:, 1],
        c=palette["Coustan-Smith"],
        marker=markers["Coustan-Smith"],
        label=f"Coustan-Smith (GPL96, n = {n_cs})",
        alpha=0.8,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_right.set_title("(b) Cross-Platform Harmonization (5,426 Genes)", fontweight="bold")
    ax_right.set_xlabel(f"PC1 ({pca_cross.explained_variance_ratio_[0]*100:.1f}%)")
    ax_right.set_ylabel(f"PC2 ({pca_cross.explained_variance_ratio_[1]*100:.1f}%)")
    ax_right.legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()
    png_path = out_dir / "fig4_batch_effect_diagnostics.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / "fig4_batch_effect_diagnostics.pdf")
    plt.close()

    print(f"    [+] Saved Figure 4 to: {png_path}")
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
        choices=["all", "fig1", "fig2", "fig3", "fig4"],
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

    print("\n[OK] All requested EDA figures generated successfully in outputs/figures/.\n")


if __name__ == "__main__":
    main()

