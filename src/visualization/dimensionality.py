"""
Figure 2: Dimensionality Reduction & Manifold Learning (PCA Scree, 2D PCA, t-SNE, UMAP).

Analyzes linear and non-linear unsupervised separability of ALL vs. AML
transcription profiles across high-dimensional microarray feature space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

from src.config import FIGURES_DIR
from src.data.io import load_dataset
from src.visualization.style import apply_style, save_figure
from src.visualization.utils import get_feature_and_label_matrices


def plot_dimensionality_reduction(
    df: pd.DataFrame,
    style_cfg: Optional[Dict] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Renders 4-panel dimensionality reduction analysis:
    (a) PCA scree plot & cumulative explained variance
    (b) 2D PCA projection
    (c) 2D t-SNE projection
    (d) 2D UMAP projection

    Parameters:
        df: Expression DataFrame with sample metadata and labels.
        style_cfg: Optional style configuration dictionary from apply_style.
        seed: Random seed for reproducibility.

    Returns:
        matplotlib.figure.Figure: The rendered 4-panel figure.
    """
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    X_raw, y = get_feature_and_label_matrices(df)

    # Standardize log-transformed features
    X_log = np.log2(np.maximum(X_raw.to_numpy(dtype=float), 0) + 1.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    n_samples, n_features = X_scaled.shape

    # 1. Full PCA for scree plot
    pca_full = PCA(random_state=seed)
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(explained_var)

    # 2. 2D PCA
    pca_2d = PCA(n_components=2, random_state=seed)
    X_pca = pca_2d.fit_transform(X_scaled)

    # 3. 2D t-SNE (adjust perplexity adaptively for smaller cohorts/tests)
    perplexity = min(15, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        n_iter_without_progress=300,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # 4. 2D UMAP (adjust n_neighbors adaptively for smaller cohorts/tests)
    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    X_umap = reducer.fit_transform(X_scaled)

    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # --- Panel A: Scree Plot ---
    n_bars = min(15, len(explained_var))
    ax_scree = axes[0, 0]
    ax_cum = ax_scree.twinx()

    x_idx = np.arange(1, n_bars + 1)
    ax_scree.bar(
        x_idx,
        explained_var[:n_bars],
        color="#555555" if style_cfg.get("style") == "ticks" else palette.get(0, "#333333"),
        alpha=0.65,
        label="Component Variance (%)",
    )
    ax_cum.plot(
        x_idx,
        cum_var[:n_bars],
        color="#111111" if style_cfg.get("style") == "ticks" else palette.get(1, "#999999"),
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Cumulative Variance (%)",
    )
    ax_scree.set_title("(a) PCA Scree Plot", fontweight="bold")
    ax_scree.set_xlabel("Principal Component (PC)")
    ax_scree.set_ylabel("Component Variance (%)")
    ax_cum.set_ylabel("Cumulative Variance (%)")
    ax_scree.set_xticks(x_idx[::2] if len(x_idx) > 4 else x_idx)
    ax_cum.grid(False)

    lines_1, labels_1 = ax_scree.get_legend_handles_labels()
    lines_2, labels_2 = ax_cum.get_legend_handles_labels()
    ax_scree.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=7)

    # Helper function for 2D projections
    def render_projection(ax, coords, title, xlabel, ylabel):
        for label_val, name in [(0, "ALL (Class 0)"), (1, "AML (Class 1)")]:
            mask = (y == label_val).values
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=palette.get(label_val, "#333333"),
                marker=markers.get(label_val, "o"),
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
        f"(c) 2D t-SNE (Perplexity={perplexity})",
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
    return fig


def generate_figure_2(
    style_cfg: Optional[Dict] = None,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Union[Path, str] = FIGURES_DIR,
) -> Path:
    """
    Runner for Figure 2: loads Golub gene-level data, renders Figure 2,
    and persists 300 DPI PNG and vector PDF.
    """
    print("[*] Generating Figure 2: Dimensionality Reduction (PCA / t-SNE / UMAP)...")
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    df = load_dataset("golub", level="genes", split="full")
    fig = plot_dimensionality_reduction(df, style_cfg=style_cfg)
    png_path, _ = save_figure(
        fig,
        "fig2_dimensionality_reduction",
        out_dir=out_dir,
        dpi=dpi,
        save_pdf=save_pdf,
        close=True,
    )
    print(f"    [+] Saved Figure 2 to: {png_path}")
    return png_path

