"""
Figure 1: Expression Distributions (Raw vs. Log2(x + 1)).

Compares raw microarray fluorescence intensity against log2(x + 1) transformed
values for the primary Golub cohort and RMA log2 normalized values for GSE51082.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from src.config import FIGURES_DIR
from src.data.io import load_dataset
from src.visualization.style import apply_style, save_figure
from src.visualization.utils import get_feature_and_label_matrices


def plot_expression_distributions(
    df_golub: pd.DataFrame,
    df_gse: pd.DataFrame,
    style_cfg: Optional[Dict] = None,
    sample_size: int = 50000,
    seed: int = 42,
) -> plt.Figure:
    """
    Renders 4-panel expression density distributions comparing raw microarray
    fluorescence intensity against log2-transformed / standardized values.

    Parameters:
        df_golub: Golub expression DataFrame (probes level).
        df_gse: GSE51082 expression DataFrame (probes level).
        style_cfg: Optional style configuration dictionary from apply_style.
        sample_size: Maximum measurements sampled for KDE rendering.
        seed: Random seed for subsampling reproducibility.

    Returns:
        matplotlib.figure.Figure: The rendered 4-panel figure.
    """
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    X_golub_raw, _ = get_feature_and_label_matrices(df_golub)
    X_gse_raw, _ = get_feature_and_label_matrices(df_gse)

    full_g_raw = X_golub_raw.to_numpy(dtype=float).ravel()
    full_gse_rma = X_gse_raw.to_numpy(dtype=float).ravel()

    median_g = float(np.median(full_g_raw))
    skew_g = float(stats.skew(full_g_raw))
    median_gse = float(np.median(full_gse_rma))
    skew_gse = float(stats.skew(full_gse_rma))

    full_g_log = np.log2(np.maximum(full_g_raw, 0) + 1.0)
    std_g_log = float(np.std(full_g_log))
    skew_g_log = float(stats.skew(full_g_log))
    std_gse = float(np.std(full_gse_rma)) if np.std(full_gse_rma) > 0 else 1.0

    # Subsample for smooth KDE rendering
    rng = np.random.RandomState(seed)
    n_sample_g = min(sample_size, len(full_g_raw))
    sample_g_raw = rng.choice(full_g_raw, size=n_sample_g, replace=False)
    sample_g_log = np.log2(np.maximum(sample_g_raw, 0) + 1.0)

    n_sample_gse = min(sample_size, len(full_gse_rma))
    sample_gse_rma = rng.choice(full_gse_rma, size=n_sample_gse, replace=False)
    sample_gse_std = (sample_gse_rma - np.mean(full_gse_rma)) / std_gse

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    palette = style_cfg["palette"]
    golub_color = palette.get("Golub", "#222222")
    gse_color = palette.get("GSE51082", "#888888")

    # (a) Golub Raw
    sns.histplot(
        sample_g_raw,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color=golub_color,
        edgecolor=golub_color,
        stat="density",
        alpha=0.3,
    )
    axes[0, 0].set_title(
        f"(a) Golub (Hu6800) - Raw Intensity\n(Skew: {skew_g:.2f}, Median: {median_g:.1f})",
        fontweight="bold",
        fontsize=8,
    )
    axes[0, 0].set_xlabel("Raw Fluorescence Intensity")
    axes[0, 0].set_ylabel(r"Density ($\times 10^{-3}$)")
    axes[0, 0].set_xlim(0, 16500)
    axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 1e3:.1f}"))

    # (b) GSE51082 RMA (Pre-normalized log2)
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
    axes[0, 1].set_title(
        rf"(b) Agarwal / GSE51082 - RMA $\log_2$ Intensity" + "\n" + f"(Skew: {skew_gse:.2f}, Median: {median_gse:.1f})",
        fontweight="bold",
        fontsize=8,
    )
    axes[0, 1].set_xlabel(r"RMA $\log_2$ Normalized Intensity")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_xlim(3, 15)

    # (c) Golub Log2
    sns.histplot(
        sample_g_log,
        bins=50,
        kde=True,
        ax=axes[1, 0],
        color=golub_color,
        edgecolor=golub_color,
        stat="density",
        alpha=0.3,
    )
    axes[1, 0].set_title(
        rf"(c) Golub - $\log_2(x + 1)$ Transform" + "\n" + f"(Skew: {skew_g_log:.2f}, Std: {std_g_log:.2f})",
        fontweight="bold",
        fontsize=8,
    )
    axes[1, 0].set_xlabel(r"$\log_2(\text{Intensity} + 1)$")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(4, 15)

    # (d) GSE51082 Standardized
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
    axes[1, 1].set_title(
        rf"(d) Agarwal / GSE51082 - Standardized (Z-Score)" + "\n" + f"(Skew: {skew_gse:.2f}, Std: 1.00)",
        fontweight="bold",
        fontsize=8,
    )
    axes[1, 1].set_xlabel("Standardized Z-Score")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_xlim(-4, 4)

    plt.tight_layout()
    return fig


def generate_figure_1(
    style_cfg: Optional[Dict] = None,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Union[Path, str] = FIGURES_DIR,
) -> Path:
    """
    Runner for Figure 1: loads benchmark datasets, renders Figure 1,
    and persists 300 DPI PNG and vector PDF.
    """
    print("[*] Generating Figure 1: Expression Distributions...")
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    golub_df = load_dataset("golub", level="probes", split="full")
    gse_df = load_dataset("gse51082", level="probes", split="all_aml")

    fig = plot_expression_distributions(golub_df, gse_df, style_cfg=style_cfg)
    png_path, _ = save_figure(
        fig,
        "fig1_expression_distributions",
        out_dir=out_dir,
        dpi=dpi,
        save_pdf=save_pdf,
        close=True,
    )
    print(f"    [+] Saved Figure 1 to: {png_path}")
    return png_path

