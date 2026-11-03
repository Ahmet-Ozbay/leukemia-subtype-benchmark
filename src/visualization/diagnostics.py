"""
Figures 4 & 5: Batch Diagnostics, Covariate Shift, Tissue Confounding, and Global Landscape.

Provides modular, composable panel renderers for:
- (a) Golub Train (n=38) vs. Test (n=34) covariate shift PCA.
- (b) Cross-platform Affymetrix harmonization PCA on shared 5,426 HUGO genes.
- (c) Tissue source confounding cross-tabulation (BM vs. PB) with Fisher's exact test.
- (d) 6-Class global leukemia transcriptomic landscape PCA on GSE51082 (n=139).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR
from src.data.io import load_dataset
from src.visualization.style import apply_style, save_figure
from src.visualization.utils import get_feature_and_label_matrices


# ==============================================================================
# Composable Panel Renderers
# ==============================================================================

def render_train_test_shift(
    ax: plt.Axes,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    style_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """Renders Panel A: Golub Train vs. Independent Test Covariate Shift PCA."""
    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    X_train_raw, _ = get_feature_and_label_matrices(df_train)
    X_test_raw, _ = get_feature_and_label_matrices(df_test)

    X_train_log = np.log2(np.maximum(X_train_raw.to_numpy(dtype=float), 0) + 1.0)
    X_test_log = np.log2(np.maximum(X_test_raw.to_numpy(dtype=float), 0) + 1.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_log)
    X_test_s = scaler.transform(X_test_log)

    n_comp = min(2, X_train_s.shape[0], X_train_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    pc1_var = pca.explained_variance_ratio_[0] * 100.0 if n_comp >= 1 else 0.0
    pc2_var = pca.explained_variance_ratio_[1] * 100.0 if n_comp >= 2 else 0.0

    ax.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1] if n_comp >= 2 else np.zeros(len(X_train_pca)),
        c=palette.get("train", "#222222"),
        marker=markers.get("train", "o"),
        label=f"Training Set (n = {len(df_train)})",
        alpha=0.85,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.scatter(
        X_test_pca[:, 0],
        X_test_pca[:, 1] if n_comp >= 2 else np.zeros(len(X_test_pca)),
        c=palette.get("test", "#888888"),
        marker=markers.get("test", "^"),
        label=f"Independent Test (n = {len(df_test)})",
        alpha=0.85,
        s=32,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("(a) Golub Train vs. Independent Test Shift", fontweight="bold", fontsize=8, pad=18)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=7.5)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=7.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=6.8,
    )

    return pc1_var, pc2_var


def render_cross_platform_pca(
    ax: plt.Axes,
    df_harm_primary: pd.DataFrame,
    df_harm_secondary: pd.DataFrame,
    style_cfg: Dict[str, Any],
    sec_name: str = "Agarwal (GSE51082)",
) -> Tuple[float, float]:
    """Renders Panel B: Cross-Platform Harmonization PCA on shared genes."""
    palette = style_cfg["palette"]
    markers = style_cfg["markers"]

    X_p_raw, _ = get_feature_and_label_matrices(df_harm_primary)
    X_s_raw, _ = get_feature_and_label_matrices(df_harm_secondary)

    # Standardize primary (log2 transformed if raw) and secondary (already log2 RMA)
    X_p_s = StandardScaler().fit_transform(np.log2(np.maximum(X_p_raw.to_numpy(dtype=float), 0) + 1.0))
    X_s_s = StandardScaler().fit_transform(X_s_raw.to_numpy(dtype=float))

    X_combined = np.vstack([X_p_s, X_s_s])
    n_comp = min(2, X_combined.shape[0], X_combined.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_cross_pca = pca.fit_transform(X_combined)

    pc1_var = pca.explained_variance_ratio_[0] * 100.0 if n_comp >= 1 else 0.0
    pc2_var = pca.explained_variance_ratio_[1] * 100.0 if n_comp >= 2 else 0.0

    n_p = len(df_harm_primary)
    n_s = len(df_harm_secondary)

    sec_color = palette.get("GSE51082", palette.get("Coustan-Smith", "#888888"))
    sec_marker = markers.get("GSE51082", markers.get("Coustan-Smith", "D"))

    ax.scatter(
        X_cross_pca[:n_p, 0],
        X_cross_pca[:n_p, 1] if n_comp >= 2 else np.zeros(n_p),
        c=palette.get("Golub", "#222222"),
        marker=markers.get("Golub", "o"),
        label=f"Golub (Hu6800, n = {n_p})",
        alpha=0.8,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.scatter(
        X_cross_pca[n_p:, 0],
        X_cross_pca[n_p:, 1] if n_comp >= 2 else np.zeros(n_s),
        c=sec_color,
        marker=sec_marker,
        label=f"{sec_name} (GPL96, n = {n_s})",
        alpha=0.8,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )
    shared_genes_count = X_p_raw.shape[1]
    ax.set_title(f"(b) Cross-Platform Harmonization ({shared_genes_count:,} Genes)", fontweight="bold", fontsize=8, pad=18)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=7.5)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=7.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=6.8,
    )

    return pc1_var, pc2_var


def render_tissue_confounding(
    ax: plt.Axes,
    style_cfg: Dict[str, Any],
    title_prefix: str = "(c)",
    compact: bool = False,
) -> stats._mannwhitneyu.MannwhitneyuResult | Any:
    """
    Renders Tissue Confounding Diagnostic: Bone Marrow vs Peripheral Blood
    cross-tabulation across acute and non-acute leukemia classes with Fisher's exact test.
    """
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

    for _, bm, pb, ln in groups:
        total = bm + pb + ln
        bm_pcts.append((bm / total) * 100.0)
        pb_pcts.append((pb / total) * 100.0)
        ln_pcts.append((ln / total) * 100.0)

    bm_color = "#2b2b2b"
    pb_color = "#888888"
    ln_color = "#e0e0e0"

    height = 0.55
    ax.barh(y_pos, bm_pcts, color=bm_color, edgecolor="black", linewidth=0.5, height=height, label="Bone Marrow (BM)")
    ax.barh(y_pos, pb_pcts, left=bm_pcts, color=pb_color, edgecolor="black", linewidth=0.5, height=height, label="Peripheral Blood (PB)")
    ax.barh(y_pos, ln_pcts, left=np.array(bm_pcts) + np.array(pb_pcts), color=ln_color, edgecolor="black", linewidth=0.5, height=height, label="Lymph Node (LN)")

    # 2x2 Fisher's exact test on acute leukemia modeling cohort (ALL vs. AML)
    # ALL: 25 BM, 4 PB | AML: 30 BM, 7 PB
    table_model = [[25, 4], [30, 7]]
    fisher_res = stats.fisher_exact(table_model)

    font_adj = 0.0 if not compact else 0.5
    ax.set_yticks(y_pos)
    ax.set_yticklabels([g[0] for g in groups], fontsize=7 + font_adj, fontweight="medium")
    ax.set_xlabel("Tissue Proportion (%)", fontsize=7.5 + font_adj)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 4.55)
    ax.set_title(
        f"{title_prefix} Tissue Confounding Diagnostic\n"
        f"(ALL vs. AML: Fisher p = {fisher_res.pvalue:.3f}, n.s.)",
        fontweight="bold",
        fontsize=7.8 + font_adj,
        pad=18,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=6.2 + font_adj,
    )

    return fisher_res


def render_multiclass_landscape(
    ax: plt.Axes,
    df_eda: pd.DataFrame,
    style_cfg: Dict[str, Any],
    title_prefix: str = "(d)",
    compact: bool = False,
) -> Tuple[float, float]:
    """Renders 6-Class Global Transcriptomic Landscape PCA on GSE51082."""
    meta_cols = ["label", "subtype", "sample_id", "tissue", "disease", "patient_id", "title"]
    probe_cols = [c for c in df_eda.columns if c not in meta_cols]

    X = df_eda[probe_cols].to_numpy(dtype=float)
    X_s = StandardScaler().fit_transform(X)

    n_comp = min(2, X_s.shape[0], X_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_s)

    pc1_var = pca.explained_variance_ratio_[0] * 100.0 if n_comp >= 1 else 0.0
    pc2_var = pca.explained_variance_ratio_[1] * 100.0 if n_comp >= 2 else 0.0

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
        if "disease" in df_eda.columns:
            mask = (df_eda["disease"] == disease_key).values
        else:
            mask = np.zeros(len(df_eda), dtype=bool)

        if not np.any(mask):
            continue

        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1] if n_comp >= 2 else np.zeros(np.sum(mask)),
            c=cfg["color"],
            marker=cfg["marker"],
            label=cfg["label"],
            s=cfg["size"],
            alpha=cfg["alpha"],
            edgecolor="black",
            linewidth=0.5,
        )

    font_adj = 0.0 if not compact else 0.5
    ax.set_title(f"{title_prefix} 6-Class Global Landscape (n={len(df_eda)})", fontweight="bold", fontsize=8 + font_adj, pad=28)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=7.5 + font_adj)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=7.5 + font_adj)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=5.5 + font_adj,
    )

    return pc1_var, pc2_var


# ==============================================================================
# Full Composite Figures (Figures 4 & 5)
# ==============================================================================

def plot_batch_effect_diagnostics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_harm_primary: pd.DataFrame,
    df_harm_secondary: pd.DataFrame,
    df_eda: pd.DataFrame,
    style_cfg: Optional[Dict[str, Any]] = None,
    sec_name: str = "Agarwal (GSE51082)",
) -> plt.Figure:
    """
    Renders Figure 4: Comprehensive 4-panel diagnostic figure:
    (a) Golub train vs. test covariate shift PCA.
    (b) Cross-platform harmonization PCA on shared genes.
    (c) Tissue source confounding cross-tabulation with Fisher's exact test.
    (d) 6-Class global leukemia transcriptomic landscape.
    """
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    render_train_test_shift(axes[0, 0], df_train, df_test, style_cfg)
    render_cross_platform_pca(axes[0, 1], df_harm_primary, df_harm_secondary, style_cfg, sec_name=sec_name)
    render_tissue_confounding(axes[1, 0], style_cfg, title_prefix="(c)", compact=False)
    render_multiclass_landscape(axes[1, 1], df_eda, style_cfg, title_prefix="(d)", compact=False)

    plt.tight_layout()
    return fig


def plot_tissue_and_landscape(
    df_eda: pd.DataFrame,
    style_cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Renders Figure 5: Standalone 2-panel figure for tissue diagnostics and landscape:
    (a) Tissue confounding diagnostics with Fisher's exact test.
    (b) 6-Class global leukemia transcriptomic landscape PCA.
    """
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    render_tissue_confounding(axes[0], style_cfg, title_prefix="(a)", compact=True)
    render_multiclass_landscape(axes[1], df_eda, style_cfg, title_prefix="(b)", compact=True)

    plt.tight_layout()
    return fig


# ==============================================================================
# Runners for Figures 4 & 5
# ==============================================================================

def generate_figure_4(
    style_cfg: Optional[Dict[str, Any]] = None,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Union[Path, str] = FIGURES_DIR,
) -> Path:
    """
    Runner for Figure 4: loads benchmark splits and harmonized cohorts,
    renders Figure 4, and persists 300 DPI PNG and vector PDF.
    """
    print("[*] Generating Figure 4: Batch Effect, Tissue Confounding & Landscape Diagnostics...")
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    df_train = load_dataset("golub", level="genes", split="train")
    df_test = load_dataset("golub", level="genes", split="test")
    df_harm_g = load_dataset("golub", level="harmonized")

    try:
        df_harm_sec = load_dataset("gse51082", level="harmonized")
        sec_name = "Agarwal (GSE51082)"
    except FileNotFoundError:
        df_harm_sec = load_dataset("coustan_smith", level="harmonized")
        sec_name = "Coustan-Smith (GSE28497)"

    df_eda = load_dataset("gse51082", level="probes", split="full_eda")

    fig = plot_batch_effect_diagnostics(
        df_train=df_train,
        df_test=df_test,
        df_harm_primary=df_harm_g,
        df_harm_secondary=df_harm_sec,
        df_eda=df_eda,
        style_cfg=style_cfg,
        sec_name=sec_name,
    )

    png_path, _ = save_figure(
        fig,
        "fig4_batch_effect_diagnostics",
        out_dir=out_dir,
        dpi=dpi,
        save_pdf=save_pdf,
        close=True,
    )
    print(f"    [+] Saved Figure 4 to: {png_path}")
    return png_path


def generate_figure_5(
    style_cfg: Optional[Dict[str, Any]] = None,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Union[Path, str] = FIGURES_DIR,
) -> Path:
    """
    Runner for Figure 5: loads GSE51082 full cohort, renders Figure 5,
    and persists 300 DPI PNG and vector PDF.
    """
    print("[*] Generating Figure 5: Tissue Confounding & 6-Class Global Landscape...")
    if style_cfg is None:
        style_cfg = apply_style("grayscale")

    df_eda = load_dataset("gse51082", level="probes", split="full_eda")

    fig = plot_tissue_and_landscape(df_eda, style_cfg=style_cfg)
    png_path, _ = save_figure(
        fig,
        "fig5_tissue_confounding_and_landscape",
        out_dir=out_dir,
        dpi=dpi,
        save_pdf=save_pdf,
        close=True,
    )
    print(f"    [+] Saved Figure 5 to: {png_path}")
    return png_path

