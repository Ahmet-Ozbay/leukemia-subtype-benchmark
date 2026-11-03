"""
Publication-Grade Exploratory Data Analysis (EDA) & Visualization Facade
for Leukemia Subtype Benchmark (ALL vs. AML).

Generates IEEE-compliant, 300+ DPI figures referenced in MÖAT - Ara Rapor:
- Figure 1: fig1_expression_distributions (Raw vs. log2(x+1) density distributions)
- Figure 2: fig2_dimensionality_reduction (Scree plot, 2D PCA, t-SNE, UMAP)
- Figure 3: fig3_top_genes_heatmap (Top 40 differentially expressed genes clustermap)
- Figure 4: fig4_batch_effect_diagnostics (Golub train/test shift & cross-platform batch effect)
- Figure 5: fig5_tissue_confounding_and_landscape (Tissue diagnostic & 6-class global landscape)

This module serves as the primary CLI orchestrator and backward-compatible facade
over the modular `src.visualization` subpackage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.visualization.style import apply_style, save_figure
from src.visualization.utils import get_feature_and_label_matrices
from src.visualization.distributions import (
    plot_expression_distributions,
    generate_figure_1,
)
from src.visualization.dimensionality import (
    plot_dimensionality_reduction,
    generate_figure_2,
)
from src.visualization.dge import (
    CANONICAL_BIOMARKERS,
    compute_differential_expression,
    plot_top_genes_heatmap,
    generate_figure_3,
)
from src.visualization.diagnostics import (
    render_train_test_shift,
    render_cross_platform_pca,
    render_tissue_confounding,
    render_multiclass_landscape,
    plot_batch_effect_diagnostics,
    plot_tissue_and_landscape,
    generate_figure_4,
    generate_figure_5,
)


# ==============================================================================
# Legacy Compatible Figure Wrapper Functions
# ==============================================================================

def plot_figure_1_distributions(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Wrapper for Figure 1: Expression Distributions."""
    return generate_figure_1(style_cfg=style_cfg, dpi=dpi, save_pdf=save_pdf, out_dir=out_dir)


def plot_figure_2_dimensionality_reduction(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Wrapper for Figure 2: Dimensionality Reduction."""
    return generate_figure_2(style_cfg=style_cfg, dpi=dpi, save_pdf=save_pdf, out_dir=out_dir)


def plot_figure_3_heatmap(
    style_cfg: Dict,
    top_n: int = 40,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Wrapper for Figure 3: Top Differentially Expressed Genes Clustermap."""
    return generate_figure_3(style_cfg=style_cfg, top_n=top_n, dpi=dpi, save_pdf=save_pdf, out_dir=out_dir)


def plot_figure_4_batch_effects(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Wrapper for Figure 4: Batch Effect Diagnostics."""
    return generate_figure_4(style_cfg=style_cfg, dpi=dpi, save_pdf=save_pdf, out_dir=out_dir)


def plot_figure_5_tissue_and_landscape(
    style_cfg: Dict,
    dpi: int = 300,
    save_pdf: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Wrapper for Figure 5: Tissue Confounding & 6-Class Global Landscape."""
    return generate_figure_5(style_cfg=style_cfg, dpi=dpi, save_pdf=save_pdf, out_dir=out_dir)


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
