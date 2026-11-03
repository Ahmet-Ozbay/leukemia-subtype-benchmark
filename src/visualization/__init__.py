"""
Publication-Grade Visualization Engine for Leukemia Subtype Benchmark.

Provides:
- Publication styling presets ('grayscale', 'academic', 'vibrant') via `apply_style`.
- Centralized 300+ DPI figure export via `save_figure`.
- Pure visualization renderers returning matplotlib Figures.
- Reusable diagnostic panels and differential expression statistics.
- High-level runners for IEEE paper figures (Figure 1 through 5).
"""

from __future__ import annotations

from src.visualization.style import STYLE_PRESETS, apply_style, save_figure
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

__all__ = [
    # Styling & Persistence
    "STYLE_PRESETS",
    "apply_style",
    "save_figure",
    # Utilities
    "get_feature_and_label_matrices",
    "CANONICAL_BIOMARKERS",
    "compute_differential_expression",
    # Pure Renderers
    "plot_expression_distributions",
    "plot_dimensionality_reduction",
    "plot_top_genes_heatmap",
    "plot_batch_effect_diagnostics",
    "plot_tissue_and_landscape",
    # Diagnostic Panels
    "render_train_test_shift",
    "render_cross_platform_pca",
    "render_tissue_confounding",
    "render_multiclass_landscape",
    # High-level Runners
    "generate_figure_1",
    "generate_figure_2",
    "generate_figure_3",
    "generate_figure_4",
    "generate_figure_5",
]

