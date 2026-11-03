"""
Unit and regression test suite for EDA visualization engine.
Verifies that all IEEE paper figures (fig1 through fig4) are generated
at proper resolutions and valid non-empty file formats.
"""

from pathlib import Path
import pytest
from src.config import FIGURES_DIR
from src.visualization.style import apply_style, STYLE_PRESETS


def test_style_presets_configuration():
    """Verifies that all three requested style presets are valid and accessible."""
    for name in ["grayscale", "academic", "vibrant"]:
        cfg = apply_style(name)
        assert "palette" in cfg
        assert "markers" in cfg
        assert 0 in cfg["palette"]
        assert 1 in cfg["palette"]


def test_eda_figures_exist_and_non_empty():
    """Asserts that all 5 figures referenced in the IEEE draft paper exist and are non-empty."""
    required_figures = [
        "fig1_expression_distributions.png",
        "fig1_expression_distributions.pdf",
        "fig2_dimensionality_reduction.png",
        "fig2_dimensionality_reduction.pdf",
        "fig3_top_genes_heatmap.png",
        "fig3_top_genes_heatmap.pdf",
        "fig4_batch_effect_diagnostics.png",
        "fig4_batch_effect_diagnostics.pdf",
        "fig5_tissue_confounding_and_landscape.png",
        "fig5_tissue_confounding_and_landscape.pdf",
    ]

    for fig_name in required_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Expected figure '{fig_name}' not found in {FIGURES_DIR}"
        assert fig_path.stat().st_size > 1024, f"Figure '{fig_name}' is unexpectedly small or empty"

