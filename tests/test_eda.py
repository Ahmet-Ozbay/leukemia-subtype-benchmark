"""
Unit and regression test suite for EDA visualization engine.

Verifies:
1. Publication style presets ('grayscale', 'academic', 'vibrant').
2. Persistence helper `save_figure` creates valid raster PNG and vector PDF.
3. Statistical DGE analysis (Welch's t-test, Mann-Whitney U, Benjamini-Hochberg FDR).
4. Pure visualization renderers return valid Figure objects without disk I/O.
5. All 5 IEEE paper figures exist in `outputs/figures/` and are non-empty.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from src.config import FIGURES_DIR
from src.visualization.style import apply_style, save_figure, STYLE_PRESETS
from src.visualization.distributions import plot_expression_distributions
from src.visualization.dimensionality import plot_dimensionality_reduction
from src.visualization.dge import compute_differential_expression, plot_top_genes_heatmap
from src.visualization.diagnostics import (
    plot_batch_effect_diagnostics,
    plot_tissue_and_landscape,
    render_train_test_shift,
    render_cross_platform_pca,
    render_tissue_confounding,
    render_multiclass_landscape,
)


@pytest.fixture
def synthetic_expression_data():
    """Generates synthetic expression DataFrames for fast unit testing."""
    rng = np.random.RandomState(42)
    n_samples = 12
    n_genes = 25

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    # Ensure one canonical biomarker is present for labeling tests
    gene_names[0] = "CST3"
    gene_names[1] = "CD19"

    # Half ALL (0), half AML (1)
    labels = np.array([0] * 6 + [1] * 6)
    subtypes = ["B-ALL"] * 6 + ["AML"] * 6

    # Expression data with slight shift between classes
    data_all = rng.normal(loc=5.0, scale=1.0, size=(6, n_genes))
    data_aml = rng.normal(loc=7.0, scale=1.0, size=(6, n_genes))
    expr_matrix = np.vstack([data_all, data_aml])

    df = pd.DataFrame(expr_matrix, columns=gene_names)
    df["label"] = labels
    df["subtype"] = subtypes
    df["sample_id"] = [f"SAMPLE_{i}" for i in range(n_samples)]
    df["disease"] = ["PRE B-ALL"] * 3 + ["T-ALL"] * 3 + ["AML"] * 6

    df_secondary = pd.DataFrame(
        rng.normal(loc=6.0, scale=1.2, size=(8, n_genes)),
        columns=gene_names,
    )
    df_secondary["label"] = [0] * 4 + [1] * 4
    df_secondary["sample_id"] = [f"SEC_{i}" for i in range(8)]

    return df, df_secondary


def test_style_presets_configuration():
    """Verifies that all three requested style presets are valid and accessible."""
    for name in ["grayscale", "academic", "vibrant"]:
        cfg = apply_style(name)
        assert "palette" in cfg
        assert "markers" in cfg
        assert 0 in cfg["palette"]
        assert 1 in cfg["palette"]


def test_save_figure_creates_png_and_pdf(tmp_path):
    """Verifies save_figure exports 300 DPI PNG and vector PDF properly."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot([0, 1], [0, 1])

    png_path, pdf_path = save_figure(
        fig,
        name="test_figure",
        out_dir=tmp_path,
        dpi=150,
        save_pdf=True,
        close=True,
    )

    assert png_path.exists()
    assert png_path.stat().st_size > 500
    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 500


def test_compute_differential_expression(synthetic_expression_data):
    """Verifies statistical DGE calculations, column structure, and FDR bounds."""
    df, _ = synthetic_expression_data
    from src.visualization.utils import get_feature_and_label_matrices

    X, y = get_feature_and_label_matrices(df)
    res_df = compute_differential_expression(X, y)

    expected_cols = ["gene", "t_stat", "p_welch", "fdr_welch", "u_stat", "p_mwu", "fdr_mwu"]
    for col in expected_cols:
        assert col in res_df.columns

    assert len(res_df) == X.shape[1]
    # Check that FDR values are valid probabilities [0, 1]
    assert (res_df["fdr_welch"] >= 0.0).all()
    assert (res_df["fdr_welch"] <= 1.0).all()
    # Check sorting by p_welch ascending
    assert res_df["p_welch"].is_monotonic_increasing


def test_plot_expression_distributions_synthetic(synthetic_expression_data):
    """Verifies plot_expression_distributions returns a valid 4-panel Figure."""
    df_a, df_b = synthetic_expression_data
    fig = plot_expression_distributions(df_a, df_b, sample_size=100)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_dimensionality_reduction_synthetic(synthetic_expression_data):
    """Verifies plot_dimensionality_reduction renders 4 panels (scree, PCA, t-SNE, UMAP)."""
    df, _ = synthetic_expression_data
    fig = plot_dimensionality_reduction(df)
    assert isinstance(fig, plt.Figure)
    # 4 panels, with scree plot having twinx => 5 axes
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_top_genes_heatmap_synthetic(synthetic_expression_data):
    """Verifies plot_top_genes_heatmap generates a valid Seaborn clustermap."""
    df, _ = synthetic_expression_data
    g = plot_top_genes_heatmap(df, top_n=10)
    assert isinstance(g, sns.matrix.ClusterGrid)
    assert g.fig is not None
    plt.close(g.fig)


def test_diagnostics_panels_and_figures_synthetic(synthetic_expression_data):
    """Verifies composable diagnostic panels and composite Figures 4 and 5."""
    df_a, df_b = synthetic_expression_data
    style_cfg = apply_style("grayscale")

    # Test Figure 4 (4 panels)
    fig4 = plot_batch_effect_diagnostics(
        df_train=df_a,
        df_test=df_a,
        df_harm_primary=df_a,
        df_harm_secondary=df_b,
        df_eda=df_a,
        style_cfg=style_cfg,
    )
    assert isinstance(fig4, plt.Figure)
    assert len(fig4.axes) == 4
    plt.close(fig4)

    # Test Figure 5 (2 panels)
    fig5 = plot_tissue_and_landscape(df_a, style_cfg=style_cfg)
    assert isinstance(fig5, plt.Figure)
    assert len(fig5.axes) == 2
    plt.close(fig5)


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
