"""
Publication styling engine for Leukemia Subtype Benchmark.
Provides configurable style presets:
- 'grayscale' (Default): IEEE monochrome-friendly with high-contrast hatching, markers, and line styles.
- 'academic': Colorblind-safe publication palette (Deep Blue for ALL, Crimson for AML).
- 'vibrant': High-contrast modern presentation palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "grayscale": {
        "style": "ticks",
        "context": "paper",
        "font_family": "sans-serif",
        "font_size": 9,
        "palette": {
            0: "#333333",       # ALL: Dark gray / Charcoal
            1: "#999999",       # AML: Light/Mid gray
            "ALL": "#333333",
            "AML": "#999999",
            "train": "#222222",
            "test": "#888888",
            "Golub": "#222222",
            "GSE51082": "#888888",
            "Agarwal": "#888888",
        },
        "markers": {
            0: "o",             # Circle for ALL
            1: "s",             # Square for AML
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "GSE51082": "D",
            "Agarwal": "D",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "GSE51082": "--",
            "Agarwal": "--",
        },
        "cmap": "Greys",
        "heatmap_cmap": "mako",  # High perceptual uniformity even when printed in B&W
    },
    "academic": {
        "style": "ticks",
        "context": "paper",
        "font_family": "sans-serif",
        "font_size": 9,
        "palette": {
            0: "#1f77b4",       # ALL: Deep Blue
            1: "#d62728",       # AML: Crimson Red
            "ALL": "#1f77b4",
            "AML": "#d62728",
            "train": "#1f77b4",
            "test": "#ff7f0e",
            "Golub": "#1f77b4",
            "GSE51082": "#2ca02c",
            "Agarwal": "#2ca02c",
        },
        "markers": {
            0: "o",
            1: "s",
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "GSE51082": "s",
            "Agarwal": "s",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "GSE51082": "--",
            "Agarwal": "--",
        },
        "cmap": "Blues",
        "heatmap_cmap": "vlag",
    },
    "vibrant": {
        "style": "whitegrid",
        "context": "paper",
        "font_family": "sans-serif",
        "font_size": 9,
        "palette": {
            0: "#0077b6",       # ALL: Vibrant Blue
            1: "#e63946",       # AML: Vibrant Coral/Crimson
            "ALL": "#0077b6",
            "AML": "#e63946",
            "train": "#0096c7",
            "test": "#f77f00",
            "Golub": "#0077b6",
            "GSE51082": "#52b788",
            "Agarwal": "#52b788",
        },
        "markers": {
            0: "o",
            1: "s",
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "GSE51082": "^",
            "Agarwal": "^",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "GSE51082": "--",
            "Agarwal": "--",
        },
        "cmap": "viridis",
        "heatmap_cmap": "coolwarm",
    },
}


def apply_style(preset: str = "grayscale") -> Dict[str, Any]:
    """
    Applies matplotlib and seaborn styling rules for publication-grade rendering.
    
    Parameters:
        preset: 'grayscale' (default), 'academic', or 'vibrant'.
        
    Returns:
        The active configuration dictionary containing palette, markers, and cmaps.
    """
    if preset not in STYLE_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(STYLE_PRESETS.keys())}.")
        
    cfg = STYLE_PRESETS[preset]
    
    sns.set_theme(
        style=cfg["style"],
        context=cfg["context"],
        font=cfg["font_family"],
        rc={
            "font.size": cfg["font_size"],
            "axes.titlesize": cfg["font_size"] + 1,
            "axes.labelsize": cfg["font_size"],
            "xtick.labelsize": cfg["font_size"] - 1,
            "ytick.labelsize": cfg["font_size"] - 1,
            "legend.fontsize": cfg["font_size"] - 1,
            "figure.titlesize": cfg["font_size"] + 2,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.8,
            "grid.alpha": 0.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    
    return cfg


def save_figure(
    fig: plt.Figure,
    name: str,
    out_dir: Union[Path, str] = FIGURES_DIR,
    dpi: int = 300,
    save_pdf: bool = True,
    close: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Exports a matplotlib figure to 300+ DPI PNG and vector PDF.

    Parameters:
        fig: matplotlib Figure object or figure-like object with savefig method.
        name: Base file name without extension.
        out_dir: Directory where figures should be saved.
        dpi: Resolution for raster export (default: 300).
        save_pdf: Whether to export a scalable vector PDF alongside PNG.
        close: Whether to close the figure after saving.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{name}.png"
    pdf_path = target_dir / f"{name}.pdf" if save_pdf else None

    # Handle both plt.Figure and seaborn ClusterGrid
    fig_obj = getattr(fig, "fig", fig)

    fig_obj.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_path:
        fig_obj.savefig(pdf_path, bbox_inches="tight")
    if close:
        plt.close(fig_obj)

    return png_path, pdf_path


