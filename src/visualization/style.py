"""
Publication styling engine for Leukemia Subtype Benchmark.
Provides configurable style presets:
- 'grayscale' (Default): IEEE monochrome-friendly with high-contrast hatching, markers, and line styles.
- 'academic': Colorblind-safe publication palette (Deep Blue for ALL, Crimson for AML).
- 'vibrant': High-contrast modern presentation palette.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

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
            "Armstrong": "#888888",
        },
        "markers": {
            0: "o",             # Circle for ALL
            1: "s",             # Square for AML
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "Armstrong": "D",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "Armstrong": "--",
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
            "Armstrong": "#2ca02c",
        },
        "markers": {
            0: "o",
            1: "s",
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "Armstrong": "s",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "Armstrong": "--",
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
            0: "#0077b6",       # ALL: Oceanic Cyan/Blue
            1: "#e63946",       # AML: Vibrant Coral/Crimson
            "ALL": "#0077b6",
            "AML": "#e63946",
            "train": "#0096c7",
            "test": "#f77f00",
            "Golub": "#0077b6",
            "Armstrong": "#52b788",
        },
        "markers": {
            0: "o",
            1: "s",
            "ALL": "o",
            "AML": "s",
            "train": "o",
            "test": "^",
            "Golub": "o",
            "Armstrong": "^",
        },
        "linestyles": {
            0: "-",
            1: "--",
            "ALL": "-",
            "AML": "--",
            "train": "-",
            "test": "--",
            "Golub": "-",
            "Armstrong": "--",
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

