"""
Re-exports probe-to-gene collapsing utilities from src.features.collapser.
"""

from src.features.collapser import ProbeCollapser, collapse_dataset

__all__ = ["ProbeCollapser", "collapse_dataset"]
