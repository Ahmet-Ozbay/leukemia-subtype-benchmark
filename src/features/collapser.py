"""
Zero-leakage probe-to-gene collapsing transformer for scikit-learn pipelines.
Guarantees that sample-dependent aggregation (e.g. max_variance) fits probe selection
strictly on training folds/partitions, eliminating feature-selection test leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ProbeCollapser(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that aggregates probe-level expression features
    into unique HUGO gene-level expression features.

    Parameters:
        mapping_df: DataFrame containing at least ['probe_id', 'gene_symbol', 'is_mapped'].
        strategy: Aggregation strategy: 'max_variance' (default), 'mean', or 'median'.
        drop_multi_mapped: If True, excludes probes mapping to multiple genes.
    """

    def __init__(
        self,
        mapping_df: pd.DataFrame,
        strategy: str = "max_variance",
        drop_multi_mapped: bool = False,
    ):
        self.mapping_df = mapping_df
        self.strategy = strategy
        self.drop_multi_mapped = drop_multi_mapped

        allowed_strategies = ("max_variance", "mean", "median")
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Allowed: {allowed_strategies}")

        # Fitted attributes
        self.selected_probes_: Dict[str, str] = {}  # gene_symbol -> probe_id (for max_variance)
        self.gene_to_probes_: Dict[str, List[str]] = {}  # gene_symbol -> [probe_ids] (for mean/median)
        self.gene_symbols_: List[str] = []
        self.feature_names_in_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: any = None) -> ProbeCollapser:
        """
        Fits probe selection strictly on training data X.

        For 'max_variance', computes probe variances strictly on X and selects the
        probe with highest sample variance for each gene symbol.
        """
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_df = X
        else:
            raise TypeError("ProbeCollapser expects X to be a pandas DataFrame with probe IDs as column names.")

        self.feature_names_in_ = feature_names

        # Filter annotation mapping to valid, expressed probes
        valid_map = self.mapping_df[self.mapping_df["is_mapped"]].copy()
        if self.drop_multi_mapped and "is_multi_mapped" in valid_map.columns:
            valid_map = valid_map[~valid_map["is_multi_mapped"]]

        available_probes = set(feature_names)
        valid_map = valid_map[valid_map["probe_id"].isin(available_probes)]

        if valid_map.empty:
            raise ValueError("No mapped probes found in input DataFrame columns.")

        if self.strategy == "max_variance":
            # Compute probe variances strictly on training fold X
            probes_to_calc = valid_map["probe_id"].unique()
            variances = X_df[probes_to_calc].var(axis=0)

            # Join variance and rank probes per gene symbol
            scored_map = valid_map.copy()
            scored_map["variance"] = scored_map["probe_id"].map(variances)

            # Sort descending by variance, tie-breaking by probe_id for strict determinism
            scored_map = scored_map.sort_values(by=["variance", "probe_id"], ascending=[False, True])
            best_probes = scored_map.groupby("gene_symbol").first()

            self.selected_probes_ = best_probes["probe_id"].to_dict()
            self.gene_symbols_ = sorted(self.selected_probes_.keys())

        elif self.strategy in ("mean", "median"):
            grouped = valid_map.groupby("gene_symbol")["probe_id"].unique().to_dict()
            self.gene_to_probes_ = {g: list(p) for g, p in grouped.items()}
            self.gene_symbols_ = sorted(self.gene_to_probes_.keys())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms probe-level DataFrame into gene-level DataFrame using fitted mappings.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ProbeCollapser expects X to be a pandas DataFrame.")

        if not self.gene_symbols_:
            raise RuntimeError("ProbeCollapser has not been fitted. Call fit() before transform().")

        if self.strategy == "max_variance":
            # Extract the frozen selected probes chosen during fit()
            target_cols = [self.selected_probes_[g] for g in self.gene_symbols_]
            collapsed = X[target_cols].copy()
            collapsed.columns = self.gene_symbols_
            return collapsed

        elif self.strategy == "mean":
            result_dict = {}
            for gene in self.gene_symbols_:
                probes = self.gene_to_probes_[gene]
                if len(probes) == 1:
                    result_dict[gene] = X[probes[0]]
                else:
                    result_dict[gene] = X[probes].mean(axis=1)
            return pd.DataFrame(result_dict, index=X.index)

        elif self.strategy == "median":
            result_dict = {}
            for gene in self.gene_symbols_:
                probes = self.gene_to_probes_[gene]
                if len(probes) == 1:
                    result_dict[gene] = X[probes[0]]
                else:
                    result_dict[gene] = X[probes].median(axis=1)
            return pd.DataFrame(result_dict, index=X.index)

        raise ValueError(f"Unhandled strategy: {self.strategy}")


def collapse_dataset(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    mapping_df: Optional[pd.DataFrame] = None,
    strategy: str = "max_variance",
    drop_multi_mapped: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Collapses probe-level datasets into gene-level datasets with provable zero leakage:
    Variance-based selection fits strictly on train_df, and the fitted mapping transforms test_df.
    Metadata columns ('label', 'subtype') are preserved.
    """
    if mapping_df is None:
        raise ValueError("mapping_df must be provided.")

    meta_cols = [c for c in ["label", "subtype"] if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    collapser = ProbeCollapser(
        mapping_df=mapping_df,
        strategy=strategy,
        drop_multi_mapped=drop_multi_mapped,
    )

    X_train = train_df[feature_cols]
    collapser.fit(X_train)
    collapsed_X_train = collapser.transform(X_train)

    collapsed_train = pd.concat([collapsed_X_train, train_df[meta_cols]], axis=1)

    collapsed_test = None
    if test_df is not None:
        X_test = test_df[feature_cols]
        collapsed_X_test = collapser.transform(X_test)
        test_meta_cols = [c for c in meta_cols if c in test_df.columns]
        collapsed_test = pd.concat([collapsed_X_test, test_df[test_meta_cols]], axis=1)

    return collapsed_train, collapsed_test
