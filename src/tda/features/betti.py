"""Betti curve and persistence summary feature extraction.

Betti curve for dimension k:
  Given a persistence diagram D_k = {(b_i, d_i)}, the Betti number at
  filtration value t is  β_k(t) = #{i : b_i ≤ t < d_i}.
  We evaluate this on a uniform grid of t values to get a fixed-length vector.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.tda.ph.base import PersistenceDiagram
from src.tda.features.base import FeatureExtractor, TDAFeatures


class BettiFeatureExtractor(FeatureExtractor):
    """Extract Betti curves and optional summary statistics.

    Args:
        grid_size: number of grid points for Betti curves.
        grid_range: (lo, hi) for the grid, or "auto" to use data range.
        max_dim: maximum homology dimension to extract.
        summary_stats: whether to append persistence summary statistics.
    """

    def __init__(self, grid_size: int = 100,
                 grid_range: Union[str, Tuple[float, float]] = "auto",
                 max_dim: int = 1,
                 summary_stats: bool = True):
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.max_dim = max_dim
        self.summary_stats = summary_stats

    def extract(self, diagrams: Dict[int, PersistenceDiagram]) -> TDAFeatures:
        lo, hi = self._resolve_range(diagrams)
        grid = np.linspace(lo, hi, self.grid_size)
        vectors = []
        names = []

        for dim in range(self.max_dim + 1):
            diag = diagrams.get(dim)
            betti_curve = self._betti_curve(diag, grid)
            vectors.append(betti_curve)
            names.extend([f"betti_{dim}_t{i}" for i in range(self.grid_size)])

            if self.summary_stats and diag is not None and len(diag.pairs) > 0:
                stats, stat_names = self._persistence_stats(diag, dim)
                vectors.append(stats)
                names.extend(stat_names)
            elif self.summary_stats:
                # Zero stats if no pairs
                stat_names = [f"H{dim}_{s}" for s in
                              ["n_pairs", "total_pers", "mean_pers", "max_pers",
                               "std_pers", "entropy"]]
                vectors.append(np.zeros(len(stat_names), dtype=np.float32))
                names.extend(stat_names)

        feature_vector = np.concatenate(vectors).astype(np.float32)
        return TDAFeatures(feature_vector=feature_vector, feature_names=names)

    def _resolve_range(self, diagrams: Dict[int, PersistenceDiagram]) -> Tuple[float, float]:
        if self.grid_range != "auto":
            return tuple(self.grid_range)
        all_vals = []
        for diag in diagrams.values():
            if len(diag.pairs) > 0:
                all_vals.append(diag.pairs.ravel())
        if not all_vals:
            return (0.0, 1.0)
        all_vals = np.concatenate(all_vals)
        return (float(np.min(all_vals)), float(np.max(all_vals)))

    @staticmethod
    def _betti_curve(diag: Optional[PersistenceDiagram],
                     grid: np.ndarray) -> np.ndarray:
        """Compute β_k(t) = #{i : b_i ≤ t < d_i} for each t in grid."""
        curve = np.zeros(len(grid), dtype=np.float32)
        if diag is None or len(diag.pairs) == 0:
            return curve
        births = diag.pairs[:, 0]  # (n,)
        deaths = diag.pairs[:, 1]  # (n,)
        # Vectorized: for each grid point t, count pairs where birth <= t < death
        # Shape: (grid,) by broadcasting (grid, 1) vs (1, n)
        alive = (births[None, :] <= grid[:, None]) & (grid[:, None] < deaths[None, :])
        curve = alive.sum(axis=1).astype(np.float32)
        return curve

    @staticmethod
    def _persistence_stats(diag: PersistenceDiagram,
                           dim: int) -> Tuple[np.ndarray, List[str]]:
        lifetimes = diag.pairs[:, 1] - diag.pairs[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        n = len(lifetimes)
        if n == 0:
            names = [f"H{dim}_{s}" for s in
                     ["n_pairs", "total_pers", "mean_pers", "max_pers",
                      "std_pers", "entropy"]]
            return np.zeros(6, dtype=np.float32), names
        total = float(np.sum(lifetimes))
        # Persistence entropy: -sum(p_i * log(p_i)) where p_i = lifetime_i / total
        p = lifetimes / total
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        stats = np.array([
            n, total, np.mean(lifetimes), np.max(lifetimes),
            np.std(lifetimes), entropy
        ], dtype=np.float32)
        names = [f"H{dim}_{s}" for s in
                 ["n_pairs", "total_pers", "mean_pers", "max_pers",
                  "std_pers", "entropy"]]
        return stats, names
