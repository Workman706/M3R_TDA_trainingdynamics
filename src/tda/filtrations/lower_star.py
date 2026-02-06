"""Lower-star filtration on weight tensors.

Lower-star filtration definition:
  Given a scalar function f on vertices of a simplicial complex, each simplex σ
  enters the filtration at value  filt(σ) = max{ f(v) : v is a vertex of σ }.

For weight tensors:
  - Each entry of the tensor is a vertex.
  - f(i) = |w_i| (if use_abs=True) or w_i.
  - Adjacency is grid-based:
      * 1D tensor: chain graph (vertex i connected to i+1).
      * 2D tensor: 4-neighborhood grid (up, down, left, right).
      * Higher-D: flattened to 2D (first dim vs product of remaining dims).
  - Edges are added between adjacent vertices.
  - Triangles (2-simplices) are optionally added for 2D grids when max_dim >= 2:
      for each grid cell (i,j), add triangles [(i,j),(i+1,j),(i,j+1)] and
      [(i+1,j),(i,j+1),(i+1,j+1)] if all vertices exist.
  - Each simplex's filtration value = max of its vertices' function values.
"""
from __future__ import annotations

import numpy as np
from src.tda.filtrations.base import Filtration, FilteredComplex


class LowerStarFiltration(Filtration):
    """Lower-star filtration on weight tensors with grid adjacency."""

    def __init__(self, use_abs: bool = True, max_dim: int = 1):
        self.use_abs = use_abs
        self.max_dim = max_dim

    def build(self, weights: np.ndarray) -> FilteredComplex:
        w = weights.astype(np.float32).copy()
        # Reshape to at most 2D
        if w.ndim == 0:
            w = w.reshape(1)
        if w.ndim == 1:
            return self._build_1d(w)
        if w.ndim >= 2:
            w = w.reshape(w.shape[0], -1)
            return self._build_2d(w)

    def _vertex_fn(self, w: np.ndarray) -> np.ndarray:
        return np.abs(w) if self.use_abs else w

    def _build_1d(self, w: np.ndarray) -> FilteredComplex:
        """Chain graph: vertices 0..n-1, edges (i, i+1)."""
        f = self._vertex_fn(w)
        n = len(f)
        simplices = []
        # Add vertices (0-simplices)
        for i in range(n):
            simplices.append(((i,), float(f[i])))
        # Add edges (1-simplices)
        if self.max_dim >= 1:
            for i in range(n - 1):
                filt_val = float(max(f[i], f[i + 1]))
                simplices.append(((i, i + 1), filt_val))
        return FilteredComplex(simplices=simplices, num_vertices=n)

    def _build_2d(self, w: np.ndarray) -> FilteredComplex:
        """4-neighborhood grid graph on an (R, C) matrix."""
        f = self._vertex_fn(w)
        R, C = f.shape
        n = R * C

        def idx(r, c):
            return r * C + c

        simplices = []
        # Vertices
        flat_f = f.ravel()
        for i in range(n):
            simplices.append(((i,), float(flat_f[i])))

        # Edges: right and down neighbors (avoids duplicates)
        if self.max_dim >= 1:
            for r in range(R):
                for c in range(C):
                    v = idx(r, c)
                    # Right neighbor
                    if c + 1 < C:
                        u = idx(r, c + 1)
                        simplices.append(((v, u), float(max(flat_f[v], flat_f[u]))))
                    # Down neighbor
                    if r + 1 < R:
                        u = idx(r + 1, c)
                        simplices.append(((v, u), float(max(flat_f[v], flat_f[u]))))

        # Triangles from each grid square's diagonal split
        if self.max_dim >= 2:
            for r in range(R - 1):
                for c in range(C - 1):
                    tl = idx(r, c)
                    tr = idx(r, c + 1)
                    bl = idx(r + 1, c)
                    br = idx(r + 1, c + 1)
                    # Diagonal edge (tl, br) needed for both triangles
                    diag_filt = float(max(flat_f[tl], flat_f[br]))
                    simplices.append(((tl, br), diag_filt))
                    # Upper-left triangle: tl, tr, br
                    tri_filt = float(max(flat_f[tl], flat_f[tr], flat_f[br]))
                    simplices.append(((tl, tr, br), tri_filt))
                    # Lower-right triangle: tl, bl, br
                    tri_filt = float(max(flat_f[tl], flat_f[bl], flat_f[br]))
                    simplices.append(((tl, bl, br), tri_filt))

        return FilteredComplex(simplices=simplices, num_vertices=n)
