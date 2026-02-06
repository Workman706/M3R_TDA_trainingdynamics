"""GUDHI-based persistent homology computation."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import gudhi

from src.tda.filtrations.base import FilteredComplex
from src.tda.ph.base import PHComputer, PersistenceDiagram

logger = logging.getLogger(__name__)


class GUDHIPHComputer(PHComputer):
    """Compute persistence using GUDHI's SimplexTree."""

    def compute(self, fcomplex: FilteredComplex,
                max_dim: int) -> Dict[int, PersistenceDiagram]:
        st = gudhi.SimplexTree()
        for simplex, filt_val in fcomplex.simplices:
            st.insert(list(simplex), filtration=filt_val)

        st.compute_persistence()
        result = {}
        for dim in range(max_dim + 1):
            pairs_raw = st.persistence_intervals_in_dimension(dim)
            if len(pairs_raw) == 0:
                pairs = np.empty((0, 2), dtype=np.float32)
            else:
                pairs = np.array(pairs_raw, dtype=np.float32)
                # Remove infinite-death pairs for feature extraction (keep finite only)
                finite_mask = np.isfinite(pairs[:, 1])
                pairs = pairs[finite_mask]
            result[dim] = PersistenceDiagram(pairs=pairs, dimension=dim)
            logger.debug("H%d: %d finite persistence pairs", dim, len(pairs))
        return result
