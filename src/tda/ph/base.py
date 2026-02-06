"""Abstract persistent homology computation interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.tda.filtrations.base import FilteredComplex


@dataclass
class PersistenceDiagram:
    """Persistence diagram for a single homology dimension.

    pairs: np.ndarray of shape (n, 2), each row is (birth, death).
    dimension: homology dimension.
    """
    pairs: np.ndarray  # (n, 2)
    dimension: int


class PHComputer(ABC):
    """Interface for computing persistence diagrams from a filtered complex."""

    @abstractmethod
    def compute(self, fcomplex: FilteredComplex,
                max_dim: int) -> Dict[int, PersistenceDiagram]:
        """Compute persistence diagrams for dimensions 0..max_dim.

        Returns: {dim: PersistenceDiagram}.
        """
        ...
