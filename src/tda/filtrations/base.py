"""Abstract filtration interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FilteredComplex:
    """A simplicial complex with filtration values.

    simplices: list of (simplex, filtration_value) where simplex is a tuple of vertex indices.
    num_vertices: total number of vertices.
    """
    simplices: List[Tuple[tuple, float]]
    num_vertices: int


class Filtration(ABC):
    """Interface for building a filtered simplicial complex from a weight array."""

    @abstractmethod
    def build(self, weights: np.ndarray) -> FilteredComplex:
        """Build a filtered simplicial complex from a weight tensor (any shape)."""
        ...
