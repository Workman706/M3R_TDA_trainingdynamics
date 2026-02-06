"""Abstract feature extraction interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.tda.ph.base import PersistenceDiagram


@dataclass
class TDAFeatures:
    """Extracted features from persistence diagrams.

    feature_vector: 1D numpy array (concatenation of all features).
    feature_names: list of strings naming each entry.
    """
    feature_vector: np.ndarray
    feature_names: List[str]


class FeatureExtractor(ABC):
    """Interface for extracting features from persistence diagrams."""

    @abstractmethod
    def extract(self, diagrams: Dict[int, PersistenceDiagram]) -> TDAFeatures:
        ...
