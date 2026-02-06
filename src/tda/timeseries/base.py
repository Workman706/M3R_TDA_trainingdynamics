"""Abstract time-series fitting interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class FitResult:
    """Result of fitting a time-series model to feature trajectories.

    smoothed: np.ndarray of shape (n_steps, n_features) — fitted values at each step.
    steps: np.ndarray of shape (n_steps,) — training steps.
    coefficients: dict with serializable representation of the fit (e.g., spline knots).
    """
    smoothed: np.ndarray
    steps: np.ndarray
    coefficients: Dict[str, Any]


class TimeSeriesFitter(ABC):
    """Interface for fitting smooth curves to per-feature time series."""

    @abstractmethod
    def fit(self, steps: np.ndarray, features: np.ndarray) -> FitResult:
        """Fit a smooth model to features over training steps.

        Args:
            steps: (n_steps,) array of step numbers.
            features: (n_steps, n_features) array.
        Returns:
            FitResult with smoothed trajectories and coefficients.
        """
        ...
