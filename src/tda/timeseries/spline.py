"""Spline fitting for TDA feature trajectories over training time."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

from src.tda.timeseries.base import TimeSeriesFitter, FitResult

logger = logging.getLogger(__name__)


class SplineFitter(TimeSeriesFitter):
    """Fit a smoothing spline to each feature dimension independently.

    Args:
        smoothing_factor: passed to UnivariateSpline `s` parameter.
            None = let scipy choose via cross-validation.
        degree: spline degree (default 3 = cubic).
    """

    def __init__(self, smoothing_factor: Optional[float] = None, degree: int = 3):
        self.smoothing_factor = smoothing_factor
        self.degree = degree

    def fit(self, steps: np.ndarray, features: np.ndarray) -> FitResult:
        n_steps, n_features = features.shape
        smoothed = np.zeros_like(features)
        all_knots = {}
        all_coeffs = {}

        # Need at least degree+1 points for spline fitting
        if n_steps <= self.degree:
            logger.warning(
                "Only %d snapshots, need >%d for degree-%d spline. "
                "Returning raw features as 'smoothed'.",
                n_steps, self.degree, self.degree
            )
            return FitResult(
                smoothed=features.copy(),
                steps=steps.copy(),
                coefficients={"warning": "too_few_points"},
            )

        for j in range(n_features):
            y = features[:, j]
            # Skip constant features
            if np.std(y) < 1e-12:
                smoothed[:, j] = y
                all_knots[str(j)] = []
                all_coeffs[str(j)] = [float(y[0])]
                continue
            try:
                spl = UnivariateSpline(
                    steps.astype(float), y.astype(float),
                    s=self.smoothing_factor, k=self.degree
                )
                smoothed[:, j] = spl(steps)
                knots = spl.get_knots().tolist()
                coeffs = spl.get_coeffs().tolist()
                all_knots[str(j)] = knots
                all_coeffs[str(j)] = coeffs
            except Exception as e:
                logger.warning("Spline fit failed for feature %d: %s", j, e)
                smoothed[:, j] = y

        return FitResult(
            smoothed=smoothed,
            steps=steps.copy(),
            coefficients={"knots": all_knots, "coeffs": all_coeffs,
                          "degree": self.degree},
        )
