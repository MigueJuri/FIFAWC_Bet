"""Shared bookmaker de-vigging utilities.

This module centralizes margin-removal logic used by standalone scripts in this
repository. All functions expect decimal odds and return normalized
probabilities.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import root_scalar


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array of non-negative values so it sums to 1.0."""
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalize a non-positive vector.")
    return values / total


def power_devig(odds: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Remove bookmaker margin with the power method.

    Solves for exponent ``k`` such that ``sum((1/odds)^k) == 1``.
    """
    odds_array = np.asarray(odds, dtype=float)
    if odds_array.ndim != 1 or odds_array.size < 2:
        raise ValueError("Power devigging requires at least two outcomes.")

    implied = 1.0 / odds_array

    def objective(k: float) -> float:
        return float(np.sum(implied**k) - 1.0)

    solution = root_scalar(objective, bracket=[1.0, 10.0], method="brentq")
    k_value = float(solution.root)
    return normalize(implied**k_value), k_value


def shin_devig(odds: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Remove bookmaker margin using Shin's method.

    Falls back to proportional normalization when root solving is ill-posed.
    """
    odds_array = np.asarray(odds, dtype=float)
    if odds_array.ndim != 1 or odds_array.size < 2:
        raise ValueError("Shin devigging requires at least two outcomes.")

    implied = 1.0 / odds_array
    implied_sum = float(np.sum(implied))

    def shin_probs(z: float) -> np.ndarray:
        return (
            np.sqrt(z * z + 4.0 * (1.0 - z) * (implied * implied / implied_sum)) - z
        ) / (2.0 * (1.0 - z))

    def objective(z: float) -> float:
        return float(np.sum(shin_probs(z)) - 1.0)

    lower = 0.0
    upper = 1.0 - 1e-8
    try:
        lower_value = objective(lower)
        upper_value = objective(upper)
        if lower_value * upper_value > 0:
            return normalize(implied), 0.0

        solution = root_scalar(objective, bracket=[lower, upper], method="brentq")
        z_value = float(solution.root)
        return normalize(shin_probs(z_value)), z_value
    except Exception:
        return normalize(implied), 0.0
