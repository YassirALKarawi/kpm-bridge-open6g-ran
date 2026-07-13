"""Finite-sample uncertainty calibration and admission gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def split_conformal_radius(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """Return the finite-sample split-conformal residual radius."""
    scores = np.asarray(residuals, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("at least one finite residual is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return float("inf")
    rank = max(rank, 1)
    return float(np.partition(scores, rank - 1)[rank - 1])


@dataclass(frozen=True)
class QualityBudget:
    max_radius: float
    max_age_ms: float
    min_support: float


def admit(radius: float, age_ms: float, support: float, drift: bool, budget: QualityBudget) -> bool:
    """Fail closed when any telemetry-quality requirement is violated."""
    finite = np.isfinite([radius, age_ms, support]).all()
    return bool(
        finite
        and not drift
        and radius <= budget.max_radius
        and age_ms <= budget.max_age_ms
        and support >= budget.min_support
    )
