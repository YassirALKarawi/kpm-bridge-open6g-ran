"""Single-source benchmark split definitions.

The mapper, feature scaling, downstream xApp, and task-label thresholds must
all be fixed before the calibration suffix begins.  Keeping these boundaries
in one module prevents an apparently harmless preprocessing step from reading
the held-out calibration region.
"""

from __future__ import annotations

import numpy as np


FIT_FRACTION = 0.60
CALIBRATION_STRIDE = 4
PREDICTION_HORIZON = 4


def fit_stop(length: int, fit_fraction: float = FIT_FRACTION) -> int:
    """Return the exclusive end of the mapper-fitting prefix."""
    if length < 1:
        raise ValueError("length must be positive")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    return max(1, int(np.floor(fit_fraction * length)))


def mapper_fit_mask(length: int, anchor_fraction: float) -> np.ndarray:
    """Select evenly spaced paired anchors inside the fitting prefix only."""
    if not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("anchor_fraction must lie in (0, 1]")
    stop = fit_stop(length)
    desired = max(1, int(np.floor(anchor_fraction * stop)))
    indices = np.linspace(0, stop - 1, desired, dtype=int)
    mask = np.zeros(length, dtype=bool)
    mask[np.unique(indices)] = True
    return mask


def calibration_mask(length: int, stride: int = CALIBRATION_STRIDE) -> np.ndarray:
    """Select calibration observations strictly after the fitting prefix."""
    if stride < 1:
        raise ValueError("stride must be positive")
    mask = np.zeros(length, dtype=bool)
    mask[fit_stop(length) :: stride] = True
    return mask


def xapp_fit_mask(length: int, horizon: int = PREDICTION_HORIZON) -> np.ndarray:
    """Select xApp-training rows whose complete label horizon precedes calibration."""
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    stop = fit_stop(length) - horizon
    if stop < 1:
        raise ValueError("trace is too short for the requested prediction horizon")
    mask = np.zeros(length, dtype=bool)
    mask[:stop] = True
    return mask


def fitting_prefix(values: np.ndarray) -> np.ndarray:
    """Return the portion permitted for training-only preprocessing."""
    array = np.asarray(values)
    return array[: fit_stop(len(array))]
