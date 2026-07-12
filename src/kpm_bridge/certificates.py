"""Compact certificate encoding, joint calibration scores, and drift invalidation."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .calibration import split_conformal_radius

CERTIFICATE_STRUCT = struct.Struct("!16sQQfffHH")


@dataclass(frozen=True)
class TelemetryCertificate:
    schema_hash: bytes
    mapping_id: int
    calibration_epoch: int
    support: float
    age_ms: float
    radius: float
    flags: int
    kpm_count: int

    def encode(self) -> bytes:
        if len(self.schema_hash) != 16:
            raise ValueError("schema_hash must be exactly 16 bytes")
        return CERTIFICATE_STRUCT.pack(
            self.schema_hash,
            self.mapping_id,
            self.calibration_epoch,
            self.support,
            self.age_ms,
            self.radius,
            self.flags,
            self.kpm_count,
        )


@dataclass(frozen=True)
class AnchorDriftCalibration:
    mean: np.ndarray
    inverse_covariance: np.ndarray
    window: int
    threshold: float
    anchor_stride: int


def joint_standardised_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    residual = (np.asarray(predictions) - np.asarray(targets)) / np.asarray(feature_scale)
    return np.linalg.norm(residual, ord=2, axis=1)


def calibrated_joint_radius(scores: np.ndarray, alpha: float = 0.05) -> float:
    return split_conformal_radius(scores, alpha=alpha)


def detect_anchor_drift(
    scores: np.ndarray,
    threshold: float,
    anchor_stride: int = 10,
    consecutive: int = 2,
) -> tuple[np.ndarray, int | None]:
    """Fail closed after consecutive excessive residuals at trusted anchor epochs."""
    if anchor_stride < 1 or consecutive < 1:
        raise ValueError("anchor_stride and consecutive must be positive")
    drift = np.zeros(len(scores), dtype=bool)
    streak = 0
    detected: int | None = None
    for index in range(0, len(scores), anchor_stride):
        if np.isfinite(scores[index]) and scores[index] > threshold:
            streak += 1
        else:
            streak = 0
        if streak >= consecutive:
            detected = index
            drift[index:] = True
            break
    return drift, detected


def _mean_shift_scores(
    anchor_residuals: np.ndarray,
    mean: np.ndarray,
    inverse_covariance: np.ndarray,
    window: int,
) -> np.ndarray:
    if len(anchor_residuals) < window:
        return np.empty(0, dtype=float)
    scores = np.empty(len(anchor_residuals) - window + 1, dtype=float)
    for end in range(window - 1, len(anchor_residuals)):
        difference = np.mean(anchor_residuals[end - window + 1 : end + 1], axis=0) - mean
        scores[end - window + 1] = np.sqrt(
            max(0.0, window * float(difference @ inverse_covariance @ difference))
        )
    return scores


def calibrate_anchor_mean_shift(
    residual_blocks: list[np.ndarray],
    *,
    window: int = 5,
    quantile: float = 0.975,
    anchor_stride: int = 10,
    covariance_regularisation: float = 0.10,
) -> AnchorDriftCalibration:
    if not residual_blocks:
        raise ValueError("residual_blocks cannot be empty")
    anchors = [block[::anchor_stride] for block in residual_blocks]
    pooled = np.vstack(anchors)
    mean = np.mean(pooled, axis=0)
    covariance = np.cov(pooled, rowvar=False)
    covariance += covariance_regularisation * np.eye(covariance.shape[0])
    inverse = np.linalg.inv(covariance)
    # Calibrate on each stream's maximum, rather than on individual windows,
    # so the quantile controls the trace-level repeated-testing false alarm.
    calibration_maxima = np.array(
        [
            np.max(_mean_shift_scores(block, mean, inverse, window))
            for block in anchors
            if len(block) >= window
        ]
    )
    threshold = float(np.quantile(calibration_maxima, quantile))
    return AnchorDriftCalibration(mean, inverse, window, threshold, anchor_stride)


def detect_anchor_mean_shift(
    residuals: np.ndarray,
    calibration: AnchorDriftCalibration,
) -> tuple[np.ndarray, int | None]:
    anchors = residuals[:: calibration.anchor_stride]
    scores = _mean_shift_scores(
        anchors,
        calibration.mean,
        calibration.inverse_covariance,
        calibration.window,
    )
    drift = np.zeros(len(residuals), dtype=bool)
    exceed = np.flatnonzero(scores > calibration.threshold)
    if exceed.size == 0:
        return drift, None
    anchor_index = int(exceed[0] + calibration.window - 1)
    detected = min(anchor_index * calibration.anchor_stride, len(residuals) - 1)
    drift[detected:] = True
    return drift, detected


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
