"""Deterministic controlled KPM-shift generator for early validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImplementationProfile:
    scale: np.ndarray
    bias: np.ndarray
    noise_std: np.ndarray
    window: int = 1
    lag: int = 0
    missing_rate: float = 0.0
    quantisation: float = 0.0


def latent_ran_process(steps: int, seed: int) -> np.ndarray:
    """Generate six correlated canonical KPMs without paper-level claims."""
    if steps < 10:
        raise ValueError("steps must be at least 10")
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=float)
    load = np.empty(steps)
    load[0] = 0.50
    forcing = 0.16 * np.sin(2 * np.pi * t / 240.0) + 0.08 * np.sin(2 * np.pi * t / 53.0)
    for i in range(1, steps):
        load[i] = 0.94 * load[i - 1] + 0.06 * (0.52 + forcing[i]) + rng.normal(0, 0.018)
    load = np.clip(load, 0.05, 0.98)
    active_ues = np.clip(np.rint(4 + 56 * load + rng.normal(0, 1.5, steps)), 1, 64)
    prb = np.clip(load + rng.normal(0, 0.025, steps), 0, 1)
    bler = np.clip(0.012 + 0.18 * np.maximum(load - 0.62, 0) ** 1.4 + rng.normal(0, 0.004, steps), 0, 1)
    rsrp = -78.0 - 13.0 * load + rng.normal(0, 1.1, steps)
    throughput = np.clip(105.0 * load * (1.0 - 0.55 * bler) + rng.normal(0, 1.2, steps), 0, None)
    delay = np.clip(3.0 + 36.0 * load**2 + 85.0 * bler + rng.normal(0, 1.0, steps), 0, None)
    return np.column_stack([throughput, prb, bler, rsrp, active_ues, delay])


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.empty_like(values)
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    for i in range(values.shape[0]):
        start = max(0, i + 1 - window)
        result[i] = (csum[i + 1] - csum[start]) / (i + 1 - start)
    return result


def observe_implementation(
    canonical: np.ndarray, profile: ImplementationProfile, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply documented scale, window, lag, noise, quantisation, and missingness."""
    x = np.asarray(canonical, dtype=float)
    d = x.shape[1]
    for field in (profile.scale, profile.bias, profile.noise_std):
        if np.asarray(field).shape != (d,):
            raise ValueError("profile vectors must match the KPM dimension")
    rng = np.random.default_rng(seed)
    aligned = _rolling_mean(x, profile.window)
    if profile.lag > 0:
        aligned = np.vstack([np.repeat(aligned[[0]], profile.lag, axis=0), aligned[:-profile.lag]])
    raw = aligned * profile.scale + profile.bias
    raw += rng.normal(0, profile.noise_std, raw.shape)
    if profile.quantisation > 0:
        raw = np.round(raw / profile.quantisation) * profile.quantisation
    mask = rng.random(raw.shape) >= profile.missing_rate
    raw = np.where(mask, raw, np.nan)
    return raw, mask


def fit_affine_bridge(raw: np.ndarray, canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a per-KPM affine anchor map as a transparent initial baseline."""
    y = np.asarray(raw, dtype=float)
    z = np.asarray(canonical, dtype=float)
    if y.shape != z.shape:
        raise ValueError("raw and canonical arrays must have equal shape")
    slopes = np.empty(y.shape[1])
    offsets = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        valid = np.isfinite(y[:, j]) & np.isfinite(z[:, j])
        if valid.sum() < 3:
            raise ValueError("insufficient anchors for affine fitting")
        design = np.column_stack([y[valid, j], np.ones(valid.sum())])
        slopes[j], offsets[j] = np.linalg.lstsq(design, z[valid, j], rcond=None)[0]
    return slopes, offsets


def apply_affine_bridge(raw: np.ndarray, slopes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(raw, dtype=float) * slopes + offsets
