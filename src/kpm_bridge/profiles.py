"""Controlled cross-implementation observation profiles and contract inversion."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contracts import CompiledTransformPlan, KPMContract, compile_transform_plan
from .dataset import CanonicalTrace, FeatureStats

RATE_COLUMNS = (0, 5)


@dataclass(frozen=True)
class ImplementationProfile:
    key: str
    label: str
    declared_scale: np.ndarray
    declared_offset: np.ndarray
    hidden_gain: np.ndarray
    hidden_curve: np.ndarray
    window: int
    lag: int
    noise_fraction: float
    missing_rate: float
    burst_rate: float
    quantisation_fraction: float
    timestamp_jitter_ms: float
    rates_as_counters: bool = False
    reset_period: int = 0
    drift_fraction: float | None = None
    drift_gain: float = 0.0
    drift_offset: float = 0.0
    drift_noise_multiplier: float = 1.0
    drift_lag_increment: int = 0


@dataclass(frozen=True)
class Observation:
    raw: np.ndarray
    mask: np.ndarray
    reset_mask: np.ndarray
    age_ms: np.ndarray
    drift_start: int | None

    @property
    def support(self) -> np.ndarray:
        return np.mean(self.mask, axis=1)


def default_profiles(dimension: int = 8) -> dict[str, ImplementationProfile]:
    if dimension != 8:
        raise ValueError("the released stress profiles are defined for eight KPMs")
    ones = np.ones(dimension)
    zeros = np.zeros(dimension)
    unit_scale = np.array([1e-3, 1e-2, 1.0, 1.0, 1.0, 1e-3, 1e-2, 1.0])
    return {
        "P0": ImplementationProfile(
            "P0", "reference", ones, zeros, zeros, zeros, 1, 0, 0.002, 0.0, 0.0, 0.0, 2.0
        ),
        "P1": ImplementationProfile(
            "P1",
            "unit/window shift",
            unit_scale,
            zeros,
            np.array([0.05, -0.04, 0.03, 0.04, -0.03, 0.06, -0.05, 0.04]),
            np.array([0.02, 0.02, -0.01, 0.01, 0.00, 0.03, 0.02, -0.02]),
            2,
            1,
            0.020,
            0.03,
            0.002,
            0.010,
            15.0,
        ),
        "P2": ImplementationProfile(
            "P2",
            "counter/reset shift",
            unit_scale,
            zeros,
            np.array([0.08, -0.06, 0.05, 0.07, -0.04, 0.09, -0.07, 0.05]),
            np.array([0.04, 0.03, -0.02, 0.02, 0.01, 0.05, 0.03, -0.03]),
            3,
            1,
            0.030,
            0.08,
            0.004,
            0.015,
            25.0,
            rates_as_counters=True,
            reset_period=257,
        ),
        "P3": ImplementationProfile(
            "P3",
            "filtered/sparse shift",
            unit_scale,
            zeros,
            np.array([0.14, -0.10, 0.09, 0.11, -0.08, 0.16, -0.12, 0.10]),
            np.array([0.08, 0.06, -0.04, 0.04, 0.03, 0.09, 0.06, -0.05]),
            4,
            2,
            0.050,
            0.18,
            0.010,
            0.030,
            40.0,
        ),
        "P4": ImplementationProfile(
            "P4",
            "post-deployment drift",
            unit_scale,
            zeros,
            np.array([0.05, -0.04, 0.04, 0.04, -0.03, 0.06, -0.05, 0.04]),
            np.array([0.02, 0.02, -0.01, 0.01, 0.00, 0.03, 0.02, -0.02]),
            2,
            1,
            0.025,
            0.05,
            0.003,
            0.012,
            20.0,
            drift_fraction=0.55,
            drift_gain=0.35,
            drift_offset=1.25,
            drift_noise_multiplier=2.0,
            drift_lag_increment=2,
        ),
    }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (csum[index + 1] - csum[start]) / (index + 1 - start)
    return result


def _lag(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return values.copy()
    return np.vstack([np.repeat(values[[0]], lag, axis=0), values[:-lag]])


def _semantic_filter(
    values: np.ndarray,
    stats: FeatureStats,
    gain: np.ndarray,
    curve: np.ndarray,
    window: int,
    lag: int,
) -> np.ndarray:
    standard = (values - stats.location) / stats.scale
    hidden = values + stats.scale * (
        gain * np.tanh(standard)
        + curve * np.sign(standard) * np.tanh(standard) ** 2
    )
    return _lag(_rolling_mean(hidden, window), lag)


def _to_counters(
    values: np.ndarray, dt_s: float, reset_period: int
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    reset_mask = np.zeros(values.shape, dtype=bool)
    for column in RATE_COLUMNS:
        increments = np.maximum(values[:, column], 0.0) * dt_s
        counter = np.empty(len(values), dtype=float)
        running = 0.0
        for index, increment in enumerate(increments):
            if index == 0 or (reset_period > 0 and index % reset_period == 0):
                running = 0.0
                reset_mask[index, column] = True
            running += increment
            counter[index] = running
        result[:, column] = counter
    return result, reset_mask


_QUANTITIES = (
    "downlink_throughput",
    "downlink_bler",
    "downlink_mcs",
    "downlink_snr",
    "rsrp",
    "uplink_throughput",
    "uplink_bler",
    "uplink_buffer",
)
_TARGET_UNITS = ("bit/s", "%", "count", "dB", "dBm", "bit/s", "%", "count")


def canonical_contracts() -> list[KPMContract]:
    """Return the registered eight-feature canonical xApp contract."""
    return [
        KPMContract(
            name=f"canonical.{quantity}",
            quantity=quantity,
            unit=unit,
            entity_scope="ue",
            aggregation="mean" if index in RATE_COLUMNS else "gauge",
            window_ms=250,
            clock="canonical-event-time",
            counter_semantics="gauge",
            schema_version="canonical-v1",
            provenance="registered-portable-xapp",
        )
        for index, (quantity, unit) in enumerate(zip(_QUANTITIES, _TARGET_UNITS, strict=True))
    ]


def profile_contracts(profile: ImplementationProfile) -> list[KPMContract]:
    """Materialize the declared source semantics of a controlled profile."""
    units = list(_TARGET_UNITS)
    for index in RATE_COLUMNS:
        units[index] = "kbit" if profile.rates_as_counters else (
            "kbit/s" if np.isclose(profile.declared_scale[index], 1e-3) else "bit/s"
        )
    for index in (1, 6):
        units[index] = "ratio" if np.isclose(profile.declared_scale[index], 1e-2) else "%"
    return [
        KPMContract(
            name=f"{profile.key}.{quantity}",
            quantity=quantity,
            unit=units[index],
            entity_scope="ue",
            aggregation="mean" if profile.window > 1 else (
                "mean" if index in RATE_COLUMNS else "gauge"
            ),
            window_ms=250 * profile.window,
            clock="e2-node",
            counter_semantics=(
                "cumulative" if profile.rates_as_counters and index in RATE_COLUMNS else "gauge"
            ),
            schema_version="e2sm-kpm-controlled-v1",
            provenance=f"controlled-profile-{profile.key}",
        )
        for index, quantity in enumerate(_QUANTITIES)
    ]


def compile_profile_plan(profile: ImplementationProfile) -> CompiledTransformPlan:
    """Compile the profile declaration into an executable canonical plan."""
    return compile_transform_plan(profile_contracts(profile), canonical_contracts())


def observe_trace(
    trace: CanonicalTrace,
    profile: ImplementationProfile,
    stats: FeatureStats,
    seed: int,
    inject_drift: bool,
) -> Observation:
    rng = np.random.default_rng(seed)
    n, d = trace.values.shape
    base = _semantic_filter(
        trace.values,
        stats,
        profile.hidden_gain,
        profile.hidden_curve,
        profile.window,
        profile.lag,
    )
    drift_start: int | None = None
    reset_mask = np.zeros((n, d), dtype=bool)
    active_lag = np.full(n, profile.lag, dtype=float)
    active_noise = np.full(n, profile.noise_fraction, dtype=float)
    if inject_drift and profile.drift_fraction is not None:
        drift_start = int(np.floor(profile.drift_fraction * n))
        direction = np.array([1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        drifted = _semantic_filter(
            trace.values,
            stats,
            profile.hidden_gain + direction * profile.drift_gain,
            profile.hidden_curve + direction * profile.drift_gain * 0.30,
            profile.window,
            profile.lag + profile.drift_lag_increment,
        )
        drifted += direction * profile.drift_offset * stats.scale
        base[drift_start:] = drifted[drift_start:]
        active_lag[drift_start:] += profile.drift_lag_increment
        active_noise[drift_start:] *= profile.drift_noise_multiplier

    if profile.rates_as_counters:
        base, reset_mask = _to_counters(
            base, trace.dt_ms / 1000.0, profile.reset_period
        )

    raw = base * profile.declared_scale + profile.declared_offset
    noise = rng.normal(size=raw.shape) * (
        active_noise[:, None] * stats.scale[None, :] * np.abs(profile.declared_scale)[None, :]
    )
    raw += noise
    step = profile.quantisation_fraction * stats.scale * np.abs(profile.declared_scale)
    nonzero = step > 0
    raw[:, nonzero] = np.round(raw[:, nonzero] / step[nonzero]) * step[nonzero]

    mask = rng.random((n, d)) >= profile.missing_rate
    burst_count = int(np.ceil(profile.burst_rate * n))
    for _ in range(burst_count):
        start = int(rng.integers(0, max(n - 8, 1)))
        width = int(rng.integers(2, 9))
        column = int(rng.integers(0, d))
        mask[start : start + width, column] = False
    raw = np.where(mask, raw, np.nan)

    base_age = (active_lag + (profile.window - 1) / 2.0) * trace.dt_ms
    age = np.maximum(0.0, base_age + np.abs(rng.normal(0, profile.timestamp_jitter_ms, n)))
    return Observation(
        raw=raw,
        mask=mask,
        reset_mask=reset_mask,
        age_ms=age,
        drift_start=drift_start,
    )


def invert_declared_contract(
    observation: Observation,
    profile: ImplementationProfile,
    dt_ms: float,
) -> np.ndarray:
    """Compatibility wrapper; benchmark paths use the compiled plan directly."""
    return compile_profile_plan(profile).apply(
        observation.raw,
        reset_mask=observation.reset_mask,
        dt_ms=dt_ms,
    )


def with_stress(
    profile: ImplementationProfile,
    *,
    missing_rate: float | None = None,
    lag: int | None = None,
    drift_gain: float | None = None,
    drift_offset: float | None = None,
) -> ImplementationProfile:
    return replace(
        profile,
        missing_rate=profile.missing_rate if missing_rate is None else missing_rate,
        lag=profile.lag if lag is None else lag,
        drift_gain=profile.drift_gain if drift_gain is None else drift_gain,
        drift_offset=profile.drift_offset if drift_offset is None else drift_offset,
    )
