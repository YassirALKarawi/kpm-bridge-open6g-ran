"""Trace-level loading and downstream-task construction for ColO-RAN."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import PREDICTION_HORIZON, fitting_prefix, xapp_fit_mask

FEATURE_NAMES = (
    "dl_brate",
    "dl_bler",
    "dl_mcs",
    "dl_snr",
    "rsrp",
    "ul_brate",
    "ul_bler",
    "ul_buff",
)

_EMBB = {3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48}
_MTC = {4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49}
_URLLC = {2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47}


@dataclass(frozen=True)
class CanonicalTrace:
    trace_id: str
    scheduler: str
    training_config: str
    experiment: str
    base_station: str
    user_equipment: str
    traffic_class: str
    time_ms: np.ndarray
    values: np.ndarray
    risk: np.ndarray | None = None

    @property
    def dt_ms(self) -> float:
        return float(np.median(np.diff(self.time_ms)))


@dataclass(frozen=True)
class FeatureStats:
    location: np.ndarray
    scale: np.ndarray


def traffic_class(user_equipment: str) -> str:
    number = int(re.search(r"\d+", user_equipment).group())
    if number in _EMBB:
        return "eMBB"
    if number in _MTC:
        return "MTC"
    if number in _URLLC:
        return "URLLC"
    raise ValueError(f"UE {user_equipment} has no documented traffic class")


def _metadata(relative_path: str) -> tuple[str, str, str, str, str]:
    parts = Path(relative_path).parts
    if len(parts) != 6:
        raise ValueError(f"unexpected dataset path: {relative_path}")
    _, scheduler, training, experiment, base_station, filename = parts
    return scheduler, training, experiment, base_station, Path(filename).stem


def load_colosseum_subset(
    root: Path = Path("data/raw/colosseum"),
    manifest_path: Path = Path("data/colosseum_subset_manifest.json"),
    min_attached_samples: int = 400,
) -> list[CanonicalTrace]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    traces: list[CanonicalTrace] = []
    for record in manifest["files"]:
        relative_path = str(record["path"])
        scheduler, training, experiment, bs, ue = _metadata(relative_path)
        frame = pd.read_csv(root / relative_path, usecols=["time", "is_attached", *FEATURE_NAMES])
        frame = frame.loc[frame["is_attached"] > 0.5].copy()
        frame = frame.drop_duplicates(subset="time", keep="last").sort_values("time")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURE_NAMES))
        if len(frame) < min_attached_samples:
            continue
        values = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
        times = frame["time"].to_numpy(dtype=float)
        trace_id = "/".join((scheduler, training, experiment, bs, ue))
        traces.append(
            CanonicalTrace(
                trace_id=trace_id,
                scheduler=scheduler,
                training_config=training,
                experiment=experiment,
                base_station=bs,
                user_equipment=ue,
                traffic_class=traffic_class(ue),
                time_ms=times,
                values=values,
            )
        )
    return sorted(traces, key=lambda trace: trace.trace_id)


def robust_feature_stats(traces: list[CanonicalTrace]) -> FeatureStats:
    if not traces:
        raise ValueError("traces cannot be empty")
    # Scaling is part of the fitted mapper and conformal score definition.  It
    # must therefore be frozen from the fitting prefix, never from calibration.
    values = np.vstack([fitting_prefix(trace.values) for trace in traces])
    location = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = np.std(values, axis=0, ddof=1)
    # Use a conservative training-only scale for joint error balls.  The
    # maximum prevents sparse zero-inflated KPMs from receiving a near-zero
    # normaliser while retaining IQR robustness for compact features.
    scale = np.maximum(robust_scale, standard_scale)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return FeatureStats(location=location, scale=scale)


def _future_mean(values: np.ndarray, horizon: int) -> np.ndarray:
    n, d = values.shape
    result = np.empty((n - horizon, d), dtype=float)
    csum = np.vstack([np.zeros((1, d)), np.cumsum(values, axis=0)])
    for index in range(n - horizon):
        result[index] = (csum[index + horizon + 1] - csum[index + 1]) / horizon
    return result


def attach_qos_risk_labels(
    traces: list[CanonicalTrace],
    horizon: int = PREDICTION_HORIZON,
) -> tuple[list[CanonicalTrace], dict[str, dict[str, float]]]:
    """Create a trace-backed one-second QoS-risk task without future leakage."""
    training = [trace for trace in traces if trace.experiment == "exp1"]
    future_by_class: dict[str, list[np.ndarray]] = {key: [] for key in ("eMBB", "MTC", "URLLC")}
    for trace in training:
        future = _future_mean(trace.values, horizon)
        # Both the label thresholds and the downstream xApp are frozen before
        # the calibration suffix.  Guarding by ``horizon`` also prevents a
        # training label from peeking across the split boundary.
        future_by_class[trace.traffic_class].append(future[xapp_fit_mask(len(future), horizon)])

    thresholds: dict[str, dict[str, float]] = {}
    for key, blocks in future_by_class.items():
        future = np.vstack(blocks)
        thresholds[key] = {
            "dl_brate_q35": float(np.quantile(future[:, 0], 0.35)),
            "dl_bler_q75": float(np.quantile(future[:, 1], 0.75)),
        }

    labelled: list[CanonicalTrace] = []
    for trace in traces:
        future = _future_mean(trace.values, horizon)
        threshold = thresholds[trace.traffic_class]
        risk = (
            (future[:, 0] <= threshold["dl_brate_q35"])
            | (future[:, 1] >= threshold["dl_bler_q75"])
        ).astype(int)
        labelled.append(
            replace(
                trace,
                time_ms=trace.time_ms[:-horizon],
                values=trace.values[:-horizon],
                risk=risk,
            )
        )
    return labelled, thresholds
