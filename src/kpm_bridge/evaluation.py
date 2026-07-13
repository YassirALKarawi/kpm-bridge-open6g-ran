"""End-to-end, deterministic evaluation used by the manuscript."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .certificates import (
    AnchorDriftCalibration,
    CertificateFlag,
    DecisionReason,
    build_certificate,
    calibrate_anchor_mean_shift,
    calibrated_joint_radius,
    certificate_flags,
    decision_reason,
    detect_anchor_mean_shift,
    joint_standardised_scores,
)
from .contracts import (
    CompiledTransformPlan,
    ContractError,
    KPMContract,
    compile_transform_plan,
)
from .dataset import (
    CanonicalTrace,
    FeatureStats,
    attach_qos_risk_labels,
    load_colosseum_subset,
    robust_feature_stats,
)
from .mappers import (
    AnchorRidgeMapper,
    BaseMapper,
    ContractMapper,
    DistributionMapper,
    KPMBridgeMapper,
    ProfiledTrace,
    RawIdentityMapper,
    TemporalRidgeMapper,
    calibration_mask,
    fit_mask,
)
from .profiles import (
    ImplementationProfile,
    canonical_contracts,
    compile_profile_plan,
    default_profiles,
    observe_trace,
    profile_contracts,
    with_stress,
)
from .splits import PREDICTION_HORIZON, fit_stop, xapp_fit_mask
from .xapp import DeploymentSpecificRiskXApp, PortableRiskXApp, decision_regret

SEED = 20260712
DEFAULT_ALPHA = 0.05
DEFAULT_ANCHOR_FRACTION = 0.10
MAX_AGE_MS = 1500.0
MIN_SUPPORT = 0.75


@dataclass
class ProfileContext:
    profile: ImplementationProfile
    plan: CompiledTransformPlan
    training_pairs: list[ProfiledTrace]
    test_pairs: list[ProfiledTrace]
    bridge: KPMBridgeMapper
    calibration_scores: np.ndarray
    calibration_action_scores: np.ndarray
    radius: float
    drift_calibration: AnchorDriftCalibration
    test_predictions: list[np.ndarray]
    test_scores: list[np.ndarray]
    drift_flags: list[np.ndarray]
    detection_indices: list[int | None]


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return SEED ^ int.from_bytes(digest[:4], "big")


def _stack(traces: list[CanonicalTrace], attribute: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(trace, attribute)) for trace in traces], axis=0)


def build_pairs(
    traces: list[CanonicalTrace],
    profile: ImplementationProfile,
    stats: FeatureStats,
    *,
    inject_test_drift: bool = True,
) -> list[ProfiledTrace]:
    pairs: list[ProfiledTrace] = []
    for trace in traces:
        observation = observe_trace(
            trace,
            profile,
            stats,
            stable_seed(profile.key, trace.trace_id),
            inject_drift=inject_test_drift and trace.experiment == "exp2",
        )
        pairs.append(ProfiledTrace(trace, observation))
    return pairs


def fit_portable_xapp(training: list[CanonicalTrace], test: list[CanonicalTrace]) -> tuple[PortableRiskXApp, dict[str, float]]:
    xapp = PortableRiskXApp(SEED)
    fit_masks = [xapp_fit_mask(len(trace.values), PREDICTION_HORIZON) for trace in training]
    train_values = np.vstack(
        [trace.values[mask] for trace, mask in zip(training, fit_masks, strict=True)]
    )
    train_risk = np.concatenate(
        [trace.risk[mask] for trace, mask in zip(training, fit_masks, strict=True)]
    )
    test_values = _stack(test, "values")
    test_risk = _stack(test, "risk")
    xapp.fit(train_values, train_risk)
    probabilities = xapp.probabilities(test_values)
    classification_actions = (probabilities >= 0.5).astype(int)
    actions = xapp.actions(test_values)
    summary = {
        "training_samples": int(len(train_values)),
        "test_samples": int(len(test_values)),
        "test_risk_prevalence": float(np.mean(test_risk)),
        "canonical_auroc": float(roc_auc_score(test_risk, probabilities)),
        "canonical_brier": float(brier_score_loss(test_risk, probabilities)),
        "canonical_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(test_risk, classification_actions)
        ),
        "canonical_f1_at_0_5": float(f1_score(test_risk, classification_actions)),
        "control_action_threshold": xapp.action_threshold,
        "control_action_rate": float(np.mean(actions)),
        "canonical_regret": float(np.mean(decision_regret(actions, test_risk))),
    }
    return xapp, summary


def _method_metrics(
    predictions: list[np.ndarray],
    pairs: list[ProfiledTrace],
    xapp: PortableRiskXApp,
    stats: FeatureStats,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    pooled_prediction = np.vstack(predictions)
    truth = np.vstack([pair.trace.values for pair in pairs])
    risk = np.concatenate([pair.trace.risk for pair in pairs])
    oracle_action = xapp.actions(truth)
    action = xapp.actions(pooled_prediction)
    standard_error = (pooled_prediction - truth) / stats.scale
    trace_rows: list[dict[str, float]] = []
    offset = 0
    for pair, prediction in zip(pairs, predictions, strict=True):
        n = len(prediction)
        local_truth = pair.trace.values
        local_oracle = oracle_action[offset : offset + n]
        local_action = action[offset : offset + n]
        local_risk = risk[offset : offset + n]
        local_error = (prediction - local_truth) / stats.scale
        trace_rows.append(
            {
                "trace_id": pair.trace.trace_id,
                "sse": float(np.sum(local_error**2)),
                "error_count": int(local_error.size),
                "agreement_count": int(np.sum(local_action == local_oracle)),
                "sample_count": int(n),
                "regret_sum": float(np.sum(decision_regret(local_action, local_risk))),
                "unsafe_count": int(np.sum((local_risk == 1) & (local_action == 0))),
            }
        )
        offset += n
    metrics = {
        "nrmse": float(np.sqrt(np.mean(standard_error**2))),
        "nmae": float(np.mean(np.abs(standard_error))),
        "decision_agreement": float(np.mean(action == oracle_action)),
        "regret": float(np.mean(decision_regret(action, risk))),
        "unsafe_rate": float(np.mean((risk == 1) & (action == 0))),
    }
    return metrics, trace_rows


def _cluster_interval(
    trace_rows: list[dict[str, float]],
    numerator: str,
    denominator: str,
    transform: Callable[[float], float] = lambda value: value,
    repetitions: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed("cluster-bootstrap", numerator, denominator))
    n = len(trace_rows)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        selected = rng.integers(0, n, n)
        top = sum(float(trace_rows[index][numerator]) for index in selected)
        bottom = sum(float(trace_rows[index][denominator]) for index in selected)
        estimates[repetition] = transform(top / bottom) if bottom > 0 else np.nan
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def _fit_mapper(
    mapper: BaseMapper,
    training_pairs: list[ProfiledTrace],
    masks: list[np.ndarray],
) -> tuple[BaseMapper, float]:
    started = time.perf_counter()
    mapper.fit(training_pairs, masks)
    return mapper, time.perf_counter() - started


def _predict_timed(mapper: BaseMapper, pairs: list[ProfiledTrace]) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    predictions = [mapper.predict(pair) for pair in pairs]
    elapsed = time.perf_counter() - started
    samples = sum(len(block) for block in predictions)
    return predictions, elapsed * 1e6 / samples


def _calibrate_bridge(
    bridge: KPMBridgeMapper,
    training_pairs: list[ProfiledTrace],
    stats: FeatureStats,
    xapp: PortableRiskXApp,
) -> tuple[np.ndarray, np.ndarray, float, AnchorDriftCalibration]:
    scores: list[np.ndarray] = []
    action_scores: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    for pair in training_pairs:
        mask = calibration_mask(len(pair.trace.values))
        full_prediction = bridge.predict(pair)
        prediction = full_prediction[mask]
        scores.append(joint_standardised_scores(prediction, pair.trace.values[mask], stats.scale))
        action_scores.append(
            np.abs(xapp.logits(prediction) - xapp.logits(pair.trace.values[mask]))
        )
        start = fit_stop(len(full_prediction))
        residual_blocks.append(
            (full_prediction[start:] - pair.trace.values[start:]) / stats.scale
        )
    pooled = np.concatenate(scores)
    pooled_action = np.concatenate(action_scores)
    return (
        pooled,
        pooled_action,
        calibrated_joint_radius(pooled, DEFAULT_ALPHA),
        calibrate_anchor_mean_shift(residual_blocks),
    )


def _drift_state(
    predictions: list[np.ndarray],
    pairs: list[ProfiledTrace],
    stats: FeatureStats,
    calibration: AnchorDriftCalibration,
) -> tuple[list[np.ndarray], list[int | None]]:
    flags: list[np.ndarray] = []
    indices: list[int | None] = []
    for prediction, pair in zip(predictions, pairs, strict=True):
        residuals = (prediction - pair.trace.values) / stats.scale
        drift, detected = detect_anchor_mean_shift(residuals, calibration)
        flags.append(drift)
        indices.append(detected)
    return flags, indices


def _selective_metrics(
    context: ProfileContext,
    xapp: PortableRiskXApp,
    stats: FeatureStats,
    *,
    alpha: float = DEFAULT_ALPHA,
    ignore_drift: bool = False,
    ignore_margin: bool = False,
    materialize_certificates: bool = False,
) -> dict[str, float]:
    radius = calibrated_joint_radius(context.calibration_scores, alpha)
    # A functional conformal projection certifies the fixed xApp logit.  The
    # generic L2 telemetry ball is retained separately for model-independent
    # consumers and coverage reporting.
    action_radius = calibrated_joint_radius(context.calibration_action_scores, alpha)
    margin = action_radius
    total = accepted = disagreements = unsafe = covered = valid_count = 0
    detection_delays: list[float] = []
    detections = false_alarms = drift_traces = 0
    pre_drift_total = pre_drift_accepted = pre_drift_disagreements = 0
    pre_drift_covered = 0
    post_drift_total = post_drift_accepted = post_drift_disagreements = 0
    post_drift_covered = 0
    trace_rows: list[dict[str, float]] = []
    certificate_count = certificate_bytes = 0
    certificate_seconds = 0.0
    reason_counts = {reason: 0 for reason in DecisionReason}
    for pair, prediction, scores, drift, detected in zip(
        context.test_pairs,
        context.test_predictions,
        context.test_scores,
        context.drift_flags,
        context.detection_indices,
        strict=True,
    ):
        n = len(prediction)
        logits = xapp.logits(prediction)
        action = xapp.actions(prediction)
        oracle = xapp.actions(pair.trace.values)
        interval_clear = np.abs(logits - xapp.logit_threshold) > margin
        flags = np.empty(n, dtype=np.uint16)
        for index in range(n):
            flags[index] = int(
                certificate_flags(
                    support=float(pair.observation.support[index]),
                    min_support=MIN_SUPPORT,
                    age_ms=float(pair.observation.age_ms[index]),
                    max_age_ms=MAX_AGE_MS,
                    drift=bool(drift[index]) and not ignore_drift,
                    functional_interval_clear=bool(interval_clear[index]) or ignore_margin,
                )
            )
            reason = decision_reason(int(flags[index]))
            reason_counts[reason] += 1
            if materialize_certificates:
                encode_started = time.perf_counter()
                payload = build_certificate(
                    context.plan,
                    calibration_epoch=1,
                    support=float(pair.observation.support[index]),
                    age_ms=float(pair.observation.age_ms[index]),
                    radius=radius,
                    flags=int(flags[index]),
                ).encode()
                certificate_seconds += time.perf_counter() - encode_started
                certificate_count += 1
                certificate_bytes += len(payload)
        quality = flags == int(CertificateFlag.NONE)
        accepted += int(np.sum(quality))
        local_disagreements = int(np.sum((action != oracle) & quality))
        local_unsafe = int(np.sum((pair.trace.risk == 1) & (action == 0) & quality))
        disagreements += local_disagreements
        unsafe += local_unsafe
        total += n

        valid = np.ones(n, dtype=bool)
        local_pre_total = local_pre_accepted = local_pre_disagreements = 0
        local_pre_covered = 0
        local_post_total = local_post_accepted = local_post_disagreements = 0
        local_post_covered = 0
        if pair.observation.drift_start is not None:
            drift_traces += 1
            change = pair.observation.drift_start
            valid[change:] = False
            local_pre_total = change
            local_pre_accepted = int(np.sum(quality[:change]))
            local_pre_disagreements = int(np.sum((action[:change] != oracle[:change]) & quality[:change]))
            local_pre_covered = int(np.sum(scores[:change] <= radius))
            local_post_total = n - change
            local_post_accepted = int(np.sum(quality[change:]))
            local_post_disagreements = int(np.sum((action[change:] != oracle[change:]) & quality[change:]))
            local_post_covered = int(np.sum(scores[change:] <= radius))
            pre_drift_total += local_pre_total
            pre_drift_accepted += local_pre_accepted
            pre_drift_disagreements += local_pre_disagreements
            pre_drift_covered += local_pre_covered
            post_drift_total += local_post_total
            post_drift_accepted += local_post_accepted
            post_drift_disagreements += local_post_disagreements
            post_drift_covered += local_post_covered
            if detected is not None and detected >= pair.observation.drift_start:
                detection_delays.append((detected - pair.observation.drift_start) * pair.trace.dt_ms)
                detections += 1
            elif detected is not None and detected < pair.observation.drift_start:
                false_alarms += 1
        elif detected is not None:
            false_alarms += 1
        covered += int(np.sum((scores <= radius) & valid))
        valid_count += int(np.sum(valid))
        trace_rows.append(
            {
                "total": float(n),
                "accepted": float(np.sum(quality)),
                "disagreements": float(local_disagreements),
                "unsafe": float(local_unsafe),
                "covered": float(np.sum((scores <= radius) & valid)),
                "valid": float(np.sum(valid)),
                "pre_total": float(local_pre_total),
                "pre_accepted": float(local_pre_accepted),
                "pre_disagreements": float(local_pre_disagreements),
                "pre_covered": float(local_pre_covered),
                "post_total": float(local_post_total),
                "post_accepted": float(local_post_accepted),
                "post_disagreements": float(local_post_disagreements),
                "post_covered": float(local_post_covered),
            }
        )

    selective_low, selective_high = _cluster_interval(
        trace_rows, "disagreements", "accepted"
    )
    coverage_low, coverage_high = _cluster_interval(trace_rows, "covered", "valid")
    acceptance_low, acceptance_high = _cluster_interval(trace_rows, "accepted", "total")
    post_selective_low, post_selective_high = _cluster_interval(
        trace_rows, "post_disagreements", "post_accepted"
    )
    post_exposure_low, post_exposure_high = _cluster_interval(
        trace_rows, "post_disagreements", "post_total"
    )
    return {
        "radius": float(radius),
        "action_radius": float(action_radius),
        "coverage_valid_regime": covered / valid_count,
        "coverage_cluster_low": coverage_low,
        "coverage_cluster_high": coverage_high,
        "acceptance": accepted / total,
        "acceptance_cluster_low": acceptance_low,
        "acceptance_cluster_high": acceptance_high,
        "abstention": 1.0 - accepted / total,
        "selective_error": disagreements / accepted if accepted else float("nan"),
        "selective_error_cluster_low": selective_low,
        "selective_error_cluster_high": selective_high,
        "accepted_unsafe_rate": unsafe / accepted if accepted else float("nan"),
        "decision_error_exposure": disagreements / total,
        "pre_drift_coverage": pre_drift_covered / pre_drift_total if pre_drift_total else float("nan"),
        "pre_drift_acceptance": pre_drift_accepted / pre_drift_total if pre_drift_total else float("nan"),
        "pre_drift_selective_error": (
            pre_drift_disagreements / pre_drift_accepted
            if pre_drift_accepted
            else float("nan")
        ),
        "pre_drift_error_exposure": (
            pre_drift_disagreements / pre_drift_total
            if pre_drift_total
            else float("nan")
        ),
        "post_drift_coverage": post_drift_covered / post_drift_total if post_drift_total else float("nan"),
        "post_drift_acceptance": post_drift_accepted / post_drift_total if post_drift_total else float("nan"),
        "post_drift_selective_error": (
            post_drift_disagreements / post_drift_accepted
            if post_drift_accepted
            else float("nan")
        ),
        "post_drift_selective_error_cluster_low": post_selective_low,
        "post_drift_selective_error_cluster_high": post_selective_high,
        "post_drift_error_exposure": (
            post_drift_disagreements / post_drift_total
            if post_drift_total
            else float("nan")
        ),
        "post_drift_error_exposure_cluster_low": post_exposure_low,
        "post_drift_error_exposure_cluster_high": post_exposure_high,
        "median_detection_delay_ms": float(np.median(detection_delays)) if detection_delays else float("nan"),
        "detection_rate": detections / drift_traces if drift_traces else float("nan"),
        "false_alarm_rate_per_trace": false_alarms / len(context.test_pairs),
        "certificate_count": certificate_count,
        "certificate_bytes_total": certificate_bytes,
        "certificate_encoding_us_per_report": (
            certificate_seconds * 1e6 / certificate_count if certificate_count else float("nan")
        ),
        **{f"reason_{reason.value.lower()}": count for reason, count in reason_counts.items()},
    }


def evaluate_profile(
    profile: ImplementationProfile,
    training_traces: list[CanonicalTrace],
    test_traces: list[CanonicalTrace],
    stats: FeatureStats,
    xapp: PortableRiskXApp,
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], ProfileContext]:
    plan = compile_profile_plan(profile)
    pairs = build_pairs(training_traces + test_traces, profile, stats)
    training_pairs = [pair for pair in pairs if pair.trace.experiment == "exp1"]
    test_pairs = [pair for pair in pairs if pair.trace.experiment == "exp2"]
    masks = [fit_mask(len(pair.trace.values), anchor_fraction) for pair in training_pairs]
    methods: list[BaseMapper] = [
        RawIdentityMapper(),
        ContractMapper(plan),
        DistributionMapper("zscore"),
        DistributionMapper("coral"),
        DistributionMapper("ot"),
        AnchorRidgeMapper(plan),
        TemporalRidgeMapper(plan),
        KPMBridgeMapper(plan, stats),
    ]

    result_rows: list[dict[str, object]] = []
    trace_rows_all: list[dict[str, object]] = []
    fitted: dict[str, BaseMapper] = {}
    bridge_predictions: list[np.ndarray] | None = None
    for mapper in methods:
        mapper, fit_seconds = _fit_mapper(mapper, training_pairs, masks)
        predictions, inference_us = _predict_timed(mapper, test_pairs)
        metrics, trace_rows = _method_metrics(predictions, test_pairs, xapp, stats)
        nrmse_low, nrmse_high = _cluster_interval(
            trace_rows, "sse", "error_count", transform=np.sqrt
        )
        agreement_low, agreement_high = _cluster_interval(
            trace_rows, "agreement_count", "sample_count"
        )
        result_rows.append(
            {
                "profile": profile.key,
                "profile_label": profile.label,
                "method": mapper.name,
                **metrics,
                "nrmse_ci_low": nrmse_low,
                "nrmse_ci_high": nrmse_high,
                "agreement_ci_low": agreement_low,
                "agreement_ci_high": agreement_high,
                "fit_seconds": fit_seconds,
                "inference_us_per_sample": inference_us,
                "model_bytes": mapper.serialized_bytes(),
                "anchor_fraction": anchor_fraction,
            }
        )
        trace_rows_all.extend({"profile": profile.key, "method": mapper.name, **row} for row in trace_rows)
        fitted[mapper.name] = mapper
        if mapper.name == "KPM-Bridge":
            bridge_predictions = predictions

    # Non-portable, deployment-specific upper bound.
    deployment_masks = [
        xapp_fit_mask(len(pair.trace.values), PREDICTION_HORIZON)
        for pair in training_pairs
    ]
    raw_train = np.vstack(
        [pair.observation.raw[mask] for pair, mask in zip(training_pairs, deployment_masks, strict=True)]
    )
    risk_train = np.concatenate(
        [pair.trace.risk[mask] for pair, mask in zip(training_pairs, deployment_masks, strict=True)]
    )
    raw_test = np.vstack([pair.observation.raw for pair in test_pairs])
    risk_test = np.concatenate([pair.trace.risk for pair in test_pairs])
    truth_test = np.vstack([pair.trace.values for pair in test_pairs])
    oracle_action = xapp.actions(truth_test)
    deployment_model = DeploymentSpecificRiskXApp(SEED)
    started = time.perf_counter()
    deployment_model.fit(raw_train, risk_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    deployment_action = deployment_model.actions(raw_test)
    inference_us = (time.perf_counter() - started) * 1e6 / len(raw_test)
    result_rows.append(
        {
            "profile": profile.key,
            "profile_label": profile.label,
            "method": "Deployment retrain",
            "nrmse": float("nan"),
            "nmae": float("nan"),
            "decision_agreement": float(np.mean(deployment_action == oracle_action)),
            "regret": float(np.mean(decision_regret(deployment_action, risk_test))),
            "unsafe_rate": float(np.mean((risk_test == 1) & (deployment_action == 0))),
            "nrmse_ci_low": float("nan"),
            "nrmse_ci_high": float("nan"),
            "agreement_ci_low": float("nan"),
            "agreement_ci_high": float("nan"),
            "fit_seconds": fit_seconds,
            "inference_us_per_sample": inference_us,
            "model_bytes": float("nan"),
            "anchor_fraction": 1.0,
        }
    )
    result_rows.append(
        {
            "profile": profile.key,
            "profile_label": profile.label,
            "method": "Oracle canonical",
            "nrmse": 0.0,
            "nmae": 0.0,
            "decision_agreement": 1.0,
            "regret": float(np.mean(decision_regret(oracle_action, risk_test))),
            "unsafe_rate": float(np.mean((risk_test == 1) & (oracle_action == 0))),
            "nrmse_ci_low": 0.0,
            "nrmse_ci_high": 0.0,
            "agreement_ci_low": 1.0,
            "agreement_ci_high": 1.0,
            "fit_seconds": 0.0,
            "inference_us_per_sample": 0.0,
            "model_bytes": 0,
            "anchor_fraction": 0.0,
        }
    )

    bridge = fitted["KPM-Bridge"]
    assert isinstance(bridge, KPMBridgeMapper) and bridge_predictions is not None
    calibration_scores, calibration_action_scores, radius, drift_calibration = _calibrate_bridge(
        bridge, training_pairs, stats, xapp
    )
    test_scores = [
        joint_standardised_scores(prediction, pair.trace.values, stats.scale)
        for prediction, pair in zip(bridge_predictions, test_pairs, strict=True)
    ]
    drift_flags, detection_indices = _drift_state(
        bridge_predictions, test_pairs, stats, drift_calibration
    )
    context = ProfileContext(
        profile,
        plan,
        training_pairs,
        test_pairs,
        bridge,
        calibration_scores,
        calibration_action_scores,
        radius,
        drift_calibration,
        bridge_predictions,
        test_scores,
        drift_flags,
        detection_indices,
    )
    selective = _selective_metrics(
        context, xapp, stats, materialize_certificates=True
    )
    selective_rows = [{"profile": profile.key, "variant": "Full", **selective}]

    no_margin = _selective_metrics(context, xapp, stats, ignore_margin=True)
    selective_rows.append({"profile": profile.key, "variant": "No uncertainty gate", **no_margin})
    no_drift = _selective_metrics(context, xapp, stats, ignore_drift=True)
    selective_rows.append({"profile": profile.key, "variant": "No drift gate", **no_drift})

    # Structural ablations share the exact same anchors and test traces.
    ablation_rows: list[dict[str, object]] = []
    for label, method_name in (
        ("No residual/temporal", "Contract"),
        ("No temporal", "Anchor ridge"),
        ("Linear residual", "Temporal ridge"),
        ("Full", "KPM-Bridge"),
    ):
        row = next(item for item in result_rows if item["method"] == method_name)
        ablation_rows.append(
            {
                "profile": profile.key,
                "variant": label,
                "nrmse": row["nrmse"],
                "decision_agreement": row["decision_agreement"],
            }
        )
    no_contract = KPMBridgeMapper(plan, stats, use_contract=False)
    no_contract.fit(training_pairs, masks)
    no_contract_predictions = [no_contract.predict(pair) for pair in test_pairs]
    no_contract_metrics, _ = _method_metrics(no_contract_predictions, test_pairs, xapp, stats)
    ablation_rows.append(
        {
            "profile": profile.key,
            "variant": "No typed contract",
            "nrmse": no_contract_metrics["nrmse"],
            "decision_agreement": no_contract_metrics["decision_agreement"],
        }
    )
    return result_rows, selective_rows, ablation_rows, context


def _evaluate_bridge_once(
    profile: ImplementationProfile,
    training_traces: list[CanonicalTrace],
    test_traces: list[CanonicalTrace],
    stats: FeatureStats,
    xapp: PortableRiskXApp,
    anchor_fraction: float,
) -> dict[str, float]:
    plan = compile_profile_plan(profile)
    pairs = build_pairs(training_traces + test_traces, profile, stats)
    train_pairs = [pair for pair in pairs if pair.trace.experiment == "exp1"]
    test_pairs = [pair for pair in pairs if pair.trace.experiment == "exp2"]
    masks = [fit_mask(len(pair.trace.values), anchor_fraction) for pair in train_pairs]
    bridge = KPMBridgeMapper(plan, stats, max_iter=60).fit(train_pairs, masks)
    calibration_scores, _, radius, _ = _calibrate_bridge(bridge, train_pairs, stats, xapp)
    predictions = [bridge.predict(pair) for pair in test_pairs]
    metrics, _ = _method_metrics(predictions, test_pairs, xapp, stats)
    scores = np.concatenate(
        [
            joint_standardised_scores(prediction, pair.trace.values, stats.scale)
            for prediction, pair in zip(predictions, test_pairs, strict=True)
        ]
    )
    return {
        "anchor_fraction": anchor_fraction,
        "nrmse": metrics["nrmse"],
        "decision_agreement": metrics["decision_agreement"],
        "radius": radius,
        "coverage": float(np.mean(scores <= radius)),
        "fit_seconds": bridge.fit_seconds,
        "calibration_samples": int(len(calibration_scores)),
    }


def sensitivity_analysis(
    contexts: dict[str, ProfileContext],
    profiles: dict[str, ImplementationProfile],
    training_traces: list[CanonicalTrace],
    test_traces: list[CanonicalTrace],
    stats: FeatureStats,
    xapp: PortableRiskXApp,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20):
        result = _evaluate_bridge_once(
            profiles["P3"], training_traces, test_traces, stats, xapp, fraction
        )
        rows.append({"sweep": "anchor_fraction", "value": fraction, "profile": "P3", **result})

    # Stress a mapper trained under nominal P1 without refitting.
    nominal = contexts["P1"]
    for missing_rate in (0.0, 0.05, 0.10, 0.20, 0.30):
        stressed = with_stress(profiles["P1"], missing_rate=missing_rate)
        pairs = build_pairs(test_traces, stressed, stats, inject_test_drift=False)
        predictions = [nominal.bridge.predict(pair) for pair in pairs]
        metrics, _ = _method_metrics(predictions, pairs, xapp, stats)
        rows.append(
            {
                "sweep": "missing_rate",
                "value": missing_rate,
                "profile": "P1",
                "nrmse": metrics["nrmse"],
                "decision_agreement": metrics["decision_agreement"],
                "mean_support": float(np.mean(np.concatenate([pair.observation.support for pair in pairs]))),
            }
        )

    for lag in (0, 1, 2, 4, 6):
        stressed = with_stress(profiles["P1"], lag=lag)
        pairs = build_pairs(test_traces, stressed, stats, inject_test_drift=False)
        predictions = [nominal.bridge.predict(pair) for pair in pairs]
        metrics, _ = _method_metrics(predictions, pairs, xapp, stats)
        rows.append(
            {
                "sweep": "lag_samples",
                "value": lag,
                "profile": "P1",
                "nrmse": metrics["nrmse"],
                "decision_agreement": metrics["decision_agreement"],
                "mean_age_ms": float(np.mean(np.concatenate([pair.observation.age_ms for pair in pairs]))),
            }
        )

    for alpha in (0.01, 0.025, 0.05, 0.10, 0.20):
        metrics = [_selective_metrics(context, xapp, stats, alpha=alpha) for context in contexts.values()]
        rows.append(
            {
                "sweep": "alpha",
                "value": alpha,
                "profile": "mean(P1:P4)",
                "coverage": float(np.mean([metric["coverage_valid_regime"] for metric in metrics])),
                "abstention": float(np.mean([metric["abstention"] for metric in metrics])),
                "selective_error": float(np.nanmean([metric["selective_error"] for metric in metrics])),
                "radius": float(np.mean([metric["radius"] for metric in metrics])),
            }
        )

    drift_context = contexts["P4"]
    base_drift = profiles["P4"]
    for magnitude in (0.0, 0.25, 0.50, 0.75, 1.00, 1.25):
        ratio = magnitude / base_drift.drift_offset
        stressed = replace(
            base_drift,
            drift_gain=base_drift.drift_gain * ratio,
            drift_offset=magnitude,
            drift_noise_multiplier=1.0 + (base_drift.drift_noise_multiplier - 1.0) * ratio,
            drift_lag_increment=int(round(base_drift.drift_lag_increment * ratio)),
        )
        pairs = build_pairs(test_traces, stressed, stats, inject_test_drift=True)
        predictions = [drift_context.bridge.predict(pair) for pair in pairs]
        scores = [
            joint_standardised_scores(prediction, pair.trace.values, stats.scale)
            for prediction, pair in zip(predictions, pairs, strict=True)
        ]
        flags, indices = _drift_state(
            predictions, pairs, stats, drift_context.drift_calibration
        )
        delays = [
            (detected - pair.observation.drift_start) * pair.trace.dt_ms
            for pair, detected in zip(pairs, indices, strict=True)
            if detected is not None
            and pair.observation.drift_start is not None
            and detected >= pair.observation.drift_start
        ]
        post_scores = np.concatenate(
            [score[pair.observation.drift_start :] for pair, score in zip(pairs, scores, strict=True)]
        )
        true_detections = [
            detected is not None
            and pair.observation.drift_start is not None
            and detected >= pair.observation.drift_start
            for pair, detected in zip(pairs, indices, strict=True)
        ]
        false_alarms = [
            detected is not None
            and pair.observation.drift_start is not None
            and detected < pair.observation.drift_start
            for pair, detected in zip(pairs, indices, strict=True)
        ]
        rows.append(
            {
                "sweep": "drift_magnitude",
                "value": magnitude,
                "profile": "P4",
                "post_drift_coverage": float(np.mean(post_scores <= drift_context.radius)),
                "detection_rate": float(np.mean(true_detections)),
                "false_alarm_rate": float(np.mean(false_alarms)),
                "median_detection_delay_ms": float(np.median(delays)) if delays else float("nan"),
                "post_flagged_fraction": float(np.mean(np.concatenate(flags))),
            }
        )
    return rows


def semantic_compiler_checks() -> list[dict[str, object]]:
    """Exercise semantic fail-closed and order-invariance properties end to end."""
    profile = default_profiles()["P1"]
    sources = profile_contracts(profile)
    targets = canonical_contracts()
    canonical = np.array(
        [
            [100_000.0, 2.0, 10.0, 12.0, -65.0, 40_000.0, 1.0, 20.0],
            [120_000.0, 3.0, 11.0, 13.0, -64.0, 45_000.0, 2.0, 25.0],
        ]
    )
    raw = canonical * profile.declared_scale + profile.declared_offset
    reset = np.zeros(raw.shape, dtype=bool)
    reference_plan = compile_transform_plan(sources, targets)
    reference = reference_plan.apply(raw, reset_mask=reset, dt_ms=250.0)
    rows: list[dict[str, object]] = [
        {
            "check": "declared_unit_conversion",
            "expected": "canonical values",
            "observed": "canonical values" if np.allclose(reference, canonical) else "mismatch",
            "status": "PASS" if np.allclose(reference, canonical) else "FAIL",
        }
    ]

    permutation = np.array([3, 0, 7, 1, 6, 5, 2, 4])
    permuted_plan = compile_transform_plan(
        [sources[index] for index in permutation], targets
    )
    permuted = permuted_plan.apply(
        raw[:, permutation], reset_mask=reset[:, permutation], dt_ms=250.0
    )
    rows.append(
        {
            "check": "permuted_source_order",
            "expected": "order-invariant canonical output",
            "observed": "equal" if np.allclose(permuted, reference) else "mismatch",
            "status": "PASS" if np.allclose(permuted, reference) else "FAIL",
        }
    )
    for label, bad_source in (
        (
            "wrong_quantity_decoy",
            replace(sources[0], quantity="semantic_decoy_quantity"),
        ),
        ("permuted_scope_decoy", replace(sources[0], entity_scope="cell")),
    ):
        corrupted = [bad_source, *sources[1:]]
        rejected = False
        try:
            compile_transform_plan(corrupted, targets)
        except ContractError:
            rejected = True
        rows.append(
            {
                "check": label,
                "expected": "fail closed",
                "observed": "rejected" if rejected else "accepted",
                "status": "PASS" if rejected else "FAIL",
            }
        )
    return rows


def monitor_window_sensitivity(
    context: ProfileContext,
    stats: FeatureStats,
    xapp: PortableRiskXApp,
    *,
    candidates: tuple[int, ...] = (1, 2, 3, 5),
) -> tuple[list[dict[str, object]], int]:
    """Select a drift window on exp1 only, then report frozen exp2 sensitivity.

    The sorted exp1 streams are split 60/40.  The first block estimates each
    candidate's trace-maximum threshold; the second measures false alarms and
    a predeclared 1.25-scale residual shift.  Exp2 columns are diagnostics and
    never enter selection.
    """
    calibration_blocks: list[np.ndarray] = []
    fitting_blocks: list[np.ndarray] = []
    for pair in context.training_pairs:
        prediction = context.bridge.predict(pair)
        stop = fit_stop(len(prediction))
        residual = (prediction - pair.trace.values) / stats.scale
        calibration_blocks.append(residual[stop:])
        fitting_blocks.append(residual[:stop])
    design_stop = max(1, int(np.floor(0.60 * len(calibration_blocks))))
    direction = np.array([1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
    rows: list[dict[str, object]] = []
    for window in candidates:
        design_calibration = calibrate_anchor_mean_shift(
            calibration_blocks[:design_stop],
            window=window,
            quantile=0.975,
            anchor_stride=10,
        )
        validation_detections = [
            detect_anchor_mean_shift(block, design_calibration)[1]
            for block in calibration_blocks[design_stop:]
        ]
        validation_false_alarm = float(
            np.mean([detected is not None for detected in validation_detections])
        )
        synthetic_detected: list[bool] = []
        synthetic_delays: list[float] = []
        for block in fitting_blocks[design_stop:]:
            shifted = block.copy()
            change = int(np.floor(0.55 * len(shifted)))
            shifted[change:] += 1.25 * direction
            _, detected = detect_anchor_mean_shift(shifted, design_calibration)
            valid_detection = detected is not None and detected >= change
            synthetic_detected.append(valid_detection)
            if valid_detection:
                synthetic_delays.append((detected - change) * 250.0)

        final_calibration = calibrate_anchor_mean_shift(
            calibration_blocks,
            window=window,
            quantile=0.975,
            anchor_stride=10,
        )
        flags, indices = _drift_state(
            context.test_predictions,
            context.test_pairs,
            stats,
            final_calibration,
        )
        candidate_context = replace(
            context,
            drift_calibration=final_calibration,
            drift_flags=flags,
            detection_indices=indices,
        )
        test_metrics = _selective_metrics(candidate_context, xapp, stats)
        rows.append(
            {
                "window": window,
                "selection_validation_false_alarm": validation_false_alarm,
                "selection_synthetic_detection": float(np.mean(synthetic_detected)),
                "selection_synthetic_median_delay_ms": (
                    float(np.median(synthetic_delays))
                    if synthetic_delays
                    else float("nan")
                ),
                "test_false_alarm_rate_per_trace": test_metrics[
                    "false_alarm_rate_per_trace"
                ],
                "test_detection_rate": test_metrics["detection_rate"],
                "test_median_detection_delay_ms": test_metrics[
                    "median_detection_delay_ms"
                ],
                "test_post_drift_acceptance": test_metrics[
                    "post_drift_acceptance"
                ],
                "test_post_drift_selective_error": test_metrics[
                    "post_drift_selective_error"
                ],
                "test_post_drift_error_exposure": test_metrics[
                    "post_drift_error_exposure"
                ],
                "test_overall_selective_error": test_metrics["selective_error"],
            }
        )

    eligible = [
        row for row in rows if row["selection_validation_false_alarm"] <= 0.05
    ]
    pool = eligible if eligible else rows
    chosen = int(
        min(
            pool,
            key=lambda row: (
                -float(row["selection_synthetic_detection"]),
                float(row["selection_synthetic_median_delay_ms"]),
                int(row["window"]),
            ),
        )["window"]
    )
    for row in rows:
        row["selected_from_exp1_only"] = bool(row["window"] == chosen)
    return rows, chosen


def run_full_benchmark(output_dir: Path = Path("reproducibility/outputs")) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_colosseum_subset()
    labelled, thresholds = attach_qos_risk_labels(traces)
    training = [trace for trace in labelled if trace.experiment == "exp1"]
    test = [trace for trace in labelled if trace.experiment == "exp2"]
    stats = robust_feature_stats(training)
    xapp, xapp_summary = fit_portable_xapp(training, test)
    profiles = default_profiles()

    main_rows: list[dict[str, object]] = []
    selective_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    contexts: dict[str, ProfileContext] = {}
    for key in ("P1", "P2", "P3", "P4"):
        results, selective, ablations, context = evaluate_profile(
            profiles[key], training, test, stats, xapp
        )
        main_rows.extend(results)
        selective_rows.extend(selective)
        ablation_rows.extend(ablations)
        contexts[key] = context

    monitor_rows, selected_monitor_window = monitor_window_sensitivity(
        contexts["P4"], stats, xapp
    )
    sensitivity_rows = sensitivity_analysis(contexts, profiles, training, test, stats, xapp)
    main_frame = pd.DataFrame(main_rows)
    selective_frame = pd.DataFrame(selective_rows)
    ablation_frame = pd.DataFrame(ablation_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    semantic_frame = pd.DataFrame(semantic_compiler_checks())
    monitor_frame = pd.DataFrame(monitor_rows)
    main_frame.to_csv(output_dir / "main_results.csv", index=False)
    selective_frame.to_csv(output_dir / "selective_results.csv", index=False)
    ablation_frame.to_csv(output_dir / "ablation_results.csv", index=False)
    sensitivity_frame.to_csv(output_dir / "sensitivity_results.csv", index=False)
    semantic_frame.to_csv(output_dir / "semantic_results.csv", index=False)
    monitor_frame.to_csv(output_dir / "monitor_sensitivity_results.csv", index=False)

    split_summary = {
        "fit_fraction": 0.60,
        "calibration_stride": 4,
        "prediction_horizon": PREDICTION_HORIZON,
        "mapper_anchor_rows": int(
            sum(np.sum(fit_mask(len(trace.values), DEFAULT_ANCHOR_FRACTION)) for trace in training)
        ),
        "xapp_fit_rows": int(
            sum(np.sum(xapp_fit_mask(len(trace.values), PREDICTION_HORIZON)) for trace in training)
        ),
        "calibration_rows": int(
            sum(np.sum(calibration_mask(len(trace.values))) for trace in training)
        ),
        "feature_statistics_source": "exp1 fitting prefixes only",
        "disjoint_fit_calibration": bool(
            all(
                not np.any(
                    xapp_fit_mask(len(trace.values), PREDICTION_HORIZON)
                    & calibration_mask(len(trace.values))
                )
                for trace in training
            )
        ),
    }
    full_certificates = selective_frame[selective_frame.variant == "Full"]

    summary = {
        "status": "FULL_DETERMINISTIC_BENCHMARK",
        "seed": SEED,
        "input_traces": len(traces),
        "training_traces": len(training),
        "test_traces": len(test),
        "training_rows": int(sum(len(trace.values) for trace in training)),
        "test_rows": int(sum(len(trace.values) for trace in test)),
        "features": 8,
        "risk_thresholds": thresholds,
        "feature_location": stats.location.tolist(),
        "feature_scale": stats.scale.tolist(),
        "xapp": xapp_summary,
        "certificate_bytes": 48,
        "certificate_materialization": {
            "reports": int(full_certificates.certificate_count.sum()),
            "encoded_bytes": int(full_certificates.certificate_bytes_total.sum()),
            "mean_encoding_us_per_report": float(
                np.average(
                    full_certificates.certificate_encoding_us_per_report,
                    weights=full_certificates.certificate_count,
                )
            ),
        },
        "split_protocol": split_summary,
        "compiled_plans": {
            key: {
                "schema_hash_hex": contexts[key].plan.schema_hash.hex(),
                "mapping_id": contexts[key].plan.mapping_id,
                "semantic_cost": contexts[key].plan.total_semantic_cost,
            }
            for key in contexts
        },
        "semantic_checks": {
            "checks": int(len(semantic_frame)),
            "passed": int((semantic_frame.status == "PASS").sum()),
        },
        "monitor_selection": {
            "selected_window": selected_monitor_window,
            "selection_data": "exp1 only",
            "validation_false_alarm_ceiling": 0.05,
            "synthetic_shift_scale": 1.25,
            "test_data_used_for_selection": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
