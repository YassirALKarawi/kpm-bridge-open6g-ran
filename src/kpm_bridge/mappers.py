"""Cross-implementation mapping baselines and the temporal KPM-Bridge model."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import CompiledTransformPlan
from .dataset import CanonicalTrace, FeatureStats
from .profiles import Observation
from .splits import calibration_mask as split_calibration_mask
from .splits import mapper_fit_mask


@dataclass(frozen=True)
class ProfiledTrace:
    trace: CanonicalTrace
    observation: Observation


def fit_mask(length: int, anchor_fraction: float, fit_fraction: float = 0.60) -> np.ndarray:
    if fit_fraction != 0.60:
        raise ValueError("the registered benchmark fixes fit_fraction at 0.60")
    return mapper_fit_mask(length, anchor_fraction)


def calibration_mask(length: int, fit_fraction: float = 0.60, stride: int = 4) -> np.ndarray:
    if fit_fraction != 0.60:
        raise ValueError("the registered benchmark fixes fit_fraction at 0.60")
    return split_calibration_mask(length, stride)


def _stack_selected(arrays: Iterable[np.ndarray], masks: Iterable[np.ndarray]) -> np.ndarray:
    blocks = [array[mask] for array, mask in zip(arrays, masks, strict=True) if np.any(mask)]
    if not blocks:
        raise ValueError("selection produced no samples")
    return np.vstack(blocks)


def _median_impute(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, medians)


def _symmetric_power(matrix: np.ndarray, exponent: float, floor: float = 1e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors * values**exponent) @ vectors.T


class BaseMapper:
    name = "base"
    supervised = False

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "BaseMapper":
        raise NotImplementedError

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        raise NotImplementedError

    def serialized_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


class RawIdentityMapper(BaseMapper):
    name = "Direct"

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "RawIdentityMapper":
        raw = _stack_selected((pair.observation.raw for pair in pairs), masks)
        self.medians = np.nanmedian(raw, axis=0)
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        return _median_impute(pair.observation.raw, self.medians)


class ContractMapper(BaseMapper):
    name = "Contract"

    def __init__(self, plan: CompiledTransformPlan):
        self.plan = plan

    def _values(self, pair: ProfiledTrace) -> np.ndarray:
        return self.plan.apply(
            pair.observation.raw,
            reset_mask=pair.observation.reset_mask,
            dt_ms=pair.trace.dt_ms,
        )

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "ContractMapper":
        values = _stack_selected((self._values(pair) for pair in pairs), masks)
        self.medians = np.nanmedian(values, axis=0)
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        return _median_impute(self._values(pair), self.medians)


class DistributionMapper(BaseMapper):
    """Featurewise z-score, CORAL, or Gaussian optimal-transport alignment."""

    def __init__(self, kind: str):
        if kind not in {"zscore", "coral", "ot"}:
            raise ValueError("kind must be zscore, coral, or ot")
        self.kind = kind
        self.name = {"zscore": "Z-score", "coral": "CORAL", "ot": "Gaussian OT"}[kind]

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "DistributionMapper":
        raw = _stack_selected((pair.observation.raw for pair in pairs), masks)
        target = _stack_selected((pair.trace.values for pair in pairs), masks)
        self.raw_median = np.nanmedian(raw, axis=0)
        raw = _median_impute(raw, self.raw_median)
        self.source_mean = np.mean(raw, axis=0)
        self.target_mean = np.mean(target, axis=0)
        source_std = np.maximum(np.std(raw, axis=0, ddof=1), 1e-8)
        target_std = np.maximum(np.std(target, axis=0, ddof=1), 1e-8)
        if self.kind == "zscore":
            self.transform = np.diag(target_std / source_std)
        else:
            regularisation = 1e-5
            source_cov = np.cov(raw, rowvar=False) + regularisation * np.eye(raw.shape[1])
            target_cov = np.cov(target, rowvar=False) + regularisation * np.eye(raw.shape[1])
            source_half = _symmetric_power(source_cov, 0.5)
            source_inverse_half = _symmetric_power(source_cov, -0.5)
            if self.kind == "coral":
                self.transform = source_inverse_half @ _symmetric_power(target_cov, 0.5)
            else:
                middle = source_half @ target_cov @ source_half
                self.transform = source_inverse_half @ _symmetric_power(middle, 0.5) @ source_inverse_half
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        raw = _median_impute(pair.observation.raw, self.raw_median)
        return (raw - self.source_mean) @ self.transform + self.target_mean


class AnchorRidgeMapper(BaseMapper):
    name = "Anchor ridge"
    supervised = True

    def __init__(self, plan: CompiledTransformPlan, use_contract: bool = True):
        self.plan = plan
        self.use_contract = use_contract

    def _values(self, pair: ProfiledTrace) -> np.ndarray:
        if self.use_contract:
            return self.plan.apply(
                pair.observation.raw,
                reset_mask=pair.observation.reset_mask,
                dt_ms=pair.trace.dt_ms,
            )
        return pair.observation.raw

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "AnchorRidgeMapper":
        source = _stack_selected((self._values(pair) for pair in pairs), masks)
        target = _stack_selected((pair.trace.values for pair in pairs), masks)
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=2.0),
        )
        self.model.fit(source, target)
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        return self.model.predict(self._values(pair))


def temporal_design(
    pair: ProfiledTrace,
    plan: CompiledTransformPlan,
    use_contract: bool = True,
) -> np.ndarray:
    current = (
        plan.apply(
            pair.observation.raw,
            reset_mask=pair.observation.reset_mask,
            dt_ms=pair.trace.dt_ms,
        )
        if use_contract
        else pair.observation.raw
    )
    n, d = current.shape
    blocks: list[np.ndarray] = []
    for lag in (0, 1, 2, 4, 8):
        shifted = np.full((n, d), np.nan, dtype=float)
        if lag == 0:
            shifted[:] = current
        else:
            shifted[lag:] = current[:-lag]
        blocks.append(shifted)
    delta = np.full_like(current, np.nan)
    delta[1:] = current[1:] - current[:-1]
    blocks.extend(
        [
            delta,
            np.isfinite(current).astype(float),
            pair.observation.age_ms[:, None] / 1000.0,
            pair.observation.support[:, None],
        ]
    )
    return np.column_stack(blocks)


class TemporalRidgeMapper(BaseMapper):
    name = "Temporal ridge"
    supervised = True

    def __init__(self, plan: CompiledTransformPlan, use_contract: bool = True):
        self.plan = plan
        self.use_contract = use_contract

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "TemporalRidgeMapper":
        source = _stack_selected(
            (temporal_design(pair, self.plan, self.use_contract) for pair in pairs), masks
        )
        target = _stack_selected((pair.trace.values for pair in pairs), masks)
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=8.0),
        )
        self.model.fit(source, target)
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        return self.model.predict(temporal_design(pair, self.plan, self.use_contract))


class KPMBridgeMapper(BaseMapper):
    name = "KPM-Bridge"
    supervised = True

    def __init__(
        self,
        plan: CompiledTransformPlan,
        stats: FeatureStats,
        max_iter: int = 70,
        use_contract: bool = True,
        random_state: int = 20260712,
    ):
        self.plan = plan
        self.stats = stats
        self.max_iter = max_iter
        self.use_contract = use_contract
        self.random_state = random_state

    def fit(self, pairs: list[ProfiledTrace], masks: list[np.ndarray]) -> "KPMBridgeMapper":
        started = time.perf_counter()
        source = _stack_selected(
            (temporal_design(pair, self.plan, self.use_contract) for pair in pairs), masks
        )
        target = _stack_selected((pair.trace.values for pair in pairs), masks)
        target_standard = (target - self.stats.location) / self.stats.scale
        self.models: list[HistGradientBoostingRegressor] = []
        for column in range(target.shape[1]):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.075,
                max_iter=self.max_iter,
                max_leaf_nodes=15,
                max_depth=5,
                min_samples_leaf=30,
                l2_regularization=0.20,
                early_stopping=False,
                random_state=self.random_state + column,
            )
            model.fit(source, target_standard[:, column])
            self.models.append(model)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, pair: ProfiledTrace) -> np.ndarray:
        source = temporal_design(pair, self.plan, self.use_contract)
        standard = np.column_stack([model.predict(source) for model in self.models])
        return standard * self.stats.scale + self.stats.location
