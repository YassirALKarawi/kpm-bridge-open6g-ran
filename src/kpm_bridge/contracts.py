"""Typed KPM contracts and physically safe unit conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


class ContractError(ValueError):
    """Raised when a KPM contract is internally inconsistent."""


@dataclass(frozen=True)
class CompiledMapping:
    source_index: int
    target_index: int
    semantic_cost: float


@dataclass(frozen=True)
class CompiledTransformPlan:
    """Versioned, executable source-to-canonical contract plan."""

    sources: tuple["KPMContract", ...]
    targets: tuple["KPMContract", ...]
    mappings: tuple[CompiledMapping, ...]
    schema_hash: bytes
    mapping_id: int
    version: str

    @property
    def total_semantic_cost(self) -> float:
        return float(sum(item.semantic_cost for item in self.mappings))

    def apply(
        self,
        values: np.ndarray,
        *,
        reset_mask: np.ndarray | None,
        dt_ms: float,
    ) -> np.ndarray:
        """Apply unit and counter transforms in canonical target order.

        Cumulative counters are differentiated only across two finite samples.
        A negative delta is accepted as a reset exclusively when the report
        carries reset metadata for that source and event; otherwise the
        derived value remains missing.
        """
        source_values = np.asarray(values, dtype=float)
        if source_values.ndim != 2 or source_values.shape[1] != len(self.sources):
            raise ContractError("source values do not match the compiled contract plan")
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ContractError("dt_ms must be finite and positive")
        if reset_mask is None:
            resets = np.zeros(source_values.shape, dtype=bool)
        else:
            resets = np.asarray(reset_mask, dtype=bool)
            if resets.shape != source_values.shape:
                raise ContractError("reset metadata does not match source values")

        output = np.full((len(source_values), len(self.targets)), np.nan, dtype=float)
        dt_s = dt_ms / 1000.0
        for mapping in self.mappings:
            source = self.sources[mapping.source_index]
            target = self.targets[mapping.target_index]
            if source.counter_semantics == "cumulative":
                if source.dimension != "data":
                    raise ContractError("cumulative rate sources must use a data unit")
                column = convert_units(
                    source_values[:, mapping.source_index], source.unit, "bit"
                )
                rate = np.full(len(column), np.nan, dtype=float)
                for index in range(1, len(column)):
                    if not (np.isfinite(column[index]) and np.isfinite(column[index - 1])):
                        continue
                    delta = column[index] - column[index - 1]
                    if resets[index, mapping.source_index]:
                        rate[index] = column[index] / dt_s
                    elif delta >= 0.0:
                        rate[index] = delta / dt_s
                column = convert_units(rate, "bit/s", target.unit)
            elif source.counter_semantics == "delta":
                if source.dimension != "data":
                    raise ContractError("delta rate sources must use a data unit")
                column = convert_units(
                    source_values[:, mapping.source_index], source.unit, "bit"
                )
                column = convert_units(column / dt_s, "bit/s", target.unit)
            else:
                column = convert_units(
                    source_values[:, mapping.source_index], source.unit, target.unit
                )
            output[:, mapping.target_index] = column
        return output


# unit -> (physical dimension, canonical unit, multiplicative scale, offset)
_UNITS: dict[str, tuple[str, str, float, float]] = {
    "bit/s": ("throughput", "Mbit/s", 1e-6, 0.0),
    "kbit/s": ("throughput", "Mbit/s", 1e-3, 0.0),
    "Mbit/s": ("throughput", "Mbit/s", 1.0, 0.0),
    "Gbit/s": ("throughput", "Mbit/s", 1e3, 0.0),
    "ratio": ("ratio", "ratio", 1.0, 0.0),
    "%": ("ratio", "ratio", 1e-2, 0.0),
    "ms": ("time", "ms", 1.0, 0.0),
    "s": ("time", "ms", 1e3, 0.0),
    "count": ("count", "count", 1.0, 0.0),
    "dBm": ("log_power", "dBm", 1.0, 0.0),
    "dB": ("log_ratio", "dB", 1.0, 0.0),
    "bit": ("data", "bit", 1.0, 0.0),
    "kbit": ("data", "bit", 1e3, 0.0),
    "Mbit": ("data", "bit", 1e6, 0.0),
    "byte": ("data", "bit", 8.0, 0.0),
    "kB": ("data", "bit", 8e3, 0.0),
}

_AGGREGATIONS = {"gauge", "sum", "mean", "min", "max", "p50", "p95"}
_COUNTERS = {"gauge", "delta", "cumulative"}


@dataclass(frozen=True)
class KPMContract:
    """Machine-checkable semantics for one KPM stream."""

    name: str
    quantity: str
    unit: str
    entity_scope: str
    aggregation: str
    window_ms: int
    clock: str
    counter_semantics: str
    schema_version: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.name or not self.quantity or not self.entity_scope:
            raise ContractError("name, quantity, and entity_scope are required")
        if self.unit not in _UNITS:
            raise ContractError(f"unsupported unit: {self.unit}")
        if self.aggregation not in _AGGREGATIONS:
            raise ContractError(f"unsupported aggregation: {self.aggregation}")
        if self.counter_semantics not in _COUNTERS:
            raise ContractError(f"unsupported counter semantics: {self.counter_semantics}")
        if self.window_ms <= 0:
            raise ContractError("window_ms must be positive")

    @property
    def dimension(self) -> str:
        return _UNITS[self.unit][0]

    @property
    def canonical_unit(self) -> str:
        return _UNITS[self.unit][1]

    @property
    def effective_dimension(self) -> str:
        """Dimension after the declared counter transform is applied."""
        if self.counter_semantics in {"delta", "cumulative"} and self.dimension == "data":
            return "throughput"
        return self.dimension

    def type_compatible(self, target: "KPMContract") -> bool:
        return (
            self.quantity == target.quantity
            and self.effective_dimension == target.effective_dimension
            and self.entity_scope == target.entity_scope
        )

    def semantic_cost(self, target: "KPMContract") -> float:
        """Deterministic matching cost; incompatible types have infinite cost."""
        if not self.type_compatible(target):
            return float("inf")
        cost = 0.0
        cost += 0.4 * (self.aggregation != target.aggregation)
        cost += min(abs(np.log(self.window_ms / target.window_ms)), 4.0)
        cost += 0.3 * (self.clock != target.clock)
        cost += 0.5 * (self.counter_semantics != target.counter_semantics)
        return float(cost)


def convert_to_canonical(values: Iterable[float] | np.ndarray, unit: str) -> np.ndarray:
    """Convert values to the canonical unit associated with ``unit``."""
    if unit not in _UNITS:
        raise ContractError(f"unsupported unit: {unit}")
    _, _, scale, offset = _UNITS[unit]
    arr = np.asarray(values, dtype=float)
    return arr * scale + offset


def convert_units(
    values: Iterable[float] | np.ndarray,
    source_unit: str,
    target_unit: str,
) -> np.ndarray:
    """Convert values between units with the same physical dimension."""
    if source_unit not in _UNITS or target_unit not in _UNITS:
        raise ContractError("unsupported source or target unit")
    source_dimension, _, source_scale, source_offset = _UNITS[source_unit]
    target_dimension, _, target_scale, target_offset = _UNITS[target_unit]
    if source_dimension != target_dimension:
        raise ContractError("unit conversion changes physical dimension")
    array = np.asarray(values, dtype=float)
    base = array * source_scale + source_offset
    return (base - target_offset) / target_scale


def compile_contract_mapping(
    sources: list[KPMContract],
    targets: list[KPMContract],
    max_semantic_cost: float = 4.0,
) -> list[CompiledMapping]:
    """Compile a minimum-cost injective source-to-canonical mapping or fail closed."""
    if len(sources) < len(targets):
        raise ContractError("fewer source contracts than canonical targets")
    cost = np.array(
        [[source.semantic_cost(target) for target in targets] for source in sources],
        dtype=float,
    )
    if np.any(~np.isfinite(cost).any(axis=0)):
        raise ContractError("at least one canonical target has no type-compatible source")
    finite_cost = np.where(np.isfinite(cost), cost, 1e12)
    source_indices, target_indices = linear_sum_assignment(finite_cost)
    compiled: list[CompiledMapping] = []
    for source_index, target_index in zip(source_indices, target_indices, strict=True):
        value = cost[source_index, target_index]
        if not np.isfinite(value) or value > max_semantic_cost:
            raise ContractError("no mapping satisfies the semantic-cost budget")
        compiled.append(CompiledMapping(int(source_index), int(target_index), float(value)))
    if len(compiled) != len(targets):
        raise ContractError("canonical mapping is incomplete")
    return sorted(compiled, key=lambda item: item.target_index)


def compile_transform_plan(
    sources: list[KPMContract],
    targets: list[KPMContract],
    *,
    max_semantic_cost: float = 4.0,
    version: str = "kpm-bridge-plan-v1",
) -> CompiledTransformPlan:
    """Compile and cryptographically bind an executable transform plan."""
    mappings = tuple(
        compile_contract_mapping(sources, targets, max_semantic_cost=max_semantic_cost)
    )
    payload = {
        "version": version,
        "sources": [asdict(contract) for contract in sources],
        "targets": [asdict(contract) for contract in targets],
        "mappings": [asdict(mapping) for mapping in mappings],
        "max_semantic_cost": float(max_semantic_cost),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return CompiledTransformPlan(
        tuple(sources),
        tuple(targets),
        mappings,
        digest[:16],
        int.from_bytes(digest[16:24], "big"),
        version,
    )
