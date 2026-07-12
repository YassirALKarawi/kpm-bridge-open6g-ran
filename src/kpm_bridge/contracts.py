"""Typed KPM contracts and physically safe unit conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class ContractError(ValueError):
    """Raised when a KPM contract is internally inconsistent."""


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
    "byte": ("data", "byte", 1.0, 0.0),
    "kB": ("data", "byte", 1e3, 0.0),
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

    def type_compatible(self, target: "KPMContract") -> bool:
        return (
            self.quantity == target.quantity
            and self.dimension == target.dimension
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
