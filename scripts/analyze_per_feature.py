#!/usr/bin/env python3
"""Produce per-KPM reconstruction diagnostics for the final manuscript."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kpm_bridge.dataset import (
    FEATURE_NAMES,
    attach_qos_risk_labels,
    load_colosseum_subset,
    robust_feature_stats,
)
from kpm_bridge.evaluation import build_pairs
from kpm_bridge.mappers import KPMBridgeMapper, TemporalRidgeMapper, fit_mask
from kpm_bridge.profiles import default_profiles


def main() -> None:
    traces, _ = attach_qos_risk_labels(load_colosseum_subset())
    training = [trace for trace in traces if trace.experiment == "exp1"]
    test = [trace for trace in traces if trace.experiment == "exp2"]
    stats = robust_feature_stats(training)
    rows: list[dict[str, object]] = []
    for key in ("P1", "P2", "P3"):
        profile = default_profiles()[key]
        pairs = build_pairs(training + test, profile, stats)
        train_pairs = [pair for pair in pairs if pair.trace.experiment == "exp1"]
        test_pairs = [pair for pair in pairs if pair.trace.experiment == "exp2"]
        masks = [fit_mask(len(pair.trace.values), 0.10) for pair in train_pairs]
        for mapper in (TemporalRidgeMapper(profile), KPMBridgeMapper(profile, stats)):
            mapper.fit(train_pairs, masks)
            prediction = np.vstack([mapper.predict(pair) for pair in test_pairs])
            target = np.vstack([pair.trace.values for pair in test_pairs])
            standard = (prediction - target) / stats.scale
            for index, feature in enumerate(FEATURE_NAMES):
                rows.append(
                    {
                        "profile": key,
                        "method": mapper.name,
                        "feature": feature,
                        "nrmse": float(np.sqrt(np.mean(standard[:, index] ** 2))),
                        "nmae": float(np.mean(np.abs(standard[:, index]))),
                    }
                )
    output = Path("reproducibility/outputs/per_feature_results.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
