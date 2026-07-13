#!/usr/bin/env python3
"""Run a deterministic non-claim smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kpm_bridge.calibration import split_conformal_radius
from kpm_bridge.shiftbench import (
    ImplementationProfile,
    apply_affine_bridge,
    fit_affine_bridge,
    latent_ran_process,
    observe_implementation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical = latent_ran_process(2400, seed=20260712)
    profile = ImplementationProfile(
        scale=np.array([1000.0, 100.0, 100.0, 1.0, 1.0, 0.001]),
        bias=np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.0]),
        noise_std=np.array([120.0, 0.7, 0.5, 0.7, 0.3, 0.0007]),
        window=5,
        lag=2,
        missing_rate=0.03,
        quantisation=0.01,
    )
    raw, mask = observe_implementation(canonical, profile, seed=23)
    slopes, offsets = fit_affine_bridge(raw[:600], canonical[:600])
    estimate = apply_affine_bridge(raw, slopes, offsets)

    scale = np.nanstd(canonical[:600], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    residual = np.sqrt(np.nanmean(((estimate - canonical) / scale) ** 2, axis=1))
    radius = split_conformal_radius(residual[600:1200], alpha=0.05)
    test = residual[1200:]
    coverage = float(np.mean(test[np.isfinite(test)] <= radius))

    summary = {
        "status": "SMOKE_ONLY_NOT_A_PAPER_RESULT",
        "seed": 20260712,
        "steps": int(canonical.shape[0]),
        "dimensions": int(canonical.shape[1]),
        "observed_support": float(mask.mean()),
        "conformal_alpha": 0.05,
        "calibrated_radius": float(radius),
        "test_coverage": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
