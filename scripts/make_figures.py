#!/usr/bin/env python3
"""Generate publication-quality vector figures from audited benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "reproducibility" / "outputs"
FIGURES = ROOT / "manuscript" / "figures"

BLUE = "#00629B"
NAVY = "#16324F"
CYAN = "#4C9FBE"
GREEN = "#2A9D8F"
ORANGE = "#E07A3F"
PURPLE = "#7656A5"
RED = "#C14953"
GREY = "#6B7280"
LIGHT = "#EEF3F7"
DARK = "#1F2937"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.titlesize": 8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    png = FIGURES / f"{name}.png"
    pdf_tmp = FIGURES / f".{name}.pdf.tmp"
    png_tmp = FIGURES / f".{name}.png.tmp"
    pdf_metadata = {
        "Creator": "KPM-Bridge reproducibility pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf_tmp,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=pdf_metadata,
    )
    fig.savefig(png_tmp, format="png", bbox_inches="tight", pad_inches=0.025, dpi=300)
    pdf_tmp.replace(pdf)
    png_tmp.replace(png)
    plt.close(fig)


def polish(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply one compact IEEE-style visual grammar to every result axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D7E0E8", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.8, width=0.65, color=GREY)


def _box(ax, x, y, w, h, title, subtitle="", color=BLUE, fill="white", lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fill,
        edgecolor=color,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    title_size = 6.4 if len(title.replace("\n", "")) <= 15 else 5.8
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", color=DARK, weight="bold", fontsize=title_size, linespacing=0.9)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center", color=GREY, fontsize=5.1, linespacing=0.9)
    return patch


def _arrow(ax, start, end, color=GREY, style="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.0,
        linestyle=style,
        color=color,
        zorder=2,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Stylised E2 nodes: radio tower, edge server, and telemetry streams.
    tower_x = 0.045
    ax.plot([tower_x, tower_x - 0.018, tower_x + 0.018, tower_x], [0.80, 0.43, 0.43, 0.80], color=NAVY, lw=1.2)
    ax.plot([tower_x - 0.012, tower_x + 0.012], [0.60, 0.60], color=NAVY, lw=1.0)
    for radius in (0.025, 0.040):
        ax.add_patch(mpl.patches.Arc((tower_x, 0.77), radius * 2, radius * 1.3, theta1=20, theta2=160, color=CYAN, lw=1.0))
    ax.text(tower_x, 0.34, "Heterogeneous\nE2 nodes", ha="center", va="top", fontsize=6.5, weight="bold", color=DARK)
    xs = [0.12, 0.265, 0.41, 0.555, 0.70, 0.845]
    widths = [0.115] * 6
    titles = [
        ("E2\ndecode", "syntax + schema"),
        ("Contract\ncompiler", "type-safe match"),
        ("Event-time\naligner", "window / lag / reset"),
        ("Constrained\nmapper", "anchors + residual"),
        ("Certificate\ngate", "coverage / age / drift"),
        ("Portable\nAI xApp", "act or abstain"),
    ]
    colors = [CYAN, BLUE, BLUE, PURPLE, GREEN, NAVY]
    for x, w, (title, subtitle), color in zip(xs, widths, titles, colors, strict=True):
        _box(ax, x, 0.53, w, 0.25, title, subtitle, color=color, fill="white")
    _arrow(ax, (0.08, 0.655), (0.117, 0.655), BLUE)
    for left, w, right in zip(xs[:-1], widths[:-1], xs[1:], strict=True):
        _arrow(ax, (left + w, 0.655), (right - 0.003, 0.655), BLUE)

    # Evidence plane.
    ax.add_patch(FancyBboxPatch((0.14, 0.10), 0.70, 0.25, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=LIGHT, edgecolor="#C7D4DF", lw=0.8))
    ax.text(0.155, 0.315, "EVIDENCE PLANE", fontsize=6.0, color=BLUE, weight="bold", ha="left", va="center")
    evidence = [
        (0.17, "Contract\nregistry", "versioned hash"),
        (0.34, "Trusted\nanchors", "10% fit budget"),
        (0.51, "Split\ncalibration", "joint + xApp score"),
        (0.68, "Mean-shift\nmonitor", "5 anchors / 2.5 s"),
    ]
    for x, title, subtitle in evidence:
        _box(ax, x, 0.155, 0.13, 0.12, title, subtitle, color=GREY, fill="white", lw=0.75)
    for left, right in zip(evidence[:-1], evidence[1:], strict=True):
        _arrow(ax, (left[0] + 0.13, 0.215), (right[0] - 0.005, 0.215), GREY)
    for x0, x1 in ((0.235, 0.322), (0.405, 0.467), (0.575, 0.612), (0.745, 0.757)):
        _arrow(ax, (x0, 0.278), (x1, 0.525), GREY, "--")

    # Fail-closed outcomes.
    _box(ax, 0.845, 0.13, 0.13, 0.14, "Decision", "ALLOW / ABSTAIN", color=GREEN, fill="#F5FBF8")
    _arrow(ax, (0.925, 0.53), (0.92, 0.275), GREEN)
    ax.text(0.50, 0.95, "KPM-Bridge: semantic translation and evidence-bound portable inference", ha="center", va="center", color=NAVY, fontsize=8.5, weight="bold")
    save(fig, "fig_architecture")


def semantic_contract() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        (0.02, "P1", "kbit/s · 500 ms", "gauge · UE"),
        (0.02, "P2", "bytes · reset", "cumulative · UE"),
        (0.02, "P3", "filtered · 1 s", "18% missing"),
        (0.02, "P4", "schema stable", "statistics drift"),
    ]
    ys = [0.74, 0.53, 0.32, 0.11]
    for (_, key, line1, line2), y in zip(cards, ys, strict=True):
        _box(ax, 0.02, y, 0.18, 0.15, key, f"{line1}\n{line2}", color=CYAN, fill="white")
        _arrow(ax, (0.205, y + 0.075), (0.305, 0.50), CYAN)

    _box(ax, 0.31, 0.34, 0.18, 0.32, "Typed contract", "quantity · unit · scope\naggregation · window · clock\ncounter · provenance", color=BLUE, fill="#F7FAFC")
    _arrow(ax, (0.495, 0.50), (0.585, 0.50), BLUE)
    _box(ax, 0.59, 0.34, 0.18, 0.32, "Canonical state", "$\\hat{z}_t \\in \\mathbb{R}^{8}$\ncommon event time", color=PURPLE, fill="#FAF8FD")
    _arrow(ax, (0.775, 0.50), (0.855, 0.50), PURPLE)
    _box(ax, 0.86, 0.34, 0.12, 0.32, "48-B certificate", "schema · support\nage · radius · drift", color=GREEN, fill="#F5FBF8")
    ax.text(0.40, 0.78, "reject incompatible quantity/scope", ha="center", color=RED, fontsize=6.0)
    ax.text(0.40, 0.22, "compile required transforms", ha="center", color=GREEN, fontsize=6.0)
    ax.text(0.50, 0.93, "Protocol-valid reports can still denote different statistical objects", ha="center", color=NAVY, fontsize=8.3, weight="bold")
    save(fig, "fig_semantic_contract")


def main_performance(main: pd.DataFrame) -> None:
    methods = ["Direct", "Contract", "Z-score", "CORAL", "Gaussian OT", "Anchor ridge", "Temporal ridge", "KPM-Bridge"]
    stable = main[(main.profile.isin(["P1", "P2", "P3"])) & (main.method.isin(methods))]
    means = stable.groupby("method", sort=False)[["nrmse", "decision_agreement"]].mean().reindex(methods)
    y = np.arange(len(methods))
    colors = [BLUE if method == "KPM-Bridge" else PURPLE if "ridge" in method else "#AAB6C2" for method in methods]

    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    bars = ax.barh(y, means.nrmse, color=colors, height=0.62, edgecolor="white")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized RMSE")
    ax.set_xlim(0, 3.45)
    for bar, value in zip(bars, means.nrmse, strict=True):
        ax.text(min(value + 0.04, 3.35), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=5.8, color=DARK)
    polish(ax, "x")
    save(fig, "fig_stationary_nrmse")

    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    agreement = 100 * means.decision_agreement
    bars = ax.barh(y, agreement, color=colors, height=0.62, edgecolor="white")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Canonical-decision agreement (%)")
    ax.set_xlim(75, 93)
    for bar, value in zip(bars, agreement, strict=True):
        ax.text(min(value + 0.18, 92.6), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=5.8, color=DARK)
    polish(ax, "x")
    save(fig, "fig_stationary_agreement")


def selective_tradeoff(sensitivity: pd.DataFrame) -> None:
    data = sensitivity[sensitivity.sweep == "alpha"].sort_values("value")
    alpha = 100 * data.value.to_numpy()
    fig, ax = plt.subplots(figsize=(3.48, 2.30))
    ax.plot(alpha, 100 * data.coverage, "o-", color=BLUE, markerfacecolor="white", label="Joint coverage")
    ax.plot(alpha, 100 * (1 - data.abstention), "s-", color=GREEN, markerfacecolor="white", label="Accepted decisions")
    ax.plot(alpha, 100 * (1 - data.value), "--", color=GREY, label="Nominal $1-\\alpha$")
    ax.set_xlabel("Miscoverage level $\\alpha$ (%)")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(38, 102)
    ax.legend(frameon=False, loc="best", handlelength=2.2)
    polish(ax)
    save(fig, "fig_coverage_availability")

    fig, ax = plt.subplots(figsize=(3.48, 2.30))
    error = 100 * data.selective_error
    ax.plot(alpha, error, "o-", color=RED, markerfacecolor="white")
    ax.fill_between(alpha, 0, error, color=RED, alpha=0.10)
    ax.set_xlabel("Miscoverage level $\\alpha$ (%)")
    ax.set_ylabel("Selective decision error (%)")
    ax.set_ylim(0, max(2.1, 1.12 * error.max()))
    polish(ax)
    save(fig, "fig_selective_error")


def sensitivity_figure(sensitivity: pd.DataFrame) -> None:
    anchor = sensitivity[sensitivity.sweep == "anchor_fraction"].sort_values("value")
    fig, ax = plt.subplots(figsize=(3.48, 2.25))
    ax.plot(100 * anchor.value, anchor.nrmse, "o-", color=BLUE, markerfacecolor="white")
    ax.set_xlabel("Paired-anchor fraction (%)")
    ax.set_ylabel("Normalized RMSE")
    polish(ax)
    save(fig, "fig_anchor_sensitivity")

    missing = sensitivity[sensitivity.sweep == "missing_rate"].sort_values("value")
    fig, ax = plt.subplots(figsize=(3.48, 2.25))
    ax.plot(100 * missing.value, 100 * missing.decision_agreement, "o-", color=GREEN, markerfacecolor="white")
    ax.set_xlabel("Missing entries (%)")
    ax.set_ylabel("Decision agreement (%)")
    polish(ax)
    save(fig, "fig_missingness_sensitivity")

    lag = sensitivity[sensitivity.sweep == "lag_samples"].sort_values("value")
    fig, ax = plt.subplots(figsize=(3.48, 2.25))
    ax.plot(lag.value, 100 * lag.decision_agreement, "o-", color=PURPLE, markerfacecolor="white")
    ax.axvline(4, color=GREY, ls="--", lw=0.8, label="1.0-s additional lag")
    ax.set_xlabel("Extra lag (250-ms samples)")
    ax.set_ylabel("Decision agreement (%)")
    ax.legend(frameon=False, loc="lower left")
    polish(ax)
    save(fig, "fig_lag_sensitivity")


def drift_figure(selective: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    drift = sensitivity[sensitivity.sweep == "drift_magnitude"].sort_values("value")
    p4 = selective[(selective.profile == "P4")].copy()
    order = ["Full", "No uncertainty gate", "No drift gate"]
    p4 = p4.set_index("variant").reindex(order)
    fig, ax = plt.subplots(figsize=(3.48, 2.35))
    ax.plot(drift.value, 100 * drift.detection_rate, "o-", color=GREEN, markerfacecolor="white", label="Detection")
    ax.plot(drift.value, 100 * drift.post_drift_coverage, "s-", color=BLUE, markerfacecolor="white", label="Pre-calibrated coverage")
    ax.plot(drift.value, 100 * drift.false_alarm_rate, "--", color=RED, label="False alarm")
    ax.set_xlabel("Injected standardized mean shift")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, loc="best")
    polish(ax)
    save(fig, "fig_drift_sensitivity")

    fig, ax = plt.subplots(figsize=(3.48, 2.35))
    bars = ax.bar(np.arange(3), 100 * p4.selective_error, color=[GREEN, ORANGE, RED], width=0.60, edgecolor="white")
    ax.set_xticks(np.arange(3), ["Full", "No uncertainty", "No drift"], rotation=10, ha="right")
    ax.set_ylabel("Selective decision error (%)")
    for bar, value in zip(bars, 100 * p4.selective_error, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=6.4)
    ax.set_ylim(0, 57)
    polish(ax, "y")
    save(fig, "fig_drift_ablation")


def complexity(main: pd.DataFrame) -> None:
    methods = ["Direct", "Contract", "Z-score", "CORAL", "Gaussian OT", "Anchor ridge", "Temporal ridge", "KPM-Bridge"]
    stable = main[(main.profile.isin(["P1", "P2", "P3"])) & main.method.isin(methods)]
    aggregate = stable.groupby("method")[["inference_us_per_sample", "model_bytes"]].mean().reindex(methods)
    colors = [BLUE if method == "KPM-Bridge" else "#AAB6C2" for method in methods]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    ax.barh(y, aggregate.inference_us_per_sample, color=colors, height=0.62, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Amortized CPU time ($\\mu$s/sample)")
    polish(ax, "x")
    save(fig, "fig_runtime")

    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    ax.barh(y, aggregate.model_bytes / (1024**2), color=colors, height=0.62, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Serialized model size (MiB)")
    polish(ax, "x")
    save(fig, "fig_model_size")


def certificate_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    states = [
        (0.04, 0.58, "UNSEEN", "no schema"),
        (0.23, 0.58, "COMPILED", "typed plan"),
        (0.42, 0.58, "CALIBRATED", "sets bound"),
        (0.61, 0.58, "ACTIVE", "epoch valid"),
    ]
    for x, y, title, subtitle in states:
        _box(ax, x, y, 0.14, 0.22, title, subtitle, color=BLUE if title != "ACTIVE" else GREEN, fill="white")
    for left, right in zip(states[:-1], states[1:], strict=True):
        _arrow(ax, (left[0] + 0.14, 0.69), (right[0] - 0.006, 0.69), BLUE)
    ax.text(0.205, 0.76, "contract", ha="center", fontsize=5.5, color=GREY)
    ax.text(0.395, 0.76, "anchors", ha="center", fontsize=5.5, color=GREY)
    ax.text(0.585, 0.76, "activate", ha="center", fontsize=5.5, color=GREY)

    _box(ax, 0.79, 0.58, 0.17, 0.22, "DECISION", "ALLOW or ABSTAIN", color=NAVY, fill="#F7FAFC")
    _arrow(ax, (0.75, 0.69), (0.785, 0.69), GREEN)
    _box(ax, 0.51, 0.12, 0.18, 0.22, "INVALID", "fail closed + reason", color=RED, fill="#FFF7F7")
    _arrow(ax, (0.68, 0.57), (0.62, 0.345), RED)
    ax.text(0.73, 0.43, "drift / schema / age", ha="center", fontsize=5.5, color=RED)
    _arrow(ax, (0.51, 0.23), (0.31, 0.57), PURPLE)
    ax.text(0.39, 0.34, "recompile", ha="center", fontsize=5.5, color=PURPLE)
    _arrow(ax, (0.60, 0.34), (0.50, 0.57), GREEN)
    ax.text(0.54, 0.43, "recalibrate", ha="center", fontsize=5.5, color=GREEN)
    ax.text(0.50, 0.94, "Certificate lifecycle and fail-closed state transitions", ha="center", va="center", fontsize=8.3, weight="bold", color=NAVY)
    save(fig, "fig_certificate_lifecycle")


def feature_diagnostics(per_feature: pd.DataFrame) -> None:
    features = ["dl_brate", "dl_bler", "dl_mcs", "dl_snr", "rsrp", "ul_brate", "ul_bler", "ul_buff"]
    labels = ["DL rate", "DL BLER", "DL MCS", "DL SNR", "RSRP", "UL rate", "UL BLER", "UL buffer"]
    bridge = per_feature[per_feature.method == "KPM-Bridge"].pivot(index="feature", columns="profile", values="nrmse").reindex(features)
    ridge = per_feature[per_feature.method == "Temporal ridge"].groupby("feature").nrmse.mean().reindex(features)
    bridge_mean = per_feature[per_feature.method == "KPM-Bridge"].groupby("feature").nrmse.mean().reindex(features)
    improvement = 100 * (ridge - bridge_mean) / ridge
    fig, ax = plt.subplots(figsize=(3.48, 2.55))
    image = ax.imshow(bridge.to_numpy(), cmap="Blues", vmin=0, vmax=max(1.2, bridge.to_numpy().max()), aspect="auto")
    ax.set_xticks(range(3), bridge.columns)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Controlled implementation profile")
    for i in range(len(labels)):
        for j in range(3):
            value = bridge.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color="white" if value > 0.7 else DARK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    cbar.set_label("Normalized RMSE", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8)
    save(fig, "fig_feature_nrmse")

    fig, ax = plt.subplots(figsize=(3.48, 2.55))
    colors = [GREEN if value >= 0 else RED for value in improvement]
    y = np.arange(len(labels))
    bars = ax.barh(y, improvement, color=colors, height=0.62, edgecolor="white")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("NRMSE reduction vs. temporal ridge (%)")
    for bar, value in zip(bars, improvement, strict=True):
        label = f"{value:.2f}" if abs(value) < 0.1 else f"{value:.1f}"
        ax.text(value + 0.9, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=5.8)
    ax.set_xlim(-2.5, 52.0)
    polish(ax, "x")
    save(fig, "fig_feature_gain")


def main() -> None:
    configure()
    main_results = pd.read_csv(OUTPUTS / "main_results.csv")
    selective = pd.read_csv(OUTPUTS / "selective_results.csv")
    sensitivity = pd.read_csv(OUTPUTS / "sensitivity_results.csv")
    per_feature = pd.read_csv(OUTPUTS / "per_feature_results.csv")
    main_performance(main_results)
    selective_tradeoff(sensitivity)
    sensitivity_figure(sensitivity)
    drift_figure(selective, sensitivity)
    complexity(main_results)
    feature_diagnostics(per_feature)
    print(json.dumps({"figure_count": 13, "output": str(FIGURES)}, indent=2))


if __name__ == "__main__":
    main()
