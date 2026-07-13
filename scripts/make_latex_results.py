#!/usr/bin/env python3
"""Create LaTeX macros and tables directly from benchmark output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "reproducibility" / "outputs"
GENERATED = ROOT / "manuscript" / "generated"


def write(name: str, content: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / name
    temporary = GENERATED / f".{name}.tmp"
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def macros(main: pd.DataFrame, selective: pd.DataFrame, summary: dict) -> None:
    stable = main[(main.profile.isin(["P1", "P2", "P3"])) & ~main.method.isin(["Oracle canonical", "Deployment retrain"])]
    means = stable.groupby("method").mean(numeric_only=True)
    bridge = means.loc["KPM-Bridge"]
    bridge_profiles = stable[stable.method == "KPM-Bridge"]
    best_nrmse_baseline = means.drop(index="KPM-Bridge").nrmse.min()
    best_agreement_baseline = means.drop(index="KPM-Bridge").decision_agreement.max()
    stable_selective = selective[
        selective.profile.isin(["P1", "P2", "P3"]) & (selective.variant == "Full")
    ].mean(numeric_only=True)
    p4 = selective[selective.profile == "P4"].set_index("variant")
    error_reduction = 1.0 - p4.loc["Full", "selective_error"] / p4.loc["No drift gate", "selective_error"]
    total_rows = summary["training_rows"] + summary["test_rows"]
    lines = [
        f"\\newcommand{{\\DatasetRows}}{{{total_rows:,}}}",
        f"\\newcommand{{\\TrainingRows}}{{{summary['training_rows']:,}}}",
        f"\\newcommand{{\\TestRows}}{{{summary['test_rows']:,}}}",
        f"\\newcommand{{\\StableNrmse}}{{{bridge.nrmse:.3f}}}",
        f"\\newcommand{{\\StableAgreement}}{{{100 * bridge.decision_agreement:.2f}\\%}}",
        f"\\newcommand{{\\NrmseGainBest}}{{{100 * (best_nrmse_baseline - bridge.nrmse) / best_nrmse_baseline:.1f}\\%}}",
        f"\\newcommand{{\\AgreementGainBest}}{{{100 * (bridge.decision_agreement - best_agreement_baseline):.2f} percentage points}}",
        f"\\newcommand{{\\StableCoverage}}{{{100 * stable_selective.coverage_valid_regime:.2f}\\%}}",
        f"\\newcommand{{\\StableAcceptance}}{{{100 * stable_selective.acceptance:.2f}\\%}}",
        f"\\newcommand{{\\StableSelectiveError}}{{{100 * stable_selective.selective_error:.3f}\\%}}",
        f"\\newcommand{{\\DriftDetection}}{{{100 * p4.loc['Full', 'detection_rate']:.1f}\\%}}",
        f"\\newcommand{{\\DriftDelay}}{{{p4.loc['Full', 'median_detection_delay_ms'] / 1000:.2f}~s}}",
        f"\\newcommand{{\\DriftFalseAlarm}}{{{100 * p4.loc['Full', 'false_alarm_rate_per_trace']:.1f}\\%}}",
        f"\\newcommand{{\\DriftSelectiveError}}{{{100 * p4.loc['Full', 'selective_error']:.2f}\\%}}",
        f"\\newcommand{{\\NoDriftSelectiveError}}{{{100 * p4.loc['No drift gate', 'selective_error']:.1f}\\%}}",
        f"\\newcommand{{\\DriftErrorReduction}}{{{100 * error_reduction:.1f}\\%}}",
        f"\\newcommand{{\\BridgeLatency}}{{{bridge.inference_us_per_sample:.1f}~$\\mu$s/sample}}",
        f"\\newcommand{{\\BridgeLatencyRange}}{{{bridge_profiles.inference_us_per_sample.min():.1f}--{bridge_profiles.inference_us_per_sample.max():.1f}~$\\mu$s/sample}}",
        f"\\newcommand{{\\BridgeFitRange}}{{{bridge_profiles.fit_seconds.min():.2f}--{bridge_profiles.fit_seconds.max():.2f}~s}}",
        f"\\newcommand{{\\TemporalLatency}}{{{means.loc['Temporal ridge', 'inference_us_per_sample']:.2f}~$\\mu$s/sample}}",
        f"\\newcommand{{\\BridgeModelSize}}{{{bridge.model_bytes / 2**20:.2f}~MiB}}",
        f"\\newcommand{{\\CertificateBytes}}{{{summary['certificate_bytes']}~B}}",
        "\\newcommand{\\HundredUeCertificateRate}{0.384~Mbit/s}",
    ]
    write("results_macros.tex", "\n".join(lines))


def main_table(main: pd.DataFrame) -> None:
    order = ["Direct", "Contract", "Z-score", "CORAL", "Gaussian OT", "Anchor ridge", "Temporal ridge", "KPM-Bridge"]
    stable = main[(main.profile.isin(["P1", "P2", "P3"])) & main.method.isin(order)]
    means = stable.groupby("method").mean(numeric_only=True).reindex(order)
    labels = {"Gaussian OT": "Gaussian OT", "Anchor ridge": "Anchor ridge", "Temporal ridge": "Temporal ridge", "KPM-Bridge": "KPM-Bridge"}
    rows = []
    for method, row in means.iterrows():
        label = labels.get(method, method)
        values = f"{row.nrmse:.3f} & {100 * row.decision_agreement:.2f} & {row.regret:.4f} & {row.inference_us_per_sample:.2f}"
        if method == "KPM-Bridge":
            bold_values = values.replace(" & ", "} & \\textbf{")
            rows.append(f"\\textbf{{{label}}} & \\textbf{{{bold_values}}} \\\\")
        else:
            rows.append(f"{label} & {values} \\\\")
    content = r"""
\begin{table}[t]
\caption{Mean stationary reconstruction, decision, regret, and runtime performance across profiles P1--P3.}
\label{tab:main}
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrr}
\toprule
Method & NRMSE$\downarrow$ & Agree (\%)$\uparrow$ & Regret$\downarrow$ & $\mu$s/sample \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("table_main.tex", content)


def selective_table(selective: pd.DataFrame) -> None:
    full = selective[selective.variant == "Full"].set_index("profile").reindex(["P1", "P2", "P3", "P4"])
    rows = []
    for profile, row in full.iterrows():
        detection = "--" if np.isnan(row.detection_rate) else f"{100 * row.detection_rate:.1f}"
        delay = "--" if np.isnan(row.median_detection_delay_ms) else f"{row.median_detection_delay_ms / 1000:.2f}"
        rows.append(
            f"{profile} & {100 * row.coverage_valid_regime:.2f} & {100 * row.acceptance:.2f} & "
            f"{100 * row.selective_error:.3f} & {100 * row.selective_error_wilson_high:.3f} & "
            f"{detection} & {delay} \\\\"
        )
    content = r"""
\begin{table}[t]
\caption{Calibration coverage, selective inference, and drift-detection outcomes at $\alpha=0.05$.}
\label{tab:selective}
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lrrrrrr}
\toprule
Profile & Cov. (\%) & Accept (\%) & Sel. err. (\%) & Wilson hi. & Detect (\%) & Delay (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("table_selective.tex", content)


def ablation_table(ablation: pd.DataFrame) -> None:
    stable = ablation[ablation.profile.isin(["P1", "P2", "P3"])]
    means = stable.groupby("variant")[["nrmse", "decision_agreement"]].mean()
    order = ["No residual/temporal", "No temporal", "Linear residual", "No typed contract", "Full"]
    note = {
        "No residual/temporal": "contract normalization only",
        "No temporal": "current report only",
        "Linear residual": "ridge instead of trees",
        "No typed contract": "statistical fit; no type rejection",
        "Full": "contracts + temporal nonlinear residual",
    }
    rows = []
    for variant in order:
        row = means.loc[variant]
        label = f"\\textbf{{{variant}}}" if variant == "Full" else variant
        rows.append(
            f"{label} & {note[variant]} & {row.nrmse:.3f} & {100 * row.decision_agreement:.2f} \\\\"
        )
    content = r"""
\begin{table}[t]
\caption{Mean stationary ablation results across P1--P3, with the mechanism retained by each variant.}
\label{tab:ablation}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrr}
\toprule
Variant & Retained mechanism & NRMSE & Agree (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("table_ablation.tex", content)


def profile_table() -> None:
    content = r"""
\begin{table*}[t]
\caption{Controlled implementation profiles applied to public ColO-RAN traces; profile names denote stress settings rather than vendors.}
\label{tab:profiles}
\centering
\scriptsize
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{llllllll}
\toprule
Profile & Declared representation & Window & Lag & Entry loss & Noise & Quantization & Additional condition \\
\midrule
P1 & scaled units / gauges & 2 samples & 1 sample & 3\% & 2\% & 1\% & mild hidden filter \\
P2 & rates as resetting counters & 3 samples & 1 sample & 8\% & 3\% & 1.5\% & resets every 257 samples \\
P3 & scaled units / gauges & 4 samples & 2 samples & 18\% & 5\% & 3\% & sparse filtered reports \\
P4 & scaled units / gauges & 2 samples & 1 sample & 5\% & 2.5\% & 1.2\% & shift at 55\% of each test trace \\
\bottomrule
\end{tabular}
\end{table*}
"""
    write("table_profiles.tex", content)


def main() -> None:
    main_results = pd.read_csv(OUTPUTS / "main_results.csv")
    selective = pd.read_csv(OUTPUTS / "selective_results.csv")
    ablation = pd.read_csv(OUTPUTS / "ablation_results.csv")
    summary = json.loads((OUTPUTS / "benchmark_summary.json").read_text(encoding="utf-8"))
    macros(main_results, selective, summary)
    main_table(main_results)
    selective_table(selective)
    ablation_table(ablation)
    profile_table()
    print(json.dumps({"generated_files": 5, "output": str(GENERATED)}, indent=2))


if __name__ == "__main__":
    main()
