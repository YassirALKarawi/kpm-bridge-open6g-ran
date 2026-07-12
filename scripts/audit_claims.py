#!/usr/bin/env python3
"""Fail-closed audit of manuscript claims against generated benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from audit_references import entries, field


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "reproducibility" / "outputs"
MANUSCRIPT = ROOT / "manuscript" / "main.tex"


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def check(self, claim: str, computed: object, expected: object, passed: bool) -> None:
        self.rows.append(
            {
                "claim": claim,
                "computed": json.dumps(computed, ensure_ascii=False),
                "expected": json.dumps(expected, ensure_ascii=False),
                "status": "PASS" if passed else "FAIL",
            }
        )

    def equal(self, claim: str, computed: object, expected: object) -> None:
        self.check(claim, computed, expected, computed == expected)

    def rounded(self, claim: str, computed: object, expected: object, digits: int) -> None:
        actual = np.round(np.asarray(computed, dtype=float), digits).tolist()
        target = np.asarray(expected, dtype=float).tolist()
        self.check(claim, actual, target, actual == target)

    def close(self, claim: str, computed: float, expected: float, tolerance: float = 1e-9) -> None:
        self.check(claim, computed, expected, bool(abs(computed - expected) <= tolerance))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_abstract_word_count(tex: str) -> int:
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not abstract:
        return 0
    text = abstract.group(1)
    text = re.sub(
        r"\\(?:DatasetRows|StableNrmse|StableAgreement|NrmseGainBest|AgreementGainBest|"
        r"StableCoverage|StableSelectiveError|StableAcceptance|DriftDetection|DriftDelay|"
        r"DriftErrorReduction)",
        "VALUE",
        text,
    )
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[$\\{}~=]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def main() -> None:
    audit = Audit()
    main_results = pd.read_csv(OUTPUTS / "main_results.csv")
    selective = pd.read_csv(OUTPUTS / "selective_results.csv")
    features = pd.read_csv(OUTPUTS / "per_feature_results.csv")
    ablations = pd.read_csv(OUTPUTS / "ablation_results.csv")
    sensitivity = pd.read_csv(OUTPUTS / "sensitivity_results.csv")
    summary = json.loads((OUTPUTS / "benchmark_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "colosseum_subset_manifest.json").read_text(encoding="utf-8"))
    tex = MANUSCRIPT.read_text(encoding="utf-8")

    # Dataset and benchmark identity.
    audit.equal("benchmark status", summary["status"], "FULL_DETERMINISTIC_BENCHMARK")
    audit.equal("global seed", summary["seed"], 20260712)
    audit.equal("input traces", summary["input_traces"], 106)
    audit.equal("train/test traces", [summary["training_traces"], summary["test_traces"]], [53, 53])
    audit.equal("train/test rows", [summary["training_rows"], summary["test_rows"]], [109539, 92373])
    audit.equal("total labeled rows", summary["training_rows"] + summary["test_rows"], 201912)
    audit.equal("canonical features", summary["features"], 8)
    audit.equal("certificate bytes", summary["certificate_bytes"], 48)
    audit.equal("manifest file count", manifest["file_count"], 108)
    audit.equal("manifest bytes", manifest["total_bytes"], 19295185)
    audit.equal(
        "pinned upstream commit",
        manifest["upstream_commit"],
        "bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b",
    )
    verified_files = 0
    for record in manifest["files"]:
        path = ROOT / "data" / "raw" / "colosseum" / record["path"]
        if path.exists() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]:
            verified_files += 1
    audit.equal("hash-verified raw inputs", verified_files, 108)

    # Fixed downstream xApp instrument.
    xapp = summary["xapp"]
    audit.rounded("canonical AUROC", xapp["canonical_auroc"], 0.743, 3)
    audit.rounded("canonical Brier score", xapp["canonical_brier"], 0.209, 3)
    audit.rounded("test risk prevalence percent", 100 * xapp["test_risk_prevalence"], 46.9, 1)
    audit.close("registered action threshold", xapp["control_action_threshold"], 0.2)
    audit.rounded("canonical action rate percent", 100 * xapp["control_action_rate"], 91.0, 0)

    stable = main_results[main_results.profile.isin(["P1", "P2", "P3"])]
    means = stable.groupby("method").mean(numeric_only=True)
    bridge = means.loc["KPM-Bridge"]
    baselines = means.drop(index=[name for name in ["KPM-Bridge", "Oracle canonical", "Vendor retrain"] if name in means.index])
    best_nrmse = baselines.nrmse.min()
    best_agreement = baselines.decision_agreement.max()
    audit.rounded("stationary KPM-Bridge NRMSE", bridge.nrmse, 0.685, 3)
    audit.rounded("stationary KPM-Bridge agreement percent", 100 * bridge.decision_agreement, 91.33, 2)
    audit.rounded("NRMSE gain versus strongest baseline percent", 100 * (best_nrmse - bridge.nrmse) / best_nrmse, 4.5, 1)
    audit.rounded("agreement gain versus strongest baseline points", 100 * (bridge.decision_agreement - best_agreement), 0.96, 2)
    audit.rounded("KPM-Bridge mean regret", bridge.regret, 0.0980, 4)
    audit.rounded("CORAL mean regret", means.loc["CORAL", "regret"], 0.1013, 4)
    audit.rounded("temporal-ridge mean regret", means.loc["Temporal ridge", "regret"], 0.1029, 4)

    bridge_profiles = stable[stable.method == "KPM-Bridge"].set_index("profile").loc[["P1", "P2", "P3"]]
    audit.rounded("per-profile bridge NRMSE", bridge_profiles.nrmse, [0.665, 0.675, 0.715], 3)
    audit.rounded("per-profile NRMSE CI lower", bridge_profiles.nrmse_ci_low, [0.596, 0.604, 0.634], 3)
    audit.rounded("per-profile NRMSE CI upper", bridge_profiles.nrmse_ci_high, [0.742, 0.755, 0.813], 3)
    agreements = 100 * bridge_profiles.decision_agreement
    audit.check("decision agreement range", [agreements.min(), agreements.max()], [91.1, 91.6], round(agreements.min(), 1) == 91.1 and round(agreements.max(), 1) == 91.6)

    # Feature diagnostics.
    bridge_features = features[(features.method == "KPM-Bridge") & features.profile.isin(["P1", "P2", "P3"])]
    dl_rate = bridge_features[bridge_features.feature == "dl_brate"].nrmse
    rsrp = bridge_features[bridge_features.feature == "rsrp"].nrmse
    ul_buffer = bridge_features[bridge_features.feature == "ul_buff"].nrmse
    audit.rounded("DL-rate NRMSE range", [dl_rate.min(), dl_rate.max()], [0.16, 0.24], 2)
    audit.check("RSRP NRMSE below 0.08", float(rsrp.max()), 0.08, bool(rsrp.max() < 0.08))
    audit.rounded("UL-buffer NRMSE range", [ul_buffer.min(), ul_buffer.max()], [1.11, 1.16], 2)
    feature_means = features[features.profile.isin(["P1", "P2", "P3"])].groupby(["method", "feature"]).nrmse.mean()
    def feature_gain(name: str) -> float:
        return 1.0 - feature_means.loc[("KPM-Bridge", name)] / feature_means.loc[("Temporal ridge", name)]
    audit.rounded("DL-rate improvement percent", 100 * feature_gain("dl_brate"), 49.4, 1)
    audit.rounded("RSRP improvement percent", 100 * feature_gain("rsrp"), 42.7, 1)
    audit.check("DL-BLER change below 0.1 percent", 100 * abs(feature_gain("dl_bler")), 0.1, 100 * abs(feature_gain("dl_bler")) < 0.1)

    # Calibration, selective inference, and drift.
    stable_full = selective[(selective.profile.isin(["P1", "P2", "P3"])) & (selective.variant == "Full")]
    stable_selective = stable_full.mean(numeric_only=True)
    audit.rounded("stationary joint coverage percent", 100 * stable_selective.coverage_valid_regime, 94.35, 2)
    audit.rounded("stationary acceptance percent", 100 * stable_selective.acceptance, 53.53, 2)
    audit.rounded("stationary selective error percent", 100 * stable_selective.selective_error, 0.233, 3)
    audit.rounded("mean generic radius", stable_selective.radius, 3.289, 3)
    audit.rounded("per-profile Wilson upper percent", 100 * stable_full.set_index("profile").loc[["P1", "P2", "P3"], "selective_error_wilson_high"], [0.242, 0.326, 0.270], 3)
    no_uncertainty = selective[(selective.profile.isin(["P1", "P2", "P3"])) & (selective.variant == "No uncertainty gate")]
    audit.rounded("no-uncertainty selective-error range percent", [100 * no_uncertainty.selective_error.min(), 100 * no_uncertainty.selective_error.max()], [8.23, 8.76], 2)
    audit.rounded("no-uncertainty acceptance range percent", [100 * no_uncertainty.acceptance.min(), 100 * no_uncertainty.acceptance.max()], [79.4, 92.2], 1)

    p4 = selective[selective.profile == "P4"].set_index("variant")
    audit.rounded("P4 valid-regime coverage percent", 100 * p4.loc["Full", "coverage_valid_regime"], 94.47, 2)
    audit.rounded("P4 detection percent", 100 * p4.loc["Full", "detection_rate"], 90.6, 1)
    audit.rounded("P4 median detection seconds", p4.loc["Full", "median_detection_delay_ms"] / 1000, 8.96, 2)
    audit.rounded("P4 false-alarm percent", 100 * p4.loc["Full", "false_alarm_rate_per_trace"], 9.4, 1)
    audit.rounded("P4 post-shift acceptance percent", 100 * p4.loc["Full", "post_drift_acceptance"], 4.57, 2)
    audit.rounded("P4 full selective error percent", 100 * p4.loc["Full", "selective_error"], 5.39, 2)
    audit.rounded("P4 no-drift selective error percent", 100 * p4.loc["No drift gate", "selective_error"], 51.2, 1)
    audit.rounded("P4 no-uncertainty selective error percent", 100 * p4.loc["No uncertainty gate", "selective_error"], 10.75, 2)
    error_reduction = 1 - p4.loc["Full", "selective_error"] / p4.loc["No drift gate", "selective_error"]
    audit.rounded("P4 drift-gate error reduction percent", 100 * error_reduction, 89.5, 1)

    alpha = sensitivity[sensitivity.sweep == "alpha"].set_index("value")
    audit.rounded("alpha coverage endpoints percent", 100 * alpha.loc[[0.01, 0.2], "coverage"], [98.66, 79.48], 2)
    audit.rounded("alpha acceptance endpoints percent", 100 * (1 - alpha.loc[[0.01, 0.2], "abstention"]), [42.9, 57.5], 1)
    audit.rounded("alpha selective-error endpoints percent", 100 * alpha.loc[[0.01, 0.2], "selective_error"], [1.38, 1.90], 2)
    drift = sensitivity[sensitivity.sweep == "drift_magnitude"].set_index("value")
    audit.rounded("half-scale drift detection percent", 100 * drift.loc[0.5, "detection_rate"], 11.3, 1)
    audit.rounded("half-scale post-drift coverage percent", 100 * drift.loc[0.5, "post_drift_coverage"], 90.7, 1)
    audit.rounded("one-scale drift detection percent", 100 * drift.loc[1.0, "detection_rate"], 79.2, 1)

    # Ablation and sensitivity claims.
    stable_ablation = ablations[ablations.profile.isin(["P1", "P2", "P3"])].groupby("variant").mean(numeric_only=True)
    audit.rounded(
        "ablation NRMSE sequence",
        stable_ablation.loc[["No residual/temporal", "No temporal", "Linear residual", "Full"], "nrmse"],
        [3.210, 0.752, 0.717, 0.685],
        3,
    )
    audit.rounded("untyped aggregate NRMSE", stable_ablation.loc["No typed contract", "nrmse"], 0.685, 3)
    anchors = sensitivity[sensitivity.sweep == "anchor_fraction"].set_index("value")
    audit.rounded("anchor NRMSE at 1/5/20 percent", anchors.loc[[0.01, 0.05, 0.2], "nrmse"], [0.747, 0.714, 0.710], 3)
    missing = sensitivity[sensitivity.sweep == "missing_rate"].set_index("value")
    audit.rounded("missingness agreement endpoints percent", 100 * missing.loc[[0.0, 0.3], "decision_agreement"], [91.57, 89.71], 2)
    lag = sensitivity[sensitivity.sweep == "lag_samples"].set_index("value")
    audit.rounded("zero-lag agreement percent", 100 * lag.loc[0.0, "decision_agreement"], 91.45, 2)
    audit.check("six-sample age exceeds 1.5 s", lag.loc[6.0, "mean_age_ms"], 1500, bool(lag.loc[6.0, "mean_age_ms"] > 1500))

    # Complexity and payload arithmetic.
    audit.rounded("bridge fit-time range seconds", [bridge_profiles.fit_seconds.min(), bridge_profiles.fit_seconds.max()], [1.43, 1.84], 2)
    audit.rounded("bridge batch latency microseconds", bridge.inference_us_per_sample, 22.1, 1)
    audit.rounded("bridge model size MiB", bridge.model_bytes / 2**20, 1.44, 2)
    audit.rounded("temporal ridge latency microseconds", means.loc["Temporal ridge", "inference_us_per_sample"], 1.20, 2)
    audit.rounded("temporal ridge model KiB", means.loc["Temporal ridge", "model_bytes"] / 2**10, 12.1, 1)
    audit.close("certificate rate Mbit/s", 48 * 10 * 100 * 8 / 1e6, 0.384)
    audit.close("canonical-vector rate Mbit/s", 8 * 4 * 10 * 100 * 8 / 1e6, 0.256)
    audit.close("combined payload rate Mbit/s", (48 + 8 * 4) * 10 * 100 * 8 / 1e6, 0.640)

    # Reference, structure, and submission-format checks.
    reference_audit = pd.read_csv(OUTPUTS / "reference_audit.csv")
    audit.equal("reference records", len(reference_audit), 25)
    audit.equal("verified DOI records", int((reference_audit.status == "VERIFIED").sum()), 21)
    audit.equal("declared no-DOI records", int((reference_audit.status == "NO_DOI_DECLARED").sum()), 4)
    bibliography = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    bibliography_entries = entries(bibliography)
    doi_url_pairs = 0
    for _, entry in bibliography_entries:
        doi = field(entry, "doi")
        url = field(entry, "url")
        if doi and url and url.lower() == f"https://doi.org/{doi}".lower():
            doi_url_pairs += 1
    audit.equal("DOI links paired with DOI fields", doi_url_pairs, 21)
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    bib_keys = {key for key, _ in bibliography_entries}
    audit.equal("all citation keys resolve", sorted(cited - bib_keys), [])
    audit.equal("all bibliography records cited", sorted(bib_keys - cited), [])
    abstract_words = tex_abstract_word_count(tex)
    audit.check("abstract word limit", abstract_words, "75--200", 75 <= abstract_words <= 200)
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex, re.S)
    keyword_count = len(keywords.group(1).split(",")) if keywords else 0
    audit.check("keyword limit", keyword_count, "at most 5", keyword_count <= 5)
    audit.check("exact Related Work heading", "Related Work" in tex, True, "\\section{Related Work}" in tex)
    audit.equal("algorithm count", len(re.findall(r"\\begin\{algorithm\}", tex)), 2)
    audit.equal("proposition count", len(re.findall(r"\\begin\{proposition\}", tex)), 5)
    forbidden = re.findall(r"\bChatGPT\b|\bOpenAI\b|\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|\bACKNOWLEDGMENT\b", tex, flags=re.I)
    audit.equal("no drafting disclosure or placeholders", forbidden, [])
    repo_mentions = re.findall(r"github\.com/YassirALKarawi/[^}\s]+", tex, flags=re.I)
    audit.equal("no unverified project repository URL in manuscript", repo_mentions, [])
    pdfinfo = subprocess.run(
        ["pdfinfo", str(ROOT / "manuscript" / "main.pdf")], check=True, capture_output=True, text=True
    ).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M).group(1))
    audit.equal("submission PDF pages", pages, 13)
    log = (ROOT / "manuscript" / "main.log").read_text(encoding="utf-8", errors="replace")
    fatal_layout = re.findall(r"Overfull|undefined references|Citation .* undefined|multiply defined", log, flags=re.I)
    audit.equal("no overflow or unresolved cross-reference", fatal_layout, [])

    frame = pd.DataFrame(audit.rows)
    output = OUTPUTS / "claim_audit.csv"
    frame.to_csv(output, index=False)
    failed = frame[frame.status == "FAIL"]
    print(
        json.dumps(
            {
                "checks": len(frame),
                "passed": int((frame.status == "PASS").sum()),
                "failed": len(failed),
                "output": str(output.relative_to(ROOT)),
                "failed_claims": failed.claim.tolist(),
            },
            indent=2,
        )
    )
    if not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
