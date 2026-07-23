# Figure and evidence index

The PNG files in `manuscript/figures/` are web previews for the repository front
page. They are direct rasterizations of the committed publication artwork in the
same directory; the PDF (result figures) and JPEG (architecture illustrations)
files remain the authoritative artwork used by the manuscript.

The table below lists the fifteen figures included in the 13-page manuscript:
three architecture illustrations and twelve result figures. Every result figure
is produced by `scripts/make_figures.py` from the tracked audit records in
`reproducibility/outputs/`; none is drawn by hand.

| Publication figure | Kind | Computational source | Primary numerical evidence |
|---|---|---|---|
| `manuscript/figures/fig1_kpm_bridge_architecture.jpeg` | Architecture | `scripts/make_figures.py` | Architectural specification; no numerical result |
| `manuscript/figures/fig2_typed_canonicalization.jpeg` | Architecture | `scripts/make_figures.py` | Typed-canonicalization schematic; no numerical result |
| `manuscript/figures/fig3_certificate_lifecycle.jpeg` | Architecture | `scripts/make_figures.py` | Certificate-lifecycle schematic; no numerical result |
| `manuscript/figures/fig_stationary_nrmse.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/main_results.csv` |
| `manuscript/figures/fig_stationary_agreement.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/main_results.csv` |
| `manuscript/figures/fig_feature_nrmse.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/per_feature_results.csv` |
| `manuscript/figures/fig_feature_gain.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/per_feature_results.csv` |
| `manuscript/figures/fig_coverage_availability.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/selective_results.csv` |
| `manuscript/figures/fig_selective_error.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/selective_results.csv` |
| `manuscript/figures/fig_drift_sensitivity.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/sensitivity_results.csv` |
| `manuscript/figures/fig_drift_ablation.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/main_results.csv` |
| `manuscript/figures/fig_anchor_sensitivity.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/sensitivity_results.csv` |
| `manuscript/figures/fig_missingness_sensitivity.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/sensitivity_results.csv` |
| `manuscript/figures/fig_lag_sensitivity.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/sensitivity_results.csv` |
| `manuscript/figures/fig_runtime.pdf` | Result | `scripts/make_figures.py` | `reproducibility/outputs/main_results.csv` |

The directory also retains `fig_model_size.pdf`/`.png`, a supplemental
diagnostic plot emitted by the full reproduction run; it is an evidence asset
and is not one of the fifteen manuscript figures.

## Regeneration

Regenerate the result figures (and their PNG previews) from the tracked outputs
with:

```bash
make figures
```

The target runs `scripts/make_figures.py`, which reads the CSV records in
`reproducibility/outputs/` and writes both the PDF artwork and the PNG previews
into `manuscript/figures/`.
