# KPM-Bridge

Reference implementation and reproducibility package for:

> **KPM-Bridge: An Uncertainty-Aware Cross-Vendor Telemetry Fabric for
> Portable AI xApps in Open 6G RAN**

KPM-Bridge separates E2 protocol interoperability from semantic and
statistical portability. It compiles heterogeneous KPM reports into typed
canonical contracts, aligns event-time observations, corrects residual shift
from paired anchors, emits fixed-size uncertainty certificates, and supports
selective xApp inference with explicit drift invalidation.

The manuscript targets the IEEE JSAC special issue **Towards Open and
Intelligent 6G RAN: Enabling Technologies and System Architectures**. The
official submission deadline is 5 August 2026.

## Evidence snapshot

- 108 hash-pinned public ColO-RAN files; 106 retained traces after the
  predeclared minimum-length rule.
- 201,912 labeled samples with independent experiment-level train/test splits.
- Eight canonical KPM features, four controlled implementation profiles,
  seven portable baselines, ablations, and five sensitivity sweeps.
- Stationary-profile mean: 0.685 NRMSE and 91.33% canonical-decision agreement.
- At `alpha=0.05`: 94.35% empirical joint coverage, 53.53% acceptance, and
  0.233% selective decision error.
- Under the severe P4 shift: 90.6% trace-level detection and 89.5% selective
  error reduction relative to ignoring the drift flag.
- Fixed certificate size: 48 bytes.

Profiles P1--P4 are deterministic stressors applied to public traces. They are
not measurements of, or claims about, named commercial vendors.

## Repository map

```text
src/kpm_bridge/          typed contracts, mappers, certificates, xApp gate
scripts/                 data fetch, benchmark, figures, tables, audits
tests/                   deterministic unit tests
data/                    upstream policy and hash manifest (raw data ignored)
reproducibility/outputs/ claim-generating CSV and audit records
manuscript/              IEEEtran source, references, vector figures
submission/              cover letter and author submission checklist
```

## Reproduce

Python 3.11 or newer and a LaTeX distribution containing `IEEEtran`, BibTeX,
and `latexmk` are required.

```bash
python3 -m pip install -e .
make fetch       # downloads the pinned public subset; raw files stay ignored
make benchmark   # regenerates the full deterministic CSV/JSON evidence
make assets      # regenerates tables, macros, and vector figures
make references # verifies DOI metadata through Crossref/DataCite
make paper       # builds the 13-page IEEE PDF
make audit       # runs tests plus 85 fail-closed claim/submission checks
```

For a fast installation check that is explicitly excluded from paper claims:

```bash
make smoke
```

## Data and licensing

The repository does not redistribute ColO-RAN traces. The downloader pins
upstream commit `bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b` and verifies every
file against `data/colosseum_subset_manifest.json`. See `data/README.md` for
the upstream citation and GPL-3.0 data terms.

The original KPM-Bridge software is released under the MIT License. The
manuscript and publication figures remain copyright of the authors unless a
publisher or release record states otherwise.

## Scope boundary

KPM-Bridge certifies telemetry compatibility and decision stability under the
stated assumptions. It does not prove that an xApp objective is safe,
authenticate compromised code, replace E2AP/E2SM-KPM, or solve multi-xApp
control conflicts. `SCOPE_LOCK.md` records the complete claim boundary.
