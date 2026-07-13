# Reproducibility protocol

This document separates fast software checks from the evidence used by the
KPM-Bridge manuscript. All random generators use fixed seeds, all upstream
files are pinned by commit and SHA-256 digest, and every numerical statement in
the manuscript is checked against machine-readable outputs.

## 1. Environment

- Linux or macOS with Python 3.11 or newer.
- A LaTeX distribution containing IEEEtran, BibTeX, and `latexmk` for the PDF.
- Network access only for the pinned dataset download and DOI metadata audit.

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[test]'
```

## 2. Fast deterministic checks

```bash
make test
make smoke
```

`make test` runs the contract, calibration, and certificate unit tests.
`make smoke` exercises the alignment pipeline on a small deterministic latent
process. Its output is marked `SMOKE_ONLY_NOT_A_PAPER_RESULT` and must not be
used to substantiate manuscript claims.

## 3. Obtain the benchmark traces

```bash
make fetch
```

The downloader obtains the public ColO-RAN subset pinned to commit
`bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b`. It checks all 108 byte counts and
SHA-256 hashes against `data/colosseum_subset_manifest.json`. Raw and processed
traces remain outside version control.

## 4. Regenerate the evidence

```bash
make benchmark
make assets
```

The benchmark writes its claim-bearing records to
`reproducibility/outputs/`. Table fragments and result figures are regenerated
from those records by `make assets`. The three architecture illustrations are
author-supplied publication assets and are not synthesized by the benchmark.

## 5. Verify references and build the manuscript

```bash
make references
make paper
make claims
```

The reference audit validates all DOI records through Crossref or DataCite.
The fail-closed claim audit checks dataset identity, train/test separation,
performance, sensitivity, drift, complexity, certificate arithmetic,
bibliography integrity, figure structure, and the 13-page submission layout.
A nonzero exit status indicates that the repository and manuscript are no
longer mutually consistent.

## Expected evidence identity

| Item | Expected value |
|---|---:|
| Upstream files | 108 |
| Retained traces | 106 |
| Training/test traces | 53 / 53 |
| Labeled rows | 201,912 |
| Canonical KPM features | 8 |
| Certificate size | 48 bytes |
| DOI-verified references | 40 |
| Claim/submission checks | 88 |
| Deterministic unit tests | 12 |

The complete interpretation boundary is recorded in `SCOPE_LOCK.md`.
