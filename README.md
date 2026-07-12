# KPM-Bridge

**KPM-Bridge: An Uncertainty-Aware Cross-Vendor Telemetry Fabric for Portable AI xApps in Open 6G RAN**

This is the development workspace for the second paper. The target is the IEEE
JSAC special issue on open and intelligent 6G RAN, with a 13-page IEEE
double-column submission.

## Scope

KPM-Bridge addresses a gap that remains after protocol-level interoperability:
two E2 nodes can emit syntactically valid E2SM-KPM reports while assigning
different operational meaning to names, units, entity scopes, aggregation
windows, clocks, counter resets, missing values, and vendor preprocessing. The
framework compiles those reports into typed canonical KPM contracts, aligns
asynchronous observations, attaches calibrated uncertainty, detects drift, and
admits or abstains portable xApp inference against an explicit quality budget.

KPM-Bridge is deliberately distinct from:

- **Q-ROC**, which audits action-order conflicts and commit decisions;
- **xDevSM**, which abstracts E2 service-model procedures and APIs;
- a conventional rename/unit adapter, which does not quantify semantic or
  statistical uncertainty.

## Current status

- Scientific scope and novelty boundary locked.
- Initial typed-contract, shift-benchmark, and conformal-calibration kernel.
- Manuscript v0.1 skeleton with the mathematical model and evaluation plan.
- No numerical paper claim is permitted until the full reproducible benchmark
  has been executed and audited.

## Quick checks

```bash
python3 -m pip install -e .
make test
make smoke
make paper
```

The future GitHub repository will be `kpm-bridge-open6g-ran`, kept private
during development and released as `v1.0-jsac-submission` after claim audit.
