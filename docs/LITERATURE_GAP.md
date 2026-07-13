# Literature-gap lock (12 July 2026)

## Closest prior work

The closest current work is xDevSM:

- A. Feraudo *et al.*, “xDevSM: An Open-Source Framework for Portable,
  AI-Ready xApps Across Heterogeneous O-RAN Deployments,” arXiv:2602.03821,
  2026. <https://doi.org/10.48550/arXiv.2602.03821>
- Framework repository: <https://github.com/wineslab/xDevSM>

xDevSM provides high-level APIs over E2 service models and demonstrates the
same xApp logic across heterogeneous OAI/srsRAN-style deployments. It is a
required baseline and architectural complement, not a weak comparison target.

## Defensible separation

| Question | xDevSM | KPM-Bridge |
|---|---|---|
| Can the xApp decode and issue heterogeneous E2 procedures? | Primary scope | Assumed from the E2 framework |
| Are two decoded KPM streams semantically equivalent? | Not formally certified | Typed contract and constrained mapping |
| Are windows, clocks, counters, and scopes aligned? | API/data pipeline abstraction | Explicit temporal and semantic model |
| Is translation error quantified at finite sample size? | Not a stated contribution | Conformal uncertainty certificate |
| What happens under low support or drift? | Framework-dependent | Fail-closed abstention/refinement |
| Is downstream AI decision stability bounded? | Not a stated contribution | Planned utility/action-margin bound |

The manuscript must not claim that xDevSM fails to support portability. The
claim is narrower: protocol and API portability are necessary but do not by
themselves certify semantic/statistical equivalence of AI inputs.

## Evidence backbone

- ColO-RAN public traces:
  <https://github.com/wineslab/colosseum-oran-coloran-dataset>
- ns-O-RAN-flexRIC, which documents E2AP and KPM-v3 support:
  <https://github.com/Orange-OpenSource/ns-O-RAN-flexric>
- ns-O-RAN Gym:
  <https://github.com/wineslab/ns-o-ran-gym>

The ColO-RAN corpus is large; the development repository will store manifests,
checksums, and deterministic download/subset scripts rather than committing the
full corpus. A Zenodo record will be used for any derived release data too large
for GitHub.

## Review risks to address from the first draft

1. **Synthetic-vendor criticism:** call them implementation profiles and tie
   every transformation to a documented measurement semantic; do not attribute
   undocumented behaviour to commercial vendors.
2. **Identifiability:** state anchor and observability assumptions and expose
   partial support instead of claiming universal recovery.
3. **Conformal misuse under drift:** distinguish exchangeable split conformal,
   weighted covariate-shift calibration, and online drift invalidation.
4. **Downstream relevance:** report both translation metrics and closed-loop
   xApp utility/SLA/unsafe-action outcomes.
5. **Framework overhead:** include latency, memory, wire metadata, and scaling
   with implementation count and KPM dimension.
