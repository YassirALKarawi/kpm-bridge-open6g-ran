# KPM-Bridge scientific scope lock

## Research question

Can one AI xApp consume KPM telemetry from heterogeneous O-RAN implementations
without retraining per deployment, while retaining explicit finite-sample control
over semantic translation error and refusing unsafe low-support inputs?

## Core failure mode

E2SM-KPM compatibility does not by itself imply measurement equivalence.
Implementations may differ in:

1. quantity names and identifiers;
2. units and scaling;
3. cell, slice, bearer, or UE scope;
4. sum, mean, percentile, gauge, delta, or cumulative-counter semantics;
5. observation window and reporting period;
6. clock origin, delay, jitter, and staleness;
7. quantisation, censoring, missingness, and reset behaviour;
8. hidden implementation-side filtering and software-version drift.

An AI xApp trained on one implementation may therefore accept a syntactically
valid vector whose operational meaning has changed.

## Locked contribution

KPM-Bridge is a semantic and statistical telemetry fabric between E2
termination and portable xApps. It provides:

1. **Typed KPM contracts.** Each measurement is bound to quantity, physical
   dimension, unit, entity scope, aggregation, window, clock, counter/reset
   semantics, provenance, and schema version.
2. **Constrained semantic transport.** Dimensionally valid mappings combine
   metadata matching, temporal alignment, anchor consistency, monotonicity,
   and a learned residual map.
3. **Uncertainty certificates.** Implementation- and regime-bound conformal error
   sets travel with the canonical KPM vector, support mask, age, schema hash,
   and calibration epoch.
4. **Selective portable inference.** A portable xApp uses robust inference when
   the quality budget is satisfied and abstains, requests refinement, or falls
   back when it is not.
5. **Online drift handling.** Schema, clock, support, and residual drift
   invalidate or recalibrate mappings instead of silently changing semantics.

## Explicit non-claims

- It is not a replacement for E2AP, E2SM-KPM, xDevSM, or an E2 termination.
- It does not resolve multi-xApp control conflicts; that belongs to Q-ROC/CMF.
- It does not claim universal semantic identifiability without anchors.
- It does not claim real multi-vendor hardware validation until such captures
  are actually collected.
- Synthetic implementation transformations are controlled stress tests, not evidence
  of undocumented behaviour by named commercial vendors.

## Mathematical objects

For implementation \(v\), raw metric \(i\), and time \(t\), use the
typed contract

\[
c_{v,i}=(q,u,e,a,w,\tau,\rho,\nu),
\]

where \(q\) is the physical quantity, \(u\) the unit, \(e\) the entity scope,
\(a\) the aggregation, \(w\) the window, \(\tau\) the clock basis, \(\rho\)
the counter/reset semantics, and \(\nu\) the provenance/version tuple.

Let \(z_t\) be the canonical latent KPM state and \(y_{v,t}\) the observed
implementation report. A masked asynchronous observation model is

\[
y_{v,t}=m_{v,t}\odot\left[h_v(z_{\tau_v(t)})+\epsilon_{v,t}\right].
\]

KPM-Bridge returns \((\hat z_t,\mathcal U_t,\gamma_t)\): a canonical estimate,
an uncertainty set, and a quality certificate. A downstream action is admitted
only if support, age, drift, and radius constraints in \(\gamma_t\) are met.

## Implemented analytical guarantees

1. type soundness for physical dimensions and entity scopes;
2. translation-error decomposition into type, temporal, approximation, noise,
   and drift terms;
3. finite-sample conformal coverage under the stated calibration assumptions;
4. downstream utility/decision-stability bound for Lipschitz xApp objectives;
5. complexity and memory bounds for compilation, streaming alignment, and
   calibration.

## Locked benchmark protocol

### Evidence sources

- Colosseum ColO-RAN traces as a public measured-data backbone;
- deterministic controlled profiles P1--P4 for unit, window, lag, missingness,
  counter-reset, quantisation, nonlinear filtering, and drift stress;
- no controlled profile is assigned to or described as a commercial vendor.

### Baselines

1. no adaptation;
2. static rename and unit conversion;
3. z-score alignment;
4. CORAL feature alignment;
5. optimal-transport alignment;
6. paired-anchor ridge;
7. temporal paired-anchor ridge;
8. deployment-specific retraining and an oracle canonical mapper as bounds.

### Metrics

- canonical KPM NRMSE/MAE and temporal alignment error;
- conformal coverage, interval width, calibration error, and abstention rate;
- xApp decision agreement, utility regret, SLA violations, and unsafe-action
  rate;
- adaptation samples, latency, throughput, memory, and wire overhead;
- robustness to missingness, delay, drift magnitude, anchor fraction, and
  conformal miscoverage.

### Required ablations

- contracts only;
- contracts plus temporal alignment;
- alignment plus learned residual;
- uncertainty without drift adaptation;
- full KPM-Bridge.

## Final manuscript assets

- 2 complete algorithms;
- 5 propositions with explicit assumptions and proofs;
- 9 publication figures;
- 5 tables, including related work, profiles, selective inference, ablation,
  and complexity;
- sensitivity, ablation, scalability, calibration, and failure-case results;
- deterministic seeds, generated outputs, a 40-reference DOI audit, an
  89-check claim audit, and 12 deterministic tests.
