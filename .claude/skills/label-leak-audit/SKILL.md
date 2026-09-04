---
name: label-leak-audit
description: Design and enforce label-leakage defenses in a supervised estimation pipeline (SOH, RUL, capacity, anomaly scores). Use when adding datasets, features, feature transforms, or label definitions to any benchmark/training tool, when results look "too good" (near-zero error), or when building a new estimation tool that consumes measured labels.
---

# Label-leak audit for supervised estimation pipelines

Near-perfect validation scores are the *signature* of leakage, not of skill.
This skill encodes the defense architecture proven in sohbench (label-safety
v1.1/§6): the leak was `soh := calibration_capacity / rated_capacity` with
`charge_capacity_max` (the calibration capacity itself) offered as a feature
— MAE 0.0003 "results" that were the label read back off the input.

## 1. Identify the label recipe first

Write down exactly how the label is computed from measurements (the "label
anchor"). Every quantity in that recipe — and everything affinely or
functionally derived from it — is a **label source**, banned as model input
on every dataset, regardless of correlation statistics.

## 2. Layer the defenses (all four, not one)

1. **Structural ban (static)** — a central deny-set of label-source column
   names + derived columns, enforced in the feature *selector itself* so no
   featureset (including "all columns" selectors and BYO/self-selecting
   estimators) can ever pick them. Strip them from the table BYO code sees.
2. **Value probes (dynamic)** — catch leaks arriving under other names:
   - exact copy / exact complement checks;
   - **affine reconstruction**: fit `label ~ a*x + b` per dataset group;
     R² ≥ 0.995 with n ≥ 64 ⇒ violation (legit physics tops out ~0.98);
   - **integral-coverage rules** for accumulation features: a "partial"
     integral covering ≥ 90% of the full quantity IS the label source even
     when spread keeps affine R² low (median per dataset; warn > 60%).
3. **Enforcement at every boundary** — ingestion scan (record findings on
   the dataset, don't reject it), training admission gate, transform audit
   behind a label-isolation boundary (transform never receives the label
   column), and trusted-inference gate on the model's manifest columns.
4. **Standing inventory test** — a pinned per-dataset expectation table run
   against the real data. It fails in BOTH directions: a new finding (new
   leak) and a disappeared finding (detection regression). Every edit to
   the pins requires a dated review comment — the test file is the audit
   trail.

## 3. When a leak is confirmed

- Block the *combination*, not the dataset (carrying a label source is
  legitimate; using it as input is not).
- Sweep and mark historic artifacts trained on the leaking inputs as
  `invalidated` — leaderboard entries included. Changed feature semantics
  (e.g. a rebuilt window) also invalidate old artifacts: same sweep.
- Error messages must name the column, group, statistic, and remedy — they
  double as the UI explanation.

## 4. Search/AutoML interaction (critical)

Any optimizer/agent selecting features or windows will *prefer* leaky
configurations (they score best). Leak gates must be hard constraints the
search cannot trade against — run the audit as an admission oracle before
scoring, never as a penalty term.


## Review note (2026-09-03, after the kernel extraction)

Module map since ADR-001 Wave 2: label safety, protocols (now including `forward`), the runner, report statistics, adaptation, fold-local feature pipelines and the API facade live in `sohkernel` (`sohkernel.api`, `sohkernel.feature_safety`, `sohkernel.inference_gate`, `sohkernel.pipelines`, `sohkernel.records`, `sohkernel.datasheets`); the candidate model, search engine, sandbox, evaluator client and claim flow live in `sohagent` (`sohagent.candidates`, `sohagent.search`, `sohagent.sandbox`, `sohagent.evaluator`, `sohagent.claims`). `sohbench` is the frozen workbench and only re-exports the kernel. Any `sohbench.<module>` reference above is historical.
