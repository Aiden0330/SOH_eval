# Comparison Expansion — Phase 1: feature × algorithm matrix

> **Date**: 2026-06-26
> **Status**: spec + TDD plan, ready to implement (extends merged sub-project A).
> **Decision**: user expanded goal-2 comparison beyond the 4 traditional-ML models to also cover **feature
> sets** and **algorithm families incl. deep learning / PINN**, phased (user-approved). UI basis = mockup A.
> **Extends**: [sub-project A spec](2026-06-26-subproject-A-benchmark-harness-spec.md). A is already
> algorithm-agnostic (`backend ∈ {sklearn,pytorch,external}`, `feature_mode ∈ {unified,byo}`), so this is an
> extension, not a rewrite. Non-DNN/C++ constraint applies only to our deployable algorithms (sub-project B),
> never to comparison entries.

## 0. Two phases

- **Phase 1 (this spec)** — on the existing handcrafted feature table, no raw parsing:
  - a **FeatureSet dimension** (named feature representations) → benchmark becomes `featureset × algorithm × protocol`.
  - **MLP** estimator (neural network on features) — implemented with `sklearn.neural_network.MLPRegressor`
    (a genuine NN, zero heavy deps; pytorch reserved for Phase 2 sequence models).
  - **PhysicsResidual** estimator — semi-empirical decay prior (`1 − a·√throughput`) + ML residual; the
    deployable "physics-informed" entry (md's physics-residual). True PINN is Phase 2.
- **Phase 2 (later spec)** — `RawChannel` raw per-cycle V/I/T parsing; sequence DL (CNN / LSTM / Transformer,
  pytorch); genuine PINN (NN + physics-constrained loss); `raw-sequence` becomes a FeatureSet.

## 1. Phase-1 decisions

| # | Decision | Value |
|---|---|---|
| P1-1 | Comparison axes | `featureset × algorithm × protocol` (+ seed). |
| P1-2 | Phase-1 feature sets (from existing 16 handcrafted cols) | `full`, `voltage`, `ic_dva`, `partial_window`, `capacity_time` — map to md L1 HI families. |
| P1-3 | MLP backend | `sklearn` MLPRegressor (NN on features). pytorch deferred to Phase 2. |
| P1-4 | Physics-informed entry | PhysicsResidual = √-throughput prior + ML (GBDT) residual; labeled physics-informed (not "PINN"). |
| P1-5 | Backward compatibility | `featuresets=None` ⇒ single `full` set; existing A tests unchanged. |
| P1-6 | Output clamp | all new estimators clamp to physical SOH `[0,1.3]` (reuse baseline bound). |

## 2. Feature sets (P1-2)

`FeatureSet`: a named selector returning which feature columns to expose; the runner projects the table to
`ID_COLUMNS + selected` before handing it to estimators. Defined over the real columns
(`voltage_*`, `current_*`, `ic_peak_*`, `q_win_ah`, `t_win_s`, `charge_capacity_max`, `cycle_*`, …):

| name | columns (selector) | HI family (md L1) |
|---|---|---|
| `full` | all feature columns | — |
| `voltage` | `voltage_mean/std/min/max` | voltage-window |
| `ic_dva` | `ic_peak_ah_per_v`, `ic_peak_voltage_v` | ICA/DVA (LLI/LAM) |
| `partial_window` | `q_win_ah`, `t_win_s` | partial charge window |
| `capacity_time` | `charge_capacity_max`, `time_s_max`, `cycle_index_feature`, `cycle_fraction` | throughput/time |

Selectors take the **intersection** with present columns (a set may be empty for a dataset that lacks them →
skipped with a warning). Registered via `@register_featureset`.

## 3. Phase-1 estimators

- **`MLPEstimator`** (`name="mlp"`, backend `sklearn`): `StandardScaler` → `MLPRegressor(hidden_layer_sizes=
  (64,32), early_stopping=True, max_iter=300)`; clamp `[0,1.3]`; no native interval → runner conformal-wraps.
- **`PhysicsResidualEstimator`** (`name="phys_resid"`, backend `sklearn`): fit prior
  `soh ≈ 1 − a·√t − b·t` (least squares on a throughput proxy = `cycle_index_feature`, falling back to
  `cycle_fraction`), then fit a `HistGradientBoostingRegressor` on the residual `soh − prior` using the
  feature set; predict `prior + residual`, clamp `[0,1.3]`. Documents the throughput proxy used.

## 4. Module changes

```
src/sohbench/
  featuresets.py        # NEW: FeatureSet registry + the 5 named sets + project(table, name)
  estimators/neural.py  # NEW: MLPEstimator
  estimators/physics.py # NEW: PhysicsResidualEstimator
  runner.py             # MOD: run(..., featuresets=None); loop featureset, project table, tag records
  report.py             # MOD: group by (estimator, featureset, protocol); entry carries `featureset`
  cli.py                # MOD: --featuresets names; default ["full"]; register new estimators
  contracts.py          # (unchanged) Estimator protocol already fits
```

Result schema gains a `featureset` column (`results.parquet`) and `leaderboard.json` entries gain
`"featureset"`. Backward compatible: absent ⇒ `"full"`.

## 5. Frontend impact (mockup A)

The leaderboard gains a **featureset selector/column**; the chosen mockup A's table extends to
`algorithm × featureset`. The generalization-gap and PICP panels can be filtered by featureset. Schema change
is additive; the JSON Schema gains an optional `featureset` field.

## 6. TDD task breakdown (for subagent execution)

- **T1 `featuresets.py`**: registry + 5 sets; `project(table, name)` returns id+selected cols; empty-set
  warning; tests assert column membership and intersection behavior.
- **T2 `estimators/physics.py`**: PhysicsResidual; test it beats a flat-prior baseline on a synthetic
  √-decay series; clamps OOD.
- **T3 `estimators/neural.py`**: MLPEstimator; test is-Estimator, fits/predicts in range, clamps OOD.
- **T4 `runner.py` featureset loop**: test matrix expands over featuresets; records carry `featureset`;
  `featuresets=None` still works (backward compat).
- **T5 `report.py`**: group includes featureset; leaderboard entries carry it; backward compat when absent.
- **T6 `cli.py` + e2e**: `--featuresets full voltage ic_dva`; real-data run produces a feature×algorithm
  leaderboard; assert multiple featuresets present.

## 7. Out of scope (Phase 1)

Raw per-cycle sequence parsing (`RawChannel`), pytorch sequence DL (CNN/LSTM/Transformer), genuine PINN,
weighted/adaptive conformal — all Phase 2 or the separate hardening track.
