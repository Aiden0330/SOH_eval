# Phase 2 — Deep Learning & PINN (cycle-sequence, pytorch)

> **Date**: 2026-06-27
> **Status**: spec + TDD plan, ready for subagent execution once `torch` (CPU) is installed.
> **Extends**: merged sub-project A + Phase 1. Brings the comparison's **algorithm axis** into deep learning
> and physics-informed networks — goal-2's "cover DL / PINN methods".
> **Key reframing (vs the originally-sketched Phase 2)**: DL operates on **per-cell cycle-sequences of the
> existing feature vectors**, NOT on raw within-cycle V/I/T signals. This delivers genuine temporal DL
> (the degradation-trajectory dependency that distinguishes DL from per-cycle ML) and PINN, with **zero
> harness changes** and no heterogeneous raw-file parsing. Raw within-cycle-signal models (a true
> `RawChannel`) are **deferred** as an optional later extension — heavier data-engineering, lower marginal value.

## Decisions

| # | Decision | Value |
|---|---|---|
| D-1 | DL input | per-cell **windowed cycle-sequence** of existing feature vectors (group by `cell_id`, order by `cycle_index`). |
| D-2 | Backend | pytorch CPU; optional `[dl]` extra; import-guarded so the framework still runs without torch. |
| D-3 | Contract | DL/PINN estimators implement the standard `fit(X,y,meta)`/`predict(X,meta)` — **no runner/report/protocol changes**; they self-sequence internally and join the feature × algorithm matrix. |
| D-4 | Models | `LSTMEstimator`, `CNN1DEstimator`, `TransformerEstimator`, `PINNEstimator`. |
| D-5 | Uncertainty | none native → runner conformal/weighted-conformal wraps them (already in place). MC-dropout intervals optional later. |
| D-6 | Tractability | small nets, few epochs, early stopping, train subsample cap (like GPR/SVR) to keep matrix runs feasible on CPU. |
| D-7 | Output | clamp to physical `[0,1.3]` (reuse `SOH_LOW/HIGH`). |

## Windowing (the one non-obvious piece)

For each row (cycle *t* of a cell), the model input is the window of the last `W` cycle-feature-vectors of
**that same cell** `[t-W+1 .. t]`, left-padded (repeat the first cycle) for early cycles; the target is
`soh[t]`. So predictions stay **per-row** (matching the Estimator contract), windows never span cells (no
cross-cell leakage), and a test cell's window uses only its own causal history. `W` configurable (default 8).

`torch_base.py` provides: device/seed setup, a `build_windows(X, feature_cols, W)` that groups by
`(dataset, cell_id)`, orders by `cycle_index`, and returns `(N, W, F)` tensors aligned to X's rows; a
`StandardScaler` fit on train features; a generic train loop (Adam, MSE, early stopping on a validation
split carved from the train tail); a train-subsample cap; and the `[0,1.3]` clamp.

## Models

- **`LSTMEstimator`** (`name="lstm"`, backend `pytorch`): 1–2 layer LSTM over the window → last hidden →
  linear → SOH.
- **`CNN1DEstimator`** (`name="cnn1d"`): 1D conv stack over the cycle axis of the window → GAP → linear.
- **`TransformerEstimator`** (`name="transformer"`): small Transformer encoder (1–2 layers, 2–4 heads) over
  the window + positional encoding → pooled → linear.
- **`PINNEstimator`** (`name="pinn"`): MLP on the per-cycle feature vector with a **physics-informed loss**
  = data MSE + λ₁·monotonicity penalty (penalize predicted `soh[t+1] > soh[t] + rebound_tol` for consecutive
  cycles of a cell within the batch) + λ₂·range penalty (outside `[0,1.3]`). A genuine soft-constrained PINN;
  documents λ defaults. (This is the *true PINN* line, complementary to Phase-1's deployable physics-residual.)

All four set `feature_mode="unified"` so they run per featureset and appear in the matrix; all subsample
large training sets and clamp outputs.

## Module map

```
src/sohbench/estimators/
  torch_base.py     # NEW: device/seed, build_windows, scaler, train loop, subsample, clamp
  deep.py           # NEW: LSTMEstimator, CNN1DEstimator, TransformerEstimator
  pinn.py           # NEW: PINNEstimator (+ physics loss)
pyproject.toml      # MOD: [project.optional-dependencies] dl = ["torch"]
cli.py              # MOD: try/except import of deep & pinn (register when torch present; skip cleanly if not)
```

No changes to `runner.py`, `report.py`, `protocols.py`, `conformal.py`, `contracts.py`.

## TDD task breakdown (subagent batches)

**Batch 1 — torch infra + sequence models** (`torch_base.py`, `deep.py`):
- `torch_base`: `build_windows` groups by cell, orders by cycle, left-pads; assert shape `(len(X), W, F)`,
  that a window never mixes two cells (craft a 2-cell frame and check the boundary row's window is all
  cell-B), and that the subsample cap bounds training size.
- `deep`: each of LSTM/CNN1D/Transformer is an `Estimator`, fits a smooth synthetic per-cell trend with
  MAE < 0.12, predictions clamp to `[0,1.3]` on OOD input. Gate tests with `pytest.importorskip("torch")`.

**Batch 2 — PINN + integration** (`pinn.py`, `pyproject.toml`, `cli.py`):
- `pinn`: PINNEstimator fits; on a noisy-but-monotone synthetic series its predicted trajectory has a lower
  monotonicity-violation-rate than a plain-MSE MLP ablation (physics loss demonstrably helps); clamps OOD.
- `pyproject` `[dl]` extra; `cli` import-guarded registration (framework imports fine without torch; with
  torch, `lstm/cnn1d/transformer/pinn` resolve via `get_estimator`).
- Integration: a fast real-or-synthetic run including one DL estimator produces leaderboard entries with
  `backend="pytorch"` (conformal-wrapped `ci_source`), joining the feature × algorithm matrix.

## Tractability notes

DL estimators are far slower than the sklearn ones; benchmark runs that include them should use small
featureset/seed grids or the train-subsample cap. The unit tests use tiny synthetic data and few epochs.

## Out of scope (this phase)

- True `RawChannel` (within-cycle raw V/I/T parsing across NASA .mat / CALCE xlsx / Oxford .mat / HUST) and
  raw-signal CNNs — deferred; would bundle with the data-onboarding track.
- GPU; distributed training; hyperparameter search beyond sane defaults; MC-dropout/deep-ensemble intervals
  (conformal wrapping covers UQ for now).
