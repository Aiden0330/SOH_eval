# The adaptation operator seam and the calibration budget in capacity tests (K13, 2026-09-06)

Spec 12 §9 (soh-agent `docs/specs/field-data-and-inference.md`, P21) named what the kernel should grow after K13 so
that the evaluation ladder and the inference service share one implementation of the deployment-legal operators, and
how a field task counts its calibration budget. Both landed in `sohkernel.adaptation`; the workbench (`sohbench`,
frozen) and the public `InferenceAdapter` methods are unchanged.

## 1. The seam: `fit_operator` / `apply_operator`

```python
from sohkernel.adaptation import fit_operator, apply_operator

fitted = fit_operator(
    operator, target, calibration_mask, predict,          # predict(rows) -> ndarray: the frozen model, meta bound by the caller
    columns=model_columns, interval="conformal",           # "model" | "conformal"
    source_features=training_sample_matrix,               # CORAL's unlabeled channel (None without CORAL)
    correction_columns=None,                               # residual inputs; wide_residual_columns(...) for residual_wide
    strict=False,                                          # serving: guards become refusals; the ladder keeps strict=True
)
prediction, bounds, record = apply_operator(fitted, rows, predict, predict_interval)
```

- `target` holds every row of the unit (calibration and evaluation / query rows); `soh` is read on the calibration rows
  only; `cell_id` and `cycle_index` order the monotone projection. `apply_operator` accepts the same table, its
  evaluation rows, or new query segments at serving.
- `FittedOperator` carries the CORAL map, the labelled correction (`fit_recalibration` / `fit_residual` parameters),
  whether the monotone projection runs, the conformal band, `calibration_rows`, and the audit trail `applied` /
  `refused`; `record()` is the JSON-safe summary (`operator`, `channel`, `interval`, `calibration_rows`, `applied`,
  `refused`, `verdict`, `interval_source`, `fitted`). `verdict` is `applied` | `partial` | `refused` |
  `not_applicable` (operator `none` with the model's own band).
- `interval_source` names where the served band comes from: `model` (the frozen model's band on the adapted rows),
  `model_affine` (that band mapped through the affine recalibration, bounds swapped on a negative slope),
  `model_shifted` (shifted by the realised per-row correction: residual, or any operator with the monotone step),
  `calibration_conformal` (the split-conformal quantile of the calibration residuals, leave-one-out after a labelled
  step, in-sample otherwise); `apply_operator` reports `none` when the caller passes no `predict_interval` and no
  conformal band exists. The platform's `training_conformal` / `training_conformal_shifted` / `calibration_conformal`
  wording maps onto `model` / `model_affine` + `model_shifted` / `calibration_conformal`.
- Guards are the kernel's own and unchanged (registry-pinned): recalibration and residual need four finite calibration
  rows, CORAL eight finite rows on each side, a conformal band ten; the residual step falls back to intercept-only below
  `max(8, p + 2)` rows. `strict=True` raises them (the ladder: an operator that cannot be fitted is not a rung);
  `strict=False` records them in `refused` and the remaining steps still run — serving reports the refusal and never
  works around it (Spec 12 §9).
- `InferenceAdapter` is now a thin client of the seam: `prepare` fits CORAL from the service's training tables and
  draws the calibration split (unchanged), `recalibrate` calls `fit_operator(..., strict=True, skip_coral=True)` and
  `apply_operator`, `recalibrate_interval` uses the seam's interval rule. Return types are the same dicts as before, so
  `sohbench.evaluation_service` and `sohagent.adaptation.evaluate_operator` run unmodified.
  `tests/kernel/test_adaptation.py::test_seam_reproduces_the_adapter_for_every_operator` pins the equality for every
  operator and both interval modes.

Serving (the platform's `sohagent.inference.apply_operator`) becomes:

```python
fitted = fit_operator(operator, projected, calibration_mask, lambda t: model.predict(t, meta),
                      columns=model.feature_columns(), interval="conformal",
                      source_features=_matrix(training_sample, columns) if "coral" in operator else None, strict=False)
pred, bounds, block = apply_operator(fitted, projected, lambda t: model.predict(t, meta), lambda t: model.predict_interval(t, meta))
```

`block["refused"]` feeds the response's `adaptation.refused` and `warnings`; `block["verdict"]` and
`block["interval_source"]` replace the platform's own bookkeeping.

## 2. The calibration budget in capacity tests

`calibration_split(table, K, "earliest", unit="capacity_tests")` — also `capacity_test_split(table, K)` and
`InferenceAdapter(..., calibration_unit="capacity_tests")` — implements Spec 12 §5 for field segment tables, with the
rule the platform states for its own mask (data-onboarding session, 2026-09-06), so the two cannot drift:

- The calibration rows of a unit are its labelled segments whose label comes from one of the unit's first K reference
  tests (the K smallest distinct `label_time_s`). With forward-only labels these are the segments between the first
  test and the (K+1)-th, so every evaluation row of the unit starts after every calibration row: a forward cut by
  construction, hence only the `earliest` strategy applies (`spread` raises); the adapter records
  `calibration_scenario: forward_deployment` and `supports_forward_claim: true`. K = 2 uses both test values as labels
  (the M6 bar: "≤ 2 calibration tests").
- Segments before the first test (unlabelled) and segments labelled by later tests are the evaluation side
  (`evaluation = ~calibration`, as for the row budget); the QC drops unlabelled rows in supervised loads, and
  `capacity_test_summary` (merged into the adapter's `info`) reports `units`, `units_with_fewer_tests`,
  `calibration_labeled`, `evaluation_labeled` and `evaluation_unlabeled` next to `calibration_samples`.
- A unit with fewer than K tests puts all its labelled rows in calibration and contributes no labelled evaluation row.
- With one or two test values the calibration truth spans less than the affine guard's 0.05, so `recalibration`
  falls back to intercept-only by design; the `residual` step works on the feature variation across the segments.
- The kernel never chooses the mask for the caller: `fit_operator` takes any `calibration_mask`, so a platform that
  computes the rows itself passes them explicitly; the helper exists so the definition has one home. A per-cycle
  table (no `label_time_s`) is refused with the reason. `CALIBRATION_SPLITS` stays `("earliest", "spread")` — the
  unit is a separate axis, as in the platform's task schema (`calibration_unit: rows | capacity_tests`).

## 3. Reconciliation of the two segment recipes — for the platform's test

Spec 12 §9 asks that the kernel's `segments.segment_table` and the platform's serving-side `featurize_segment` agree
column by column. The kernel cannot import the platform (direction rule), so the test belongs in soh-agent; reading
both recipes side by side, these are the places to pin (the kernel recipe is the reference):

1. `charge_capacity_max`, `q_win_ah`, `t_win_s`, `ic_*` and the `hi_*` indicators: the kernel integrates every sample
   of the span (`accumulated_charge_ah` over the span, NaN currents as zero; bridged rests inside the span contribute
   their small currents) and hands `segment_curve`'s `(v, q, t_rel)` to `window_integral`, `ic_peak` and
   `hi_from_curve`; the platform integrates the charging samples only (above the threshold) and drops bridged rests.
2. `time_s_max`, `sample_period_s`, `v_start`, `v_end`: over the span in the kernel (its endpoints are charging samples
   by construction, interior rests count in the median period) versus over the charging subset in the platform.
3. `voltage_*`, `current_*`, `temperature_mean`, `temp_min`, `temp_max`, `current_cv`, `window_coverage`,
   `current_mean_c`: the same formulas on the charging samples in both.
4. The platform adds `cycle_index_feature` and `cycle_fraction` as NaN; the kernel's table has neither (the featuresets
   derive them) — the projection must behave the same with and without them.
5. `rest_before_s` / `rest_after_s`: the kernel measures them from the record; serving takes the caller's values.

A synthetic record with one bridged rest (under `rest_split_s`) inside a charge and one discharge after it exposes
items 1 and 2; where the two disagree the platform aligns `featurize_segment` to the kernel recipe.

## 4. Tests

`tests/kernel/test_adaptation.py`: the seam reproduces the adapter for every operator and interval mode; CORAL from a
training sample, and its refusal without one; refusals recorded when not strict, raised when strict; a fit applied to
new query rows with the conformal band; the capacity-test cut on a two-unit table (K = 2, K = 1, K = 0, the spread
refusal, the per-cycle refusal); the adapter counting the budget in tests. The platform registry pins no new
constant: `CALIBRATION_UNITS` and `INTERVAL_SOURCES` are names, not thresholds.
