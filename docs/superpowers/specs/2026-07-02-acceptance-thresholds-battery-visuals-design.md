# Acceptance Thresholds And Battery Visuals Design

## Goal

Make the training and inference scorecards explain their acceptance status instead of showing a bare pass count, and prepare the dashboard to absorb algorithm details, SOH curves, and feature visualizations from `D:\battery_SOH`.

## Scope

- Add a front-end acceptance detail drawer opened from the KPI area.
- Add threshold presets: `strict`, `engineering`, and `exploratory`.
- Recompute acceptance counts client-side from existing metrics for the selected preset.
- Keep inference separate from training: inference shows success records and speed/transfer metrics, not training verdicts.
- Add a documented data shape for imported `battery_SOH` algorithm and visualization assets, but do not build a new importer in this slice unless the data already exists in `frontend/app/data/profiles`.

## Acceptance Presets

- `strict`: current benchmark contract, `mae <= 0.02`, `picp >= 0.90`, `monotonicity_violation_rate <= 0.001`, `value_range_violation <= 0`.
- `engineering`: practical cross-dataset SOH screening, `mae <= 0.08`, `picp >= 0.85`, `monotonicity_violation_rate <= 0.15`, `value_range_violation <= 0`.
- `exploratory`: early algorithm discovery, `mae <= 0.12`, `picp >= 0.80`, `monotonicity_violation_rate <= 0.25`, `value_range_violation <= 0`.

## UI Behavior

- The third KPI changes by active tab:
  - Training: label is the selected preset, value is aggregate pass subitems over total subitems.
  - Inference: label remains `推理成功`, value is succeeded records over total inference records.
- A compact control next to the KPI area lets the user switch the training threshold preset.
- A `验收详情` drawer lists:
  - Current preset and thresholds.
  - Aggregate pass/fail counts.
  - Dataset-level pass/fail counts.
  - Top failing rows with failed dimensions and measured values.
- The drawer should use existing drawer/backdrop patterns and must not make the page taller.

## battery_SOH Migration Shape

The target profile can expose optional files under `frontend/app/data/profiles/battery-soh-diagnostics/`:

- `algorithm_details.json`: algorithm name, family, training role, inference role, expected feature sets, notes.
- `soh_curves.json`: dataset/cell/cycle/true SOH/predicted SOH rows or compact per-cell series.
- `feature_visuals.json`: feature name, dataset/cell, summary stats, optional binned values.

The current slice only reads these if present and surfaces availability in the detail drawer or diagnostics notes. A later importer can populate them from `D:\battery_SOH`.

## Testing

- Core tests cover preset definitions, recomputed verdicts, aggregate and dataset-level counts, and failing detail rows.
- Static UI tests cover the drawer hooks, preset controls, and cache-busted script version.
- Browser verification checks that training shows preset-aware acceptance details and inference still shows inference success.
