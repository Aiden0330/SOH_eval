# battery_SOH Visual Assets Design

## Goal

Migrate lightweight algorithm details, model comparison summaries, feature-window metadata, and lazy curve availability from `D:\battery_SOH` into the SOH evaluation frontend profile without increasing first-screen data load.

## Approved Scope

This slice extends the existing `battery-soh-diagnostics` profile importer. It does not copy large curve JSON by default and does not build full chart interactions yet. The deliverable is a structured data contract and frontend visibility for whether algorithm details, model comparisons, feature windows, and curve drill-down assets are available.

## Architecture

- `src/sohbench/battery_dashboard_import.py` remains the single importer for battery dashboard assets.
- The importer generates three lightweight optional files when source reports exist:
  - `battery_algorithm_design.json`
  - `battery_model_comparisons.json`
  - `battery_feature_windows.json`
- The profile `manifest.json` records `curve_artifacts[]` with file name, source path, byte size, and dataset id. Curve files are copied only when `include_curves=true`.
- The frontend already reads these optional JSON files and summarizes them through `batteryVisualAssetSummary()`.

## Data Contracts

### `battery_algorithm_design.json`

```json
{
  "schema_version": 1,
  "title": "SOH 算法设计说明书",
  "task": "SOH 点估计 + 95% CI",
  "model_family": "TabularSOHEnsemble",
  "model_structure": ["ridge", "hist_gradient_boosting", "random_forest"],
  "feature_policy": {
    "allowed": ["容量侧充电窗特征"],
    "forbidden": ["DCIR", "RPT 脉冲", "cycle_fraction"]
  },
  "features": ["q_win_ah", "t_win_s", "ic_peak_ah_per_v", "ic_peak_voltage_v"],
  "source_file": "D:\\battery_SOH\\docs\\design\\soh-algorithm-design.md"
}
```

### `battery_model_comparisons.json`

```json
{
  "schema_version": 1,
  "comparisons": [
    {
      "id": "phase2_tabular_reproduction",
      "title": "XJTU 固定结构复现",
      "scope": "real SOHbenchmark XJTU handcraft_features",
      "metrics": {"mae": 0.01908, "rmse": 0.01972, "r2": 0.8359},
      "source_file": "D:\\battery_SOH\\experiments\\reports\\phase2_tabular_reproduction_metrics.json"
    }
  ]
}
```

### `battery_feature_windows.json`

```json
{
  "schema_version": 1,
  "windows": [
    {
      "dataset": "xjtu",
      "feature_voltage_window": [3.7, 4.1],
      "rows": 768,
      "curves": 6,
      "source_plot": "experiments\\reports\\phase4_xjtu_voltage_feature_window.png"
    }
  ]
}
```

### manifest curve index

```json
{
  "curve_artifacts": [
    {
      "dataset": "xjtu",
      "filename": "battery_dashboard_xjtu_curves.json",
      "source": "D:\\battery_SOH\\dashboard\\data\\xjtu_curves.json",
      "bytes": 12287881,
      "copied": false
    }
  ]
}
```

## Error Handling

Missing optional source files are skipped. Invalid optional JSON reports are skipped with no frontend failure because the diagnostics profile is supplemental. The base dashboard JSON copy behavior remains unchanged.

## Testing

- Extend `tests/test_battery_dashboard_import.py` with fixture reports and algorithm doc inputs.
- Verify generated optional JSON schemas, manifest `curve_artifacts`, and default no-curve copy behavior.
- Re-run frontend core/static tests so `batteryVisualAssetSummary()` continues to recognize optional assets.
- Rebuild standalone HTML after profile data changes.
