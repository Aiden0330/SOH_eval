# battery_SOH Curve Preview Design

## Goal

Add a lightweight SOH/feature curve preview to the diagnostics frontend by generating small sampled preview artifacts from `D:\battery_SOH\dashboard\data\*_curves.json`.

## Scope

This slice does not copy full curve JSON by default. The importer creates one compact `battery_curve_previews.json` file containing a few sampled curves per dataset. The frontend uses that file to render a dataset-selectable SVG preview inside the existing global diagnostics asset detail section.

## Data Contract

`battery_curve_previews.json`:

```json
{
  "schema_version": 1,
  "max_curves_per_dataset": 3,
  "max_points_per_curve": 80,
  "datasets": [
    {
      "dataset": "xjtu",
      "kind": "charge_curve",
      "source_file": "D:\\battery_SOH\\dashboard\\data\\xjtu_curves.json",
      "total_curves": 3026,
      "preview_curves": [
        {
          "id": "batch-1:cell-1|21",
          "point_count": 128,
          "sampled_point_count": 80,
          "x_key": "t_s",
          "series": [
            {
              "key": "voltage_v",
              "label": "Voltage",
              "min": 3.1,
              "max": 4.2,
              "points": [[0, 3.1], [1, 3.2]]
            }
          ]
        }
      ]
    }
  ]
}
```

For `ess_detection_curves.json`, `kind` is `ess_detection`, `x_key` is `sample`, and `series` is built from `dev_mv` channels.

## Frontend Behavior

- Global diagnostics still shows algorithm details, model comparisons, feature windows, and curve index.
- A new curve preview panel appears below the curve index when preview data exists.
- The user can select a dataset; the panel renders the first preview curve as an inline SVG with up to three series.
- The visualization is a compact preview for orientation, not a full analytical plot.

## Error Handling

Invalid curve files are skipped. Datasets with no numeric points do not enter `battery_curve_previews.json`. If no previews are available, the frontend keeps the existing curve index table and shows no preview panel.

## Testing

- Python importer tests verify preview artifact generation, sampling, manifest inclusion, and default full-curve exclusion.
- Frontend core tests verify dataset option selection and chart-ready preview normalization.
- Static frontend tests verify the panel, selector, and CSS hooks exist.
