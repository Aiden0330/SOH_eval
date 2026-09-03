# Task 10 Report: 7-Dataset LOCO Two-Algorithm Comparison

## Scope

- Compared `battery_soh_tabular_ensemble` and `gbdt` on all 7 real datasets.
- Protocol: `loco`
- Featureset: `battery_soh_operational`
- Seed: `0`
- Fold count: 220 `dataset/cell_id` groups
- Training models: 440
- Inference records: 3080

## Dataset Scope

- `calce_cs2`
- `hust`
- `nasa`
- `oxford`
- `rwth_isea_public`
- `snl_lfp`
- `uconn`

## Training Evidence

- Job: `20260701T193439-fea520f7`
- Status: `succeeded`
- Run dir: `runs\20260701T201900`
- Saved Pickle models: 440
- Model warnings: 0

Training leaderboard by algorithm:

| Algorithm | MAE Mean | RMSE Mean | Alpha Accuracy |
| --- | ---: | ---: | ---: |
| gbdt | 0.073925 | 0.087861 | 0.869258 |
| battery_soh_tabular_ensemble | 0.077454 | 0.088667 | 0.872017 |

## Inference Evidence

- Job: `20260701T201934-5577017e`
- Status: `succeeded`
- Run dir: `runs\20260701T203543`
- Models: 440
- Datasets: 7
- Records: 3080
- Succeeded: 3080
- Applied feature transforms: `battery_soh_operational`

Inference transfer by algorithm:

| Algorithm | Records | Success | Mean MAE | Median MAE | Mean RMSE | Mean ms / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gbdt | 1540 | 1540 | 0.080077 | 0.055555 | 0.105995 | 8.530577 |
| battery_soh_tabular_ensemble | 1540 | 1540 | 0.085676 | 0.056947 | 0.107146 | 45.918551 |

## Interpretation

- LOCO evaluates within-dataset cross-cell generalization.
- In this setting, `gbdt` is slightly better on mean MAE/RMSE in both training and inference aggregation.
- `battery_soh_tabular_ensemble` remains close in accuracy and has slightly higher alpha accuracy in the training leaderboard, but inference throughput is slower.
- All 3080 inference records succeeded, so the inference-only path scales to a much larger model matrix.

## Artifacts

- `runs\20260701T203543\loco_algorithm_comparison_summary.json`
- `docs\exports\loco-algorithm-comparison-summary-20260701.json`
- `frontend/app/data/run_config.json`
- `frontend/app/data/leaderboard.json`
- `frontend/app/data/inference_run_config.json`
- `frontend/app/data/inference_benchmark.json`
- `frontend/app/data/model_catalog.json`
- `dist/soh-eval-standalone.html`
- `docs/exports/soh-eval-standalone.html`

## Verification

- API training job completed successfully.
- API inference job completed successfully.
- `tests/test_standalone_frontend.py` passed after rebuilding the standalone export.
