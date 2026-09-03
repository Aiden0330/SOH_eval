# Task 9 Report: 7-Dataset Three-Algorithm LODO Comparison

## Scope

- Compared three algorithms on the same 7 real datasets, same feature transform, and same LODO protocol.
- Algorithms:
  - `battery_soh_tabular_ensemble`
  - `ridge`
  - `gbdt`
- Featureset: `battery_soh_operational`
- Protocol: `lodo`
- Seed: `0`

## Dataset Scope

- `calce_cs2`
- `hust`
- `nasa`
- `oxford`
- `rwth_isea_public`
- `snl_lfp`
- `uconn`

## Training Evidence

- Job: `20260701T192542-b9f52437`
- Status: `succeeded`
- Run dir: `runs\20260701T192646`
- Saved Pickle models: 21
- Model warnings: 0

Training leaderboard by algorithm:

| Algorithm | MAE Mean | RMSE Mean | Alpha Accuracy |
| --- | ---: | ---: | ---: |
| battery_soh_tabular_ensemble | 0.132241 | 0.160910 | 0.670799 |
| gbdt | 0.134763 | 0.167836 | 0.661364 |
| ridge | 0.135785 | 0.157393 | 0.682315 |

## Inference Evidence

- Job: `20260701T192724-f683b9a1`
- Status: `succeeded`
- Run dir: `runs\20260701T192741`
- Models: 21
- Datasets: 7
- Records: 147
- Succeeded: 147
- Applied feature transforms: `battery_soh_operational`

Inference transfer by algorithm:

| Algorithm | Records | Success | Mean MAE | Median MAE | Mean RMSE | Mean ms / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gbdt | 49 | 49 | 0.089604 | 0.060376 | 0.116874 | 5.455111 |
| battery_soh_tabular_ensemble | 49 | 49 | 0.093370 | 0.060936 | 0.115923 | 39.115461 |
| ridge | 49 | 49 | 0.116366 | 0.080994 | 0.138166 | 0.895969 |

## Interpretation

- Training CV leaderboard favors `battery_soh_tabular_ensemble` by a small MAE margin.
- Inference-only transfer aggregation favors `gbdt` by mean MAE.
- `ridge` is the fastest model family by a wide margin, but its transfer error is higher.
- All three algorithms completed the full 7-dataset inference matrix with no failed records.

## Artifacts

- `runs\20260701T192741\algorithm_comparison_summary.json`
- `docs\exports\algorithm-comparison-summary-20260701.json`
- `frontend/app/data/leaderboard.json`
- `frontend/app/data/inference_benchmark.json`
- `dist/soh-eval-standalone.html`
- `docs/exports/soh-eval-standalone.html`

## Verification

- API training job completed successfully.
- API inference job completed successfully.
- `tests/test_standalone_frontend.py` passed after rebuilding the standalone export.
