# Task 7 Report: 3-Real-Dataset battery_SOH Matrix Smoke

## Scope

- Ran a real-data LODO training matrix for `battery_soh_tabular_ensemble` with `battery_soh_operational`.
- Covered three real datasets: `rwth_isea_public`, `nasa`, and `oxford`.
- Used the saved Pickle models from training to run an inference-only transfer benchmark on the same three datasets.
- Refreshed front-end data artifacts and the standalone HTML export with this latest run.

## Training Evidence

- Job: `20260701T190555-66d86e40`
- Status: `succeeded`
- Run dir: `runs\20260701T190559`
- Datasets: `rwth_isea_public`, `nasa`, `oxford`
- Estimator: `battery_soh_tabular_ensemble`
- Featureset: `battery_soh_operational`
- Protocol: `lodo`
- Saved Pickle models: 3
- Model warnings: 0

Saved models:

- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_rwth_isea_public_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_nasa_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_oxford_seed0`

## Inference Evidence

- Job: `20260701T190613-28e59dfd`
- Status: `succeeded`
- Run dir: `runs\20260701T190614`
- Records: 9
- Succeeded: 9
- Applied feature transforms: `battery_soh_operational`
- Best transfer model: `battery_soh_tabular_ensemble_battery_soh_operational_lodo_nasa_seed0`
- Best transfer MAE: `0.01769245386944416`
- Fastest model: `battery_soh_tabular_ensemble_battery_soh_operational_lodo_rwth_isea_public_seed0`
- Fastest ms / 1000 samples: `27.06314233149352`

## Front-End Artifacts

- `frontend/app/data/run_config.json`
- `frontend/app/data/leaderboard.json`
- `frontend/app/data/inference_run_config.json`
- `frontend/app/data/inference_benchmark.json`
- `dist/soh-eval-standalone.html`
- `docs/exports/soh-eval-standalone.html`

## Verification

- API training job completed successfully.
- API inference job completed successfully.
- `tests/test_standalone_frontend.py` passed after rebuilding the standalone export.
