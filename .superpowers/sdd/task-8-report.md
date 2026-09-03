# Task 8 Report: 7-Real-Dataset battery_SOH LODO Full-Scope Smoke

## Scope

- Ran a full real-dataset LODO training matrix for `battery_soh_tabular_ensemble`.
- Covered all 7 real datasets currently in `D:\battery_SOH\data`, excluding only `smoke_eval_demo`.
- Used the saved Pickle models to run inference-only transfer evaluation across the same 7 datasets.
- Refreshed front-end data artifacts and standalone HTML export with the latest full-scope result.

## Dataset Scope

- `calce_cs2`
- `hust`
- `nasa`
- `oxford`
- `rwth_isea_public`
- `snl_lfp`
- `uconn`

## Training Evidence

- Job: `20260701T191416-92164f75`
- Status: `succeeded`
- Run dir: `runs\20260701T191525`
- Estimator: `battery_soh_tabular_ensemble`
- Featureset: `battery_soh_operational`
- Protocol: `lodo`
- Seeds: `0`
- Saved Pickle models: 7
- Model warnings: 0

Saved models:

- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_calce_cs2_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_hust_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_nasa_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_oxford_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_rwth_isea_public_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_snl_lfp_seed0`
- `battery_soh_tabular_ensemble_battery_soh_operational_lodo_uconn_seed0`

## Inference Evidence

- Job: `20260701T191557-672777ce`
- Status: `succeeded`
- Run dir: `runs\20260701T191616`
- Dataset count: 7
- Model count: 7
- Records: 49
- Succeeded: 49
- Applied feature transforms: `battery_soh_operational`
- Best transfer model: `battery_soh_tabular_ensemble_battery_soh_operational_lodo_calce_cs2_seed0`
- Best transfer MAE: `0.0173677553112443`
- Fastest model: `battery_soh_tabular_ensemble_battery_soh_operational_lodo_snl_lfp_seed0`
- Fastest ms / 1000 samples: `8.343283694463056`

## Front-End Artifacts

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
