# Task 5 Report: battery_SOH Closed Loop Smoke

## Scope

Run a true small-scale loop with the imported `battery_SOH` algorithm and feature plugins:

1. Register `D:\battery_SOH` plugins.
2. Train `battery_soh_tabular_ensemble` with `battery_soh_operational`.
3. Save Pickle models.
4. Restart backend.
5. Run inference-only with the saved Pickle models.
6. Refresh frontend and standalone artifacts.

## Bugs Found And Fixed

### Inference Feature Transform Replay

Symptom:

- Inference job completed, but all records failed with missing feature columns:
  `battery_soh_cycle_norm`, `battery_soh_voltage_sag`, `battery_soh_resistance_growth`.

Root cause:

- Training applied trusted featureset transforms before fitting.
- Inference loaded raw dataset tables and did not replay the transforms required by model manifests.

Fix:

- `EvaluationService.run_inference_job_inline()` now derives selected model featuresets and applies available trusted feature transforms before inference.
- `inference_run_config.json` records `applied_featureset_transforms`.

### Stable Trusted Plugin Module Names

Symptom:

- Pickled trusted-plugin models could fail after backend restart with `No module named 'sohbench_trusted_estimator_...'`.

Root cause:

- Trusted plugin module names used Python's process-random `hash(path)`.

Fix:

- Trusted plugin module names now use a stable SHA-256 path token.
- Added cross-process model-load regression coverage.

## Final Runtime Evidence

Training job:

- `20260701T181720-dbcd16c5`
- Status: `succeeded`
- Run dir: `runs\20260701T181722`
- Saved models:
  - `battery_soh_tabular_ensemble_battery_soh_operational_loco_smoke_eval_demo_smoke_cell_01_seed0`
  - `battery_soh_tabular_ensemble_battery_soh_operational_loco_smoke_eval_demo_smoke_cell_02_seed0`
  - `battery_soh_tabular_ensemble_battery_soh_operational_loco_smoke_eval_demo_smoke_cell_03_seed0`
- `model_warnings`: none.

Inference job:

- `20260701T181757-ed4e201b`
- Status: `succeeded`
- Run dir: `runs\20260701T181757`
- Records: 3
- Succeeded records: 3
- Best transfer MAE: `0.008688697906536943`
- Fastest speed: `1768.3190476201985 ms / 1000 samples`
- Applied transforms: `battery_soh_operational`

## Verification

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_battery_soh_plugins.py tests/test_api_server.py tests/test_trusted_plugins.py tests/test_battery_dashboard_import.py tests/test_standalone_frontend.py -q
node --test frontend/app/chinese-ui.test.mjs frontend/app/dashboard-core.test.mjs
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output dist\soh-eval-standalone.html
```

Results:

- Python targeted tests: 16 passed.
- Node frontend tests: 2 passed.
- Standalone frontend test: 1 passed.
- Standalone refreshed:
  - `dist\soh-eval-standalone.html`
  - `docs\exports\soh-eval-standalone.html`
