# Task 4 Report: battery_SOH Plugin Adapter

## Goal

Expose `D:\battery_SOH` model and feature capabilities through SOH_eval's existing trusted local plugin workflow.

## Implemented

- Added `src/sohbench/battery_soh_plugins.py`.
- Added `scripts/register_battery_soh_plugins.py`.
- Added `EvaluationService.import_battery_soh_plugins(...)`.
- Added API endpoint `POST /api/import/battery-soh-plugins`.
- Added a training-page button `接入 battery_SOH 算法/特征`.
- Registered real local plugins under `work/trusted_plugins`:
  - estimator: `battery_soh_tabular_ensemble`
  - featureset: `battery_soh_operational`

## TDD Evidence

RED:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_battery_soh_plugins.py -q
```

Expected failure: `ModuleNotFoundError: No module named 'sohbench.battery_soh_plugins'`.

GREEN:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_battery_soh_plugins.py tests/test_api_server.py tests/test_trusted_plugins.py tests/test_battery_dashboard_import.py -q
node --test frontend/app/chinese-ui.test.mjs frontend/app/dashboard-core.test.mjs
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_standalone_frontend.py -q
```

Results:

- Python targeted tests: 13 passed.
- Node frontend tests: 2 passed.
- Standalone frontend test: 1 passed.

## Runtime Smoke

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\register_battery_soh_plugins.py --battery-soh-root D:\battery_SOH --data-dir D:\battery_SOH\data --work-dir work\trusted_plugins --runs-dir runs --frontend-data-dir frontend\app\data --replace
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/import/battery-soh-plugins -Method POST -ContentType 'application/json' -Body ...
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/catalog
```

Verified catalog contains `battery_soh_tabular_ensemble` and `battery_soh_operational`.

## Notes

- The estimator plugin imports `D:\battery_SOH\soh_core\models\tabular_ensemble.py` at runtime.
- The feature plugin creates cycle-normalized, voltage-sag, and resistance-growth features without target leakage.
- `dist\soh-eval-standalone.html` and `docs\exports\soh-eval-standalone.html` were refreshed from the current frontend bundle.
