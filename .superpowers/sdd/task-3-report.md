# Task 3 Report: Verification

## Commands

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_battery_dashboard_import.py tests/test_api_server.py tests/test_standalone_frontend.py -q
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\import_battery_dashboard_profile.py --source-dir D:\battery_SOH\dashboard\data --frontend-data-dir frontend\app\data --profile-name battery-soh-diagnostics
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output dist\soh-eval-standalone.html
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output docs\exports\soh-eval-standalone.html
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/import/battery-dashboard -Method POST -ContentType 'application/json' -Body ...
```

## Results

- Python targeted tests: 8 passed.
- Node frontend tests: 2 passed.
- Real import from `D:\battery_SOH\dashboard\data`: published `battery-soh-diagnostics` with 12 non-curve dashboard artifacts.
- Backend smoke: `/api/health` returned `ok: true`; `/api/import/battery-dashboard` returned `ok: true`.
- Frontend static smoke: server logs show the refreshed page requested `app.js?v=battery-fusion-20260701`, `dashboard-core.mjs?v=battery-fusion-20260701`, root benchmark artifacts after profile leaderboard 404, and profile diagnostics artifacts:
  - `battery_dashboard_manifest.json`
  - `battery_dashboard_overview.json`
  - `battery_dashboard_ess_detection.json`
- Standalone HTML refreshed at `dist\soh-eval-standalone.html` and `docs\exports\soh-eval-standalone.html`, each embedding 37 JSON artifacts.

## Reviewer Follow-Ups

- Fixed ESS worst-event compatibility for real `module / period / peak_dev_mv` fields.
- Fixed empty `overview.scenarios` fallback to `manifest.scenarios`.
- Added `batteryScenarioRows()` to normalize real `overview.scenarios` fields such as `mode`, `title`, and `algorithms[].metrics.mae`.
- Bumped frontend module cache keys to avoid stale browser modules.
