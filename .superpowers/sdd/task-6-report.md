# Task 6 Report: Import Path And Plugin Catalog Hardening

## Scope

- Made the battery_SOH one-click import path user-configurable from the training page.
- Normalized `battery_soh_root`, trusted plugin `base_dir`, and trusted plugin `data_dir` to absolute paths.
- Ensured trusted plugin catalog entries persist absolute plugin paths, so reloading from a different process working directory can still locate the plugin files.
- Refreshed the standalone front-end export after the UI change.

## TDD Evidence

- Added a failing test for relative `battery_soh_root` normalization in `tests/test_battery_soh_plugins.py`.
- Added a failing test for trusted plugin catalog reload from a different working directory in `tests/test_trusted_plugins.py`.
- Added a failing front-end static test for the new `battery-soh-root` input and request payload.
- Implemented the minimal production changes after confirming those tests failed for the expected missing behavior.

## Runtime Evidence

- `POST /api/import/battery-soh-plugins` with `battery_soh_root=D:\battery_SOH` returned `ok: true`.
- API response now reports absolute plugin paths under `D:\SOH_eval\work\trusted_plugins`.
- In-app browser was navigated to `http://127.0.0.1:8010/?v=battery-root-config-20260701&api=http://127.0.0.1:8001` and confirmed `#battery-soh-root` exists with default `D:\battery_SOH`.
- `dist/soh-eval-standalone.html` and `docs/exports/soh-eval-standalone.html` were refreshed.

## Verification

- `.venv\Scripts\python.exe -m pytest tests/test_battery_soh_plugins.py tests/test_api_server.py tests/test_trusted_plugins.py tests/test_battery_dashboard_import.py tests/test_standalone_frontend.py -q`
- `node --test frontend/app/chinese-ui.test.mjs frontend/app/dashboard-core.test.mjs`
