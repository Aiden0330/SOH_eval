### Task 1: Diagnostic Artifact Import Contract

**Files:**
- Create: `src/sohbench/battery_dashboard_import.py`
- Create: `tests/test_battery_dashboard_import.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/api_server.py`
- Create: `scripts/import_battery_dashboard_profile.py`

**Interfaces:**
- Produces: `stage_battery_dashboard_artifacts(source_dir: Path, target_dir: Path, include_curves: bool = False) -> list[str]`
- Produces: `publish_battery_dashboard_profile(frontend_data_dir: Path, source_dir: Path, include_curves: bool = False, profile_name: str = "battery-soh-diagnostics") -> dict`
- Produces API endpoint: `POST /api/import/battery-dashboard`

**Global Constraints:**
- Use TDD: write failing tests before production changes.
- Do not move or copy raw data from `D:\battery_SOH\data`.
- Imported dashboard artifacts must use `battery_dashboard_` prefixes.
- Curve artifacts are optional and excluded by default.
- Existing `leaderboard.json`, `dataset_leaderboard.json`, `run_config.json`, `model_catalog.json`, and inference artifacts must remain unchanged.

**Behavior Requirements:**
- `stage_battery_dashboard_artifacts` copies only JSON files from a `dashboard/data`-style source directory.
- It must rename `manifest.json` to `battery_dashboard_manifest.json` so it cannot collide with a profile's own `manifest.json`.
- It must rename `overview.json` to `battery_dashboard_overview.json`.
- It must rename scenario files such as `xjtu.json` to `battery_dashboard_xjtu.json`.
- It must rename `ess_detection.json` to `battery_dashboard_ess_detection.json`.
- By default, files ending `_curves.json` must not be copied.
- With `include_curves=True`, curve files such as `xjtu_curves.json` and `ess_detection_curves.json` must be copied with the same prefix rule.
- Return the copied target file names sorted in deterministic order.
- `publish_battery_dashboard_profile` must stage artifacts to a temporary source directory and call the existing `publish_frontend_profile` so the profile index and profile manifest are written consistently.
- Published profile manifest should include `source: "battery_SOH_dashboard"`, `kind: "battery_soh_diagnostics"`, `include_curves`, and `diagnostic_artifacts`.
- API endpoint request JSON accepts:
  - `source_dir`, default `D:\battery_SOH\dashboard\data`
  - `profile_name`, default `battery-soh-diagnostics`
  - `include_curves`, default `false`
- API endpoint response should include `{ "ok": true, "profile": <publish result> }` on success.

**Test Requirements:**
- Add tests in `tests/test_battery_dashboard_import.py`.
- Test RED before production code exists.
- Test that default staging excludes curves and uses prefixed names.
- Test that including curves copies curve files.
- Test that publishing creates `frontend_data/profiles/battery-soh-diagnostics/manifest.json`, records imported artifact files, and does not overwrite that profile manifest with the source `manifest.json`.
- Test API endpoint by creating a temporary source dashboard directory and posting to `/api/import/battery-dashboard`.

**Commands:**
- RED/GREEN targeted: `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_battery_dashboard_import.py tests/test_api_server.py -q`

**Report:**
- Write report to `.superpowers/sdd/task-1-report.md`.
- Include RED command/failure summary, GREEN command/pass summary, changed files, and concerns.
