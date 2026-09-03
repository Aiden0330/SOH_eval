# Frontend Sync And Package Actions Design

## Goal

Expose the existing local sync and packaging scripts as trusted-local backend actions and add a Chinese frontend operations panel so users can trigger them without opening a terminal.

## Scope

- Add POST APIs for native dashboard sync, portable directory packaging, and standalone HTML packaging.
- Show the actions in the diagnostics page as an operations card.
- Return actionable metadata: output path, file count, bytes, SOH curve count, and detection-exclusion state.
- Keep ESS detection assets excluded by default.
- Keep the implementation synchronous because the actions are short local file operations compared with training jobs.

## Architecture

The backend will wrap existing script functions instead of spawning shell commands. `EvaluationService` owns path resolution and returns normalized JSON payloads. `api_server.create_handler` maps three POST endpoints to those service methods.

The frontend will keep a small `PACKAGE_STATUS` state object. The diagnostics page will render a card with three buttons and latest results. Buttons call the new API endpoints with defaults matching the current repo layout:

- source dashboard: `D:\battery_SOH\dashboard`
- app directory: `frontend/app`
- portable output: `dist/soh-eval-portable`
- standalone output: `dist/soh-eval-standalone.html`

## Endpoints

- `POST /api/sync/native-dashboard`
  - body: `{ "source": "...", "target": "...", "include_curves": true }`
  - response: `{ "ok": true, "action": "sync_native_dashboard", "manifest": {...} }`
- `POST /api/package/portable`
  - body: `{ "output": "..." }`
  - response: `{ "ok": true, "action": "package_portable", "manifest": {...} }`
- `POST /api/package/standalone`
  - body: `{ "output": "..." }`
  - response: `{ "ok": true, "action": "package_standalone", "manifest": {...} }`

## UI

Add a "数据同步与离线打包" card to the top row of Global Diagnostics. It will contain:

- a source path input for `D:\battery_SOH\dashboard`
- a compact status strip
- three buttons: "同步 SOH 看板", "生成目录包", "生成单 HTML"
- a result list showing latest output path, size, file count, curve count, and whether abnormal detection assets are excluded

## Error Handling

Backend errors return existing JSON error responses. The frontend catches errors, shows a warning status, and leaves prior successful result visible for traceability.

## Testing

- Python API tests verify all three endpoints call real functions on temporary fixtures and return manifest metadata.
- Existing script tests continue to verify ESS detection filtering.
- Frontend static tests verify the card, buttons, endpoint strings, and state renderer exist.
