# battery_SOH Contract Fusion Design

## Goal

Integrate `D:\battery_SOH` into the `SOH_eval` workbench through explicit artifacts and jobs, with `SOH_eval` remaining the primary frontend/backend system.

## Minimum Slice

This first slice imports the existing `D:\battery_SOH\dashboard\data` diagnostic outputs into `SOH_eval` frontend profiles. It does not move raw data, does not replace leaderboard artifacts, and does not rewrite the existing `battery_SOH` dashboard.

## Architecture

`battery_SOH` remains the data lake and diagnostic artifact producer. `SOH_eval` gains a small compatibility layer that stages selected dashboard JSON files with collision-safe names, publishes them as profile artifacts, and loads a lightweight diagnostic summary in the Chinese workbench.

## Artifact Contract

Imported files use the `battery_dashboard_` prefix:

- `manifest.json` becomes `battery_dashboard_manifest.json`
- `overview.json` becomes `battery_dashboard_overview.json`
- `<mode>.json` becomes `battery_dashboard_<mode>.json`
- `<mode>_curves.json` becomes `battery_dashboard_<mode>_curves.json`
- `ess_detection.json` becomes `battery_dashboard_ess_detection.json`
- `ess_detection_curves.json` becomes `battery_dashboard_ess_detection_curves.json`

Curve files are opt-in because they are much larger than summary files.

## Backend Surface

Add a small importer script and service helper that publish a `battery-soh-diagnostics` profile under `frontend/app/data/profiles/`. The profile manifest records source paths and imported file names.

## Frontend Surface

The profile loader fetches optional diagnostic artifacts. The global diagnostics tab shows a compact imported diagnostics section with scenario count, transfer count, ESS detection status, worst event, and imported source label.

## Constraints

- No dependency on a running `battery_SOH` dashboard server.
- No new Python runtime dependencies.
- No raw data copy into `SOH_eval`.
- Existing training, inference, model library, and standalone HTML behavior must remain compatible.
