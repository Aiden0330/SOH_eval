# SOH_eval — working notes for Claude Code sessions

This repository holds the **kernel** `sohkernel` (evaluation protocols, feature safety, featuresets, adaptation, datasheets,
statistics, records) and the **frozen workbench** `sohbench` (tag `v0-workbench`; no new features). The agent platform
`sohagent` lives in the sibling repository `../soh-agent` and installs this kernel editable. Direction rule
`sohbench → sohkernel ← sohagent`, enforced by `tests/kernel/test_import_direction.py`.

- Branch in use: `kernel-extraction` (pushed to the private GitHub `Aiden1981/SOH_eval`; `master` is the default branch). CI: `.github/workflows/` (ubuntu + macos matrix, kernel direction step, node suites).
- Board and plan live in `../soh-agent/docs/plans/` (`2026-09-02-master-task-list.md`, `2026-09-05-wave-6-field-readiness.md`). ADR-001: `docs/adr/2026-09-02-ADR-001-kernel-extraction-and-agent-platform.md`.

## Environment and commands

- Use `.venv-mac` (the `.venv` directory is a stale Windows environment). Suite: `.venv-mac/bin/python -m pytest -p no:cacheprovider` (~80 s, `addopts=-q`) plus `node --test frontend/app/`.
- Data root `data/` (`SOHBENCH_DATA_DIR` overrides): `raw/<dataset>/` archives, `processed/<dataset>_features.pkl` (one row per cycle: `dataset, cell_id, cycle_index, soh` + per-cycle columns), `curves/<dataset>_curves.json`, `datasheets/<dataset>.yaml`, `manifest.json` (digests, chemistry, licence, `held_out_ready`). `data/` is git-ignored except `README.md` and `manifest.json`.
- Datasets on disk 2026-09-05: calce_cs2, hust, nasa, oxford, rwth_isea_public (47 cells, NMC), snl_lfp, snl_nmc, snl_nca, uconn; downloading for Wave 6: `raw/tongji`, `raw/matr`, `raw/home_storage_lfp` (Zenodo 12091223, LFP systems 14–21), `raw/field_lfp_28` (Zenodo 13715694, CC BY-NC), `raw/forklift_lfp180` (Mendeley yz4pttm73n v2).

## Onboarding a dataset (the D1 procedure, Spec 06 in `../soh-agent/docs/specs/data-onboarding-datasheet.md`)

1. Extractor script under `scripts/` (model: `scripts/ingest_batterylife_snl.py`), verified against the raw files; one row per cycle (or per charging segment for field data — Spec 12).
2. Window scan (`scripts/build_window_bank.py`, `scripts/rebuild_partial_window_features.py`) and the partial-window bounds recorded in the manifest.
3. Leak-inventory pins (`tests/kernel/`), `scripts/build_data_manifest.py` for digests, `scripts/draft_datasheets.py` for the datasheet; `VERIFY` marks every field without a primary source.
4. The signature is the owner's (or the owner-delegated review recorded in the sheet); chemistry without a primary source stays unverified and the `lcho` folds warn.

## Session protocol

One task per session, in a worktree: `git worktree add ../wt/SOH_eval-<task> -b wave6/<task>`; run the suite from the worktree with the main venv (`../../SOH_eval/.venv-mac/bin/python -m pytest -p no:cacheprovider`; `pythonpath = ["src"]` in `pyproject.toml` makes the worktree's packages win). Commit only after the suite line; one commit per task; `git add -u` plus explicit new paths, never `git add -A` (the profile directories under `frontend/app/data/` and `.claude/launch.json` are untracked or locally modified on purpose). Merge with `git -C ../../SOH_eval merge --ff-only wave6/<task>` and remove the worktree. Update the task row in `../soh-agent` in the same session.

## Rules

- The kernel's thresholds are pinned by `../soh-agent/src/sohagent/thresholds.py`; changing one here fails the platform's registry test — change both, with the rationale.
- Label safety (Spec 01 §5, `sohkernel.feature_safety`): no feature may reconstruct the label; partial windows above 90 % of the depth of discharge are the label; `soh`-derived columns are label-source only.
- The `forward` protocol is pooled early-life by default (owner-confirmed); `earliest` is the calibration split for adaptation claims.
- Datasheets: chemistry needs a quoted primary source; `rwth_isea_public` is NMC (corrected 2026-09-03); `nasa` chemistry is unverified.
- Do not push to a new remote, do not delete raw data, do not edit the frozen workbench for features.
- `data/processed` is shared with the platform's frozen suite v1, which records its digests: before any rebuild that changes a processed table or the manifest, keep a snapshot (the K12 rebuild of 2026-09-05 left `processed/backup_pre_partial_charge_hi/`; the platform's v1 runs use `../soh-agent/runs/data-v1-frozen`). Suite v2 reads the current tables.
