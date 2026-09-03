# sohbench — SOH benchmark harness and evaluation workbench

`sohbench` evaluates battery state-of-health (SOH) estimators under leakage-safe protocols:
a unified estimator interface, hard label-safety gates, leave-one-domain protocols, conformal
intervals, adaptation operators with declared information channels, and immutable provenance.
On top of the harness sits a human-driven workbench (jobs, cases, evidence pages, trusted plugins,
a static frontend).

**Status (2026-09-02).** The durable part of the harness is being extracted into a kernel package,
`sohkernel`, so that a new agent platform (`../soh-agent`, package `sohagent`) can depend on the
kernel alone. The workbench is frozen at tag `v0-workbench`: bug fixes and extraction shims only.
Decision record: `docs/adr/2026-09-02-ADR-001-kernel-extraction-and-agent-platform.md`.

```
sohagent  ──imports──▶  sohkernel  ◀──imports──  sohbench (frozen workbench)
sohkernel never imports sohagent or sohbench   (tests/kernel/test_import_direction.py)
```

## Repository layout

| Path | Contents |
|---|---|
| `src/sohbench/` | The workbench: evaluation service, API (`sohbench.api`), cases, plugins, CLI — plus the harness modules until they move |
| `src/sohkernel/` | The kernel (skeleton today; modules move in task by task per the extraction plan) |
| `tests/`, `tests/kernel/` | Backend suite; kernel tests including the import-direction rule |
| `frontend/app/` | Static workbench frontend (no build step); `node --test frontend/app/` |
| `data/` | Data root (not tracked, except `README.md` and `manifest.json`) |
| `runs/`, `work/` | Benchmark outputs and workbench state (not tracked) |
| `scripts/` | Build, sync, acceptance and gate scripts |
| `docs/adr/` | Architecture decision records |
| `docs/superpowers/specs`, `docs/superpowers/plans` | Specifications and task plans (kernel invariants: `specs/2026-09-02-soh-kernel-invariants.md`; extraction: `plans/2026-09-02-kernel-extraction.md`) |
| `docs/architecture/` | Historical phase notes (see its `README.md`) |

## Quick start

Python 3.11+ (3.12 recommended), Node 20 for the frontend tests.

macOS / Linux:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # add ",torch" for the deep estimators
.venv/bin/python -m pytest -q                                   # backend suite (real-data tests skip when data is absent)
node --test frontend/app/                                       # frontend suites
```

Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
node --test frontend/app/
```

With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev]"`.

### Data root and environment variables

No default path points at a specific machine. Everything is resolved in `sohbench.paths`:

| Variable | Meaning | Default |
|---|---|---|
| `SOHBENCH_DATA_DIR` | Supervised data root (`processed/*_features.pkl`, `raw/`, `curves/`) | `<repo>/data` |
| `SOHBENCH_CURVE_DIR` | Raw V/I/T curve assets | `<data root>/curves` |
| `BATTERY_SOH_ROOT` | Optional sibling `battery_SOH` checkout (TabularSOHEnsemble plugin, native dashboard source) | `<repo parent>/battery_SOH` |

The data itself is not tracked; `data/manifest.json` records the inventory, provenance and
checksums of the migrated datasets, and `tests/test_label_leak_inventory.py` runs the standing
label-leak audit against whatever is present under the data root.

## Run a benchmark

```bash
python -m sohbench run --data-dir "${SOHBENCH_DATA_DIR:-./data}" \
  --datasets nasa calce_cs2 hust oxford --estimators ridge --protocols loco lodo lcho
```

`--datasets all` discovers every `processed/*_features.pkl` that satisfies the
`dataset / cell_id / cycle_index / soh` contract. Outputs land in `runs/<timestamp>/`
(`results.parquet`, `leaderboard.json`, `qc_report.json`, `run_config.json`).

## Serve the workbench

The API is a FastAPI app (`sohbench.api`, wire-compatible with the legacy stdlib
`sohbench.api_server`); the frontend is static and only needs an HTTP server:

```bash
python -m sohbench.api --port 8001 --enable-legacy-execution   # omit the flag for read-only mode
python -m http.server 8765 --directory frontend/app             # then open http://localhost:8765
```

Execution endpoints (training / inference / search jobs, plugin uploads, AI settings) return 403
until `--enable-legacy-execution` is passed; the UI shows 已连接 · 仅浏览 in that state. POSTs also
need the `X-SOHBench-Token` obtained from `GET /api/session`. `python -m sohbench.api.typegen`
regenerates `frontend/app/types/{api.d.ts,openapi.json}`; `tests/test_api_typegen.py` fails on drift.

The native battery dashboard (`frontend/app/battery-dashboard/`) is synced from the `battery_SOH`
checkout with `scripts/sync_battery_dashboard_native.py` and is not tracked; the tests that
inspect it skip when it is absent.

## Tests and CI

`.github/workflows/phase311-production-stability.yml` runs the kernel direction test, the full
backend suite and the frontend suites on Linux, macOS and Windows. The release-gate invariants
that used to live in the phase 3.0 gate script are ordinary tests now:
`tests/workbench/test_release_gates.py` (measured chain to a signed EvidencePack, reload
stability, tamper-evident audit, scope and missing-evidence blocking) and
`tests/kernel/test_golden_invariants.py` (the NASA/B0018 golden fold, run when the real data is
present). `scripts/run_phase311_production_stability.py` remains the local one-shot gate that
writes a machine-readable report.

## Kernel extraction

The extraction is behaviour-preserving: each harness module moves to `src/sohkernel` with a
re-export shim left in `sohbench`, the full suite stays green after every task, and the direction
test never skips. Behaviour changes (forward-chaining protocol, seed injection, fold-local feature
pipelines, explicit failure outcomes) are separate tasks with their own tests. Progress and the
task list: `docs/superpowers/plans/2026-09-02-kernel-extraction.md`; the invariants any tool must
satisfy: `docs/superpowers/specs/2026-09-02-soh-kernel-invariants.md`.

## Historical documents

The phase-by-phase architecture notes (June–August 2026, mostly in Chinese) are indexed in
`docs/architecture/README.md` and are kept as the audit trail; they no longer describe planned work.
Earlier specs and plans under `docs/superpowers/` (2026-06 to 2026-08) and the user guide
`docs/sohbench-evaluation-tool-user-guide.md` describe the workbench as it was built.
