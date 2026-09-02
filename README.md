# sohbench — SOH Benchmark & Evaluation Harness (Sub-project A)

A unified algorithm interface plus a fair, cross-condition comparison engine for battery State-of-Health
(SOH) estimators. The spine of the SOH_Opus project: the same machinery both evaluates our own SOH
algorithms (sub-project B) and compares different teams' existing algorithms.

- **Spec:** `docs/superpowers/specs/2026-06-26-subproject-A-benchmark-harness-spec.md`
- **Plan:** `docs/superpowers/plans/2026-06-26-subproject-A-sohbench.md`
- **Research grounding:** `docs/superpowers/specs/2026-06-26-research-grounding-and-scheme-decisions.md`

## Quick start

Python 3.11+ (3.12 recommended). Commands below use `python` from an activated virtualenv;
on macOS/Linux that is `.venv/bin/python`, on Windows `.venv\Scripts\python.exe`.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"        # add ",dl" (or "[dev,dl]") for the torch estimators
python -m pytest -q                      # backend suite (~400 tests, runs against both API impls)
node --test frontend/app/                # frontend suites (Node 20, no package.json / build step)
```

## Run a benchmark

```bash
python -m sohbench run --data-dir ./data \
  --datasets nasa calce_cs2 hust oxford --estimators ridge --protocols loco lodo lcho
```

`--datasets all` discovers every `data/processed/*_features.pkl` that satisfies the
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

## Phase 3.0 release gate

Run the measured train → inference → evidence → retest → reconciliation → EvidencePack flow,
then verify reload stability, audit digests, incompatible-scope blocking, and missing-evidence
blocking:

```bash
PYTHONPATH=src PYTHONUTF8=1 python scripts/run_phase30_release_gate.py
```

(PowerShell: `$env:PYTHONPATH='src'; $env:PYTHONUTF8='1'; python scripts\run_phase30_release_gate.py`.)

The command exits with `0` only when every positive and negative gate matches its expected
outcome. Reports are written to `work/trusted_plugins/phase30_release_gate_report.{json,md}`.

## Phase 3.1 native evidence canvas

The production audit workspace now renders SOH trajectory and canonical V / I / T / IC / DV
evidence from the same `PredictionRecord`, with no iframe runtime or duplicate selectors. See
`docs/architecture/soh-evaluation-phase-3.1-native-evidence-canvas.md` for the API contract,
layout behavior, and acceptance record.

## Phase 3.1.1 production stabilization

Run the deterministic checkout, lifecycle, hot-refresh, API, and frontend regression gate:

```bash
PYTHONPATH=src PYTHONUTF8=1 python scripts/run_phase311_production_stability.py
```

The command is also wired into `.github/workflows/phase311-production-stability.yml` and writes
machine-readable plus human-readable reports to
`work/trusted_plugins/phase311_production_stability_report.{json,md}`. It deliberately does not
substitute mock data for the measured Phase 3.0 gate; run `scripts/run_phase30_release_gate.py`
separately in the controlled local or self-hosted release environment. See
`docs/architecture/soh-evaluation-phase-3.1.1-production-stability.md` for lifecycle defaults,
hot-refresh behavior, and the verified acceptance record.

## Phase 3.9 local data connectivity

The default runtime data root is the repository's `data/` directory (`--data-dir`). Strategy B migrates the supervised
datasets and curve assets already connected to training, inference, and evidence review while
leaving unrelated large upstream archives untouched. The inventory, three-page lineage,
trajectory/evidence indexes, and current real-case timings are documented in
`docs/architecture/soh-evaluation-phase-3.9-local-data-connectivity.md`.
