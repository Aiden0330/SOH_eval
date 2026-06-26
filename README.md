# sohbench — SOH Benchmark & Evaluation Harness (Sub-project A)

A unified algorithm interface plus a fair, cross-condition comparison engine for battery State-of-Health
(SOH) estimators. The spine of the SOH_Opus project: the same machinery both evaluates our own SOH
algorithms (sub-project B) and compares different teams' existing algorithms.

- **Spec:** `docs/superpowers/specs/2026-06-26-subproject-A-benchmark-harness-spec.md`
- **Plan:** `docs/superpowers/plans/2026-06-26-subproject-A-sohbench.md`
- **Research grounding:** `docs/superpowers/specs/2026-06-26-research-grounding-and-scheme-decisions.md`

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest -q
```

## Run a benchmark

```bash
.venv/Scripts/python.exe -m sohbench run --data-dir D:\battery_SOH\data \
  --datasets nasa calce_cs2 hust oxford --estimators ridge --protocols loco lodo lcho
```

Outputs land in `runs/<timestamp>/` (`results.parquet`, `leaderboard.json`, `qc_report.json`,
`run_config.json`).
