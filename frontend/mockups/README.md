# C-for-A — sohbench Dashboard Mockups

Three design directions for the **sub-project A** frontend: a dashboard that visualizes SOH-algorithm
benchmark/comparison results. Open [`index.html`](index.html) to switch between them.

> Design system grounded via the `ui-ux-pro-max` skill: **Data-Dense Dashboard** style, blue `#1E40AF` +
> amber accent with green/amber/red status colors, **Fira Sans + Fira Code** (tabular figures for the
> leaderboard). All three are self-contained HTML (only Google Fonts external), accessible (contrast,
> focus, `prefers-reduced-motion`, no emoji icons — inline SVG), and responsive.

## The three directions

| | File | Vibe | Best when |
|---|---|---|---|
| **A** | [mockup-a-lab-light.html](mockup-a-lab-light.html) | **Lab Light** — clean, white, research-tool | day-to-day analysis; reads like a scientific tool |
| **B** | [mockup-b-ops-dark.html](mockup-b-ops-dark.html) | **Ops Dark** — dense observability / Grafana | engineers monitoring many runs; war-room screens |
| **C** | [mockup-c-editorial.html](mockup-c-editorial.html) | **Editorial Report** — bold, verdict-led narrative | stakeholder/exec summaries; "what's the finding" |

All three render the **same four views**: leaderboard (cross-algorithm × protocol), within-vs-cross-domain
generalization gap, PICP calibration vs nominal, and the data-QC report.

## Data provenance

The numbers shown come from a **real `sohbench` run** (not invented):
`ridge / gbdt / svr / gpr` × leave-one-dataset-out × 3 seeds over NASA / CALCE CS2 / HUST / Oxford.
Key figures: SVR best cross-domain MAE `0.197`; Ridge generalization gap `0.089 → 0.479` (5.4×); GPR native
PICP `0.635`; QC pass rate `99.81%`. The only **illustrative** elements are per-algorithm *within-domain*
(loco) bars for non-Ridge estimators (a full leave-one-cell sweep is pending) — these are labeled as such.

## How the real frontend will consume A

These are static mockups. The working frontend reads A's frozen artifact schema:

- `leaderboard.json` → leaderboard table + MAE/PICP bars + verdict badges + `ci_source` pills
- `qc_report.json` → QC panel (pass rate, dropped rows/columns, reasons)
- `verdicts.json` / `gates` → pass/fail badges per acceptance dimension

A JSON Schema for `leaderboard.json` ships with sub-project A (`src/sohbench/schemas/`), so the frontend can
validate before rendering.

## Selected direction

**A · Lab Light** is the chosen basis for the implemented frontend (decided 2026-06-26). B (Ops Dark) and
C (Editorial) are kept as references / alternate views. The working frontend will be built from A's layout
and the Data-Dense Dashboard design system, consuming A's `leaderboard.json` / `qc_report.json` /
`verdicts.json`. Framework TBD (mockups are framework-agnostic HTML/CSS).
