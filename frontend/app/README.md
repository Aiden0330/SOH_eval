# sohbench frontend (C-for-A) — data-driven dashboard

The implemented frontend for sub-project A, built from the selected **Mockup A "Lab Light"** design. It
renders a real `sohbench` benchmark run: the **feature × algorithm matrix**, a per-feature-set leaderboard,
PICP calibration, and the data-QC report.

No build step — plain HTML / CSS / vanilla JS.

## Run

```bash
# from the repo root, serve the app folder (fetch needs http://, not file://)
.venv/Scripts/python.exe -m http.server 8000 --directory frontend/app
# then open http://localhost:8000
```

Opening `index.html` directly via `file://` also works — it falls back to a small embedded sample so the page
always renders — but **live data requires serving** (the browser blocks `fetch` of local files over `file://`).

## Data

The app fetches three artifacts written by a benchmark run (`runs/<timestamp>/`):

| file | drives |
|---|---|
| `data/leaderboard.json` | KPIs, feature × algorithm matrix, leaderboard, PICP |
| `data/qc_report.json` | QC panel |
| `data/run_config.json` | run metadata (datasets / protocol / seeds) |

`data/` currently holds a real run: `ridge / gbdt / mlp / phys_resid` × `full / voltage / ic_dva`,
leave-one-dataset-out. Best cell: `gbdt × ic_dva` MAE ≈ 0.117 — the IC/DVA feature set generalizes far
better than the full raw-unit set (the feature layer wins generalization).

### Refresh the data from a new run

```bash
.venv/Scripts/python.exe -m sohbench run --data-dir "D:\battery_SOH\data" \
  --datasets nasa calce_cs2 hust oxford --estimators ridge gbdt svr gpr mlp phys_resid \
  --protocols lodo --featuresets full voltage ic_dva --seeds 0 1 --out-dir runs
# copy the run's leaderboard.json / qc_report.json / run_config.json into frontend/app/data/
```

The harness already emits valid JSON (NaN/N-A cells serialize as `null`), so the artifacts can be fetched and
parsed directly.

## Notes

- Interactive: the **feature-set tabs** switch the leaderboard + PICP views; the matrix is colour-graded
  (green = low MAE) with the global best outlined and N/A cells hatched.
- Accessible: contrast-checked palette, visible focus rings, `prefers-reduced-motion`, SVG icons (no emoji),
  `aria-selected` tabs.
- The design system (Data-Dense Dashboard, Fira Sans/Code, blue + amber) comes from the `ui-ux-pro-max` skill;
  see `frontend/mockups/` for the original A/B/C mockups.
