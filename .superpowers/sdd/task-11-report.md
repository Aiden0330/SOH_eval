# Task 11 Report - Algorithm Recommendation Page

## Scope

Combined the completed 7-dataset LODO and LOCO comparison results into a read-only algorithm recommendation artifact and frontend tab.

## Inputs

- `runs/20260701T192741/algorithm_comparison_summary.json`
- `runs/20260701T203543/loco_algorithm_comparison_summary.json`

## Output Artifacts

- `frontend/app/data/recommendation_summary.json`
- `docs/exports/algorithm-recommendation-summary-20260701.json`
- `docs/exports/algorithm-recommendation-report.md`
- `dist/soh-eval-standalone.html`
- `docs/exports/soh-eval-standalone.html`

## Result

- Default recommendation: `gbdt`
- LODO transfer: `gbdt` has the lowest mean MAE; `ridge` is the fastest inference baseline.
- LOCO cross-cell generalization: `gbdt` has the lowest mean MAE/RMSE and is also faster than the battery_SOH ensemble in this two-algorithm run.
- Training-process winners are reported separately from inference-transfer winners.

## Implementation

- Added `scripts/build_recommendation_summary.py`.
- Added `algorithmRecommendationView()` to `frontend/app/dashboard-core.mjs`.
- Added an `算法推荐` side-nav item, top tab, fixed-height mode panel, and renderer.
- Added `recommendation_summary.json` to frontend publishing allowlists.
- Updated standalone build coverage to assert the new JSON artifact is embedded.

## Verification

Commands run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_recommendation_summary.py
node --test frontend\app\dashboard-core.test.mjs frontend\app\chinese-ui.test.mjs
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_recommendation_summary.py tests\test_standalone_frontend.py -q
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output dist\soh-eval-standalone.html
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output docs\exports\soh-eval-standalone.html
```

All targeted commands passed.
