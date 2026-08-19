# 前端数据产物一键生成脚本

脚本位置：`scripts/build_frontend_artifacts.py`

它把一个或多个标准 benchmark run 合并为新的归档 run，并同步前端读取的四个 JSON：

- `leaderboard.json`
- `dataset_leaderboard.json`
- `qc_report.json`
- `run_config.json`

## 常用命令

同步单个新 run：

```powershell
.\.venv\Scripts\python.exe scripts\build_frontend_artifacts.py `
  --source-run runs\<run_id> `
  --phase frontend_combined
```

合并基础 run 与后续新算法/稳定性 run：

```powershell
.\.venv\Scripts\python.exe scripts\build_frontend_artifacts.py `
  --source-run runs\20260628T110144 `
  --source-run runs\20260628T114453 `
  --phase phase1_phase2_seed_stability_combined
```

如果后续 run 中包含 `seed_stability_summary.csv`，且其 `run_config.json` 有 `configs`，脚本会自动将这些算法/特征集单元作为替换对象，避免同一单元被重复聚合。

也可以手动指定替换单元：

```powershell
.\.venv\Scripts\python.exe scripts\build_frontend_artifacts.py `
  --source-run runs\<base_run> `
  --source-run runs\<new_run> `
  --replace-cell mlp:partial_window `
  --replace-cell gbdt:ic_dva
```

## 接入新数据源和新算法

1. 把新数据源处理为 `D:\SOH_eval\data\processed\*_features.pkl`，字段满足 `dataset / cell_id / cycle_index / soh` 等数据契约。
2. 运行 benchmark，使用 `--datasets all` 或显式指定数据集，生成新的 `runs/<run_id>/results.parquet`。
3. 如果是新增算法或新增数据集，直接把该 run 作为额外 `--source-run` 传入本脚本。
4. 脚本会生成新的归档 run，并同步到 `frontend/app/data/`，前端刷新后即可选择数据集并比较算法。
