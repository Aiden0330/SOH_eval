# sohbench 前端 - 数据驱动仪表盘

这是子项目 A 的静态前端，用于展示真实 `sohbench` 运行结果：特征 x 算法矩阵、数据集勾选、多 seed 稳定性、按特征集排行榜、算法优劣对比、PICP 校准和数据 QC 报告。

无需构建步骤，使用 HTML / CSS / 原生 JS module。

## 启动

```bash
# 在仓库根目录运行
.venv/Scripts/python.exe -m http.server 8000 --directory frontend/app
# 打开 http://localhost:8000
```

需要通过 HTTP 服务访问。页面使用 ES modules 并 fetch JSON 产物，直接用 `file://` 打开并不可靠。

## 数据产物

页面读取 benchmark run 写出的四个文件：

| 文件 | 用途 |
|---|---|
| `data/leaderboard.json` | KPI、特征 x 算法矩阵、全量排行榜、PICP |
| `data/dataset_leaderboard.json` | 数据集勾选和按 held-out 数据集重算的算法对比 |
| `data/qc_report.json` | QC 面板和各数据集样本数 |
| `data/run_config.json` | 运行配置：数据集、协议、随机种子、特征集，以及 `stability_summary` 稳定性面板数据 |

`data/` 当前来自合并产物 `runs/20260628T114705`：

- Phase 1：`ridge / gbdt / mlp / phys_resid` x `full / voltage / ic_dva / partial_window / capacity_time`
- Phase 2 扩展初跑：`lstm / cnn1d / transformer / pinn` x `partial_window / ic_dva / full`
- Phase 2 PINN 调参：追加 `pinn_tuned` x `ic_dva / partial_window`
- Phase 2 序列模型调参：追加 `transformer_tuned` x `ic_dva / partial_window`
- Top 组合稳定性：将 `mlp / gbdt / lstm / transformer / transformer_tuned` 的关键单元替换为 seeds `0 / 1 / 2` 聚合结果

协议为 leave-one-dataset-out，种子为 0。当前自动发现并纳入 6 个可监督 SOH feature 数据集：`calce_cs2 / hust / nasa / oxford / snl_lfp / uconn`。

当前 Top 关键单元已改为多随机种子聚合。`mlp x partial_window` 仍是当前全量聚合最优，seeds `0 / 1 / 2` 平均 MAE 约 0.097；`gbdt x ic_dva` 平均 MAE 约 0.101，且 seed 波动更小。Phase 2 深度模型中，默认 `transformer x partial_window` 多 seed 平均 MAE 约 0.108，默认 `lstm x partial_window` 约 0.112。PINN 调参后，`pinn_tuned x partial_window` MAE 约 0.199，`pinn_tuned x ic_dva` MAE 约 0.206，优于默认 PINN 但仍落后于 LSTM/Transformer。序列模型调参后的 `transformer_tuned x ic_dva` 在 seed0 曾达到 0.107，但 seeds `0 / 1 / 2` 平均约 0.136，说明该提升不稳定。勾选单个或多个数据集时，页面使用 `dataset_leaderboard.json` 按 held-out 数据集重新聚合，而不是只改显示标签。

## 刷新数据

Phase 1 全量矩阵：

```bash
.venv/Scripts/python.exe -m sohbench run --data-dir "D:\battery_SOH\data" \
  --datasets all --estimators ridge gbdt mlp phys_resid \
  --protocols lodo --featuresets full voltage ic_dva partial_window capacity_time \
  --seeds 0 --out-dir runs
```

Phase 2 受控初跑：

```bash
.venv/Scripts/python.exe scripts/run_phase2_controlled.py \
  --data-dir "D:\battery_SOH\data" \
  --datasets all --featuresets partial_window ic_dva full --seeds 0 \
  --n-max 6000 --seq-epochs 8 --pinn-epochs 20 --out-dir runs
```

`--datasets all` 会扫描 `D:\battery_SOH\data\processed\*_features.pkl`，只纳入满足 `dataset / cell_id / cycle_index / soh` 字段契约的 feature 表。后续新增同类数据集后，不需要改前端代码；重新运行 benchmark 并把四个 JSON 复制到 `frontend/app/data/` 即可。
