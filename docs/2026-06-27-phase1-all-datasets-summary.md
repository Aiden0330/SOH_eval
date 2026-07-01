# Phase 1/2 全量数据集 SOH 算法比较摘要

日期：2026-06-27  
数据源：`D:\battery_SOH\data`  
Phase 1 运行产物：`runs/20260627T150400`  
Phase 2 初跑产物：`runs/20260627T163716`  
Phase 2 多特征集扩展产物：`runs/20260627T170500`  
Phase 2 PINN 调参产物：`runs/20260627T172452`、`runs/20260627T172741`  
Phase 2 序列模型调参产物：`runs/20260628T105930`  
Top 组合多 seed 稳定性产物：`runs/20260628T114453`  
前端合并产物：`runs/20260628T114705`，并已同步到 `frontend/app/data/*.json`

## 1. 运行范围

本次运行使用 `--datasets all` 自动扫描 `D:\battery_SOH\data\processed\*_features.pkl`，并只纳入满足 `dataset / cell_id / cycle_index / soh` 字段契约的可监督 SOH feature 表。

当前纳入 6 个数据集：

| 数据集 | QC 后样本数 |
|---|---:|
| calce_cs2 | 15,287 |
| hust | 146,122 |
| nasa | 2,529 |
| oxford | 519 |
| snl_lfp | 61,055 |
| uconn | 2,855 |

QC 输入行数 232,535，剔除 4,168 行，通过率 98.21%。剔除原因包括：`soh_nonpositive=18`、`nan_feature_row=4150`，未出现 `soh_out_of_range`。

## 2. Phase 1 实验配置

| 维度 | 配置 |
|---|---|
| 协议 | `lodo`，leave-one-dataset-out |
| 随机种子 | `0` |
| 算法 | `ridge`, `gbdt`, `mlp`, `phys_resid` |
| 特征集 | `full`, `voltage`, `ic_dva`, `partial_window`, `capacity_time` |
| 区间估计 | 跨域折使用 `weighted_conformal` |

复现命令：

```powershell
.\.venv\Scripts\python.exe -m sohbench run `
  --data-dir D:\battery_SOH\data `
  --datasets all `
  --estimators ridge gbdt mlp phys_resid `
  --protocols lodo `
  --seeds 0 `
  --featuresets full voltage ic_dva partial_window capacity_time `
  --out-dir runs
```

## 3. Phase 1 全量聚合排名

按 MAE 从低到高排序，Top 12 如下：

| 排名 | 算法 | 特征集 | MAE | PICP |
|---:|---|---|---:|---:|
| 1 | mlp | partial_window | 0.070739 | 0.989 |
| 2 | gbdt | ic_dva | 0.098526 | 0.853 |
| 3 | ridge | partial_window | 0.114573 | 0.999 |
| 4 | gbdt | full | 0.117751 | 0.487 |
| 5 | gbdt | voltage | 0.121185 | 0.583 |
| 6 | gbdt | partial_window | 0.121898 | 0.986 |
| 7 | ridge | ic_dva | 0.137411 | 0.393 |
| 8 | phys_resid | full | 0.151114 | 0.528 |
| 9 | mlp | ic_dva | 0.161677 | 0.617 |
| 10 | mlp | voltage | 0.177855 | 0.525 |
| 11 | gbdt | capacity_time | 0.217392 | 0.514 |
| 12 | ridge | voltage | 0.223291 | 0.271 |

全量最佳单元是 `mlp x partial_window`，MAE 约 0.071；同时 PICP 约 0.989，说明该组合在当前聚合视角下兼顾了较低误差和较高覆盖率。

## 4. 按 held-out 数据集最佳组合

| Held-out 数据集 | 最佳算法 | 最佳特征集 | MAE |
|---|---|---|---:|
| calce_cs2 | gbdt | ic_dva | 0.084611 |
| hust | mlp | voltage | 0.059372 |
| nasa | mlp | partial_window | 0.089905 |
| oxford | gbdt | voltage | 0.067176 |
| snl_lfp | gbdt | full | 0.023118 |
| uconn | gbdt | capacity_time | 0.002812 |

结论上，单一算法没有在所有数据集上绝对占优：`mlp` 在 HUST/NASA 更突出，`gbdt` 在 CALCE/Oxford/SNL/UConn 更稳。前端的数据集勾选面板正是为查看这种数据集依赖性而设计。

## 5. Phase 1 初步结论

1. 特征集影响很强：`partial_window` 和 `ic_dva` 在全量聚合中明显优于纯 `full` 盲堆特征。
2. 传统线性模型仍有价值：`ridge x partial_window` 排名第 3，说明部分窗口特征对线性泛化较友好。
3. 树模型跨数据集稳健：`gbdt` 在多个 held-out 数据集上排名第一，尤其在 SNL/UConn 上误差很低。
4. 物理残差模型没有成为全量最优，但作为可解释/物理先验基线仍应保留。

## 6. Phase 2 受控初跑

Phase 2 已启动并完成第一轮受控运行，随后扩展到 `partial_window / ic_dva / full` 三个特征集：

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2_controlled.py `
  --data-dir D:\battery_SOH\data `
  --datasets all `
  --featuresets partial_window ic_dva full `
  --seeds 0 `
  --n-max 6000 `
  --seq-epochs 8 `
  --pinn-epochs 20 `
  --out-dir runs
```

配置说明：先用 `partial_window` 与 Phase 1 当前全量最优 `mlp x partial_window` 对齐，再加入 `ic_dva` 和 `full` 检查特征集敏感性；每个 PyTorch 估计器限制训练样本数和 epoch，确保 CPU 上可完成。

Phase 2 扩展后的全量聚合结果：

| 排名 | 算法 | 特征集 | 后端 | MAE | PICP |
|---:|---|---|---|---:|---:|
| 1 | lstm | partial_window | pytorch | 0.103070 | 0.900 |
| 2 | transformer | partial_window | pytorch | 0.104121 | 0.664 |
| 3 | lstm | ic_dva | pytorch | 0.129320 | 0.320 |
| 4 | transformer | ic_dva | pytorch | 0.142807 | 0.474 |
| 5 | cnn1d | partial_window | pytorch | 0.194591 | 0.658 |
| 6 | pinn | ic_dva | pytorch | 0.218283 | 0.592 |

合并 Phase 1 + Phase 2 后，当前 Top 5 为：

| 排名 | 算法 | 特征集 | 后端 | MAE |
|---:|---|---|---|---:|
| 1 | mlp | partial_window | sklearn | 0.070739 |
| 2 | gbdt | ic_dva | sklearn | 0.098526 |
| 3 | lstm | partial_window | pytorch | 0.103070 |
| 4 | transformer | partial_window | pytorch | 0.104121 |
| 5 | ridge | partial_window | sklearn | 0.114573 |

按 held-out 数据集看，Phase 2 深度模型的最佳组合如下：

| Held-out 数据集 | Phase 2 最佳算法 | 特征集 | MAE |
|---|---|---|---:|
| calce_cs2 | transformer | partial_window | 0.090804 |
| hust | lstm | partial_window | 0.054711 |
| nasa | transformer | full | 0.067938 |
| oxford | transformer | ic_dva | 0.077361 |
| snl_lfp | transformer | partial_window | 0.032171 |
| uconn | lstm | partial_window | 0.016672 |

PINN 专项调参使用 `ic_dva / partial_window`、`n_max=6000`、`epochs=50` 复跑最有希望的候选。正式复跑后，最佳调参结果如下：

| 算法 | 特征集 | 配置摘要 | MAE |
|---|---|---|---:|
| pinn_tuned | partial_window | hidden=64, lambda_mono=0, lambda_range=0, enforce_monotone=false | 0.198999 |
| pinn_tuned | ic_dva | hidden=64, lambda_mono=1, lambda_range=1, enforce_monotone=false | 0.205579 |
| pinn | ic_dva | 默认 Phase 2 配置 | 0.218283 |
| pinn | partial_window | 默认 Phase 2 配置 | 0.261518 |

调参后的 `pinn_tuned` 明显优于默认 PINN，但仍未接近 `lstm / transformer x partial_window` 的 0.103-0.104 MAE，也未超过 `cnn1d x partial_window` 的 0.194591。

序列模型高预算调参使用 `ic_dva / partial_window`、`n_max=12000`、`epochs=20`，比较 LSTM/Transformer 的窗口长度和模型宽度：

| 算法变体 | 特征集 | MAE | 相对默认模型 |
|---|---|---:|---|
| transformer_tuned_w8_d48_e20_n12000 | ic_dva | 0.107277 | 优于默认 transformer ic_dva 0.142807 |
| transformer_tuned_w16_d48_e20_n12000 | ic_dva | 0.114756 | 优于默认 transformer ic_dva，但不如 W=8 |
| transformer_tuned_w8_d48_e20_n12000 | partial_window | 0.112266 | 差于默认 transformer partial_window 0.104121 |
| transformer_tuned_w16_d48_e20_n12000 | partial_window | 0.111384 | 差于默认 transformer partial_window 0.104121 |
| lstm_tuned_w8_h48_e20_n12000 | partial_window | 0.122214 | 差于默认 lstm partial_window 0.103070 |
| lstm_tuned_w16_h48_e20_n12000 | partial_window | 0.118913 | 差于默认 lstm partial_window 0.103070 |

随后对当前 Top 关键单元做 seeds `0 / 1 / 2` 稳定性复跑。该脚本显式按 seed 重新实例化估计器，避免 `BenchmarkRunner` 只设置全局 `np.random.seed` 而模型自身 `random_state/seed` 不变造成伪重复。

| 排名 | 算法 | 特征集 | seeds 0/1/2 MAE 均值 | seed 间标准差 | seed 范围 |
|---:|---|---|---:|---:|---|
| 1 | mlp | partial_window | 0.097021 | 0.022790 | 0.070739-0.111313 |
| 2 | gbdt | ic_dva | 0.100977 | 0.005939 | 0.096657-0.107750 |
| 3 | transformer | partial_window | 0.107600 | 0.006968 | 0.103057-0.115622 |
| 4 | lstm | partial_window | 0.112225 | 0.008010 | 0.103070-0.117942 |
| 5 | transformer_tuned | ic_dva | 0.136223 | 0.026184 | 0.107277-0.158260 |

当前综合判断：全量总榜在单 seed 下由 `mlp x partial_window` 明显领先，但多 seed 后优势收窄，且波动大于 `gbdt x ic_dva`。若优先看稳健性，`gbdt x ic_dva` 和默认 `transformer x partial_window` 更可靠；若优先看最佳单次表现，`mlp x partial_window` 仍值得保留。`transformer_tuned x ic_dva` 证明更大 Transformer 有机会改善 IC/DVA 特征泛化，但多 seed 后不稳定，不应替代默认 Transformer。PINN 可以保留为物理约束对照，但后续若继续投入，应先重构其物理损失和验证策略，而不是单纯扩大 epoch。
