# SOH 评测工具使用说明书

版本：2026-06-29  
项目目录：`D:\SOH_eval`  
默认数据目录：`D:\SOH_eval\data`

## 1. 工具目标

本工具用于电池 SOH（State of Health）算法的统一测评。它支持在同一数据目录下发现多个数据集，统一执行数据质量检查、特征方案选择、模型训练、跨域评测、指标汇总和前端展示。

当前工作重点是建立“可复现、可比较、可扩展”的本机可信评测系统：

- 支持项目目录 `D:\SOH_eval\data` 下已有和后续新增的数据集。
- 支持内置算法、本机可信插件算法、深度模型算法。
- 支持内置特征方案和本机可信插件特征方案。
- 支持通过前端选择数据集、算法、特征方案并触发评测。
- 支持公平评测：同一轮评测中使用相同数据集、协议、特征方案、seed 和训练/测试划分。

## 2. 系统组成

| 组件 | 默认位置/地址 | 作用 |
|---|---|---|
| 数据目录 | `D:\SOH_eval\data` | 存放原始数据和 `processed/*_features.pkl` |
| 后端 API | `http://127.0.0.1:8001` | 上传数据集/算法/特征，创建评测任务，查询任务状态 |
| 前端页面 | `http://localhost:8000` | 数据集选择、上传、评测配置、结果展示 |
| 前端数据产物 | `frontend/app/data` | 前端读取的 `leaderboard.json`、`run_config.json` 等 |
| 评测结果归档 | `runs/<timestamp>` | 每轮评测的完整结果、配置和 QC 报告 |
| 可信插件目录 | `work/trusted_plugins` | 本机上传的数据集、算法、特征插件和任务记录 |

## 3. 数据目录约定

评测器默认从以下目录发现数据集：

```text
D:\SOH_eval\data\processed
```

可被自动发现的数据文件必须满足：

- 文件名格式：`<dataset_name>_features.pkl`
- 必需列：`dataset`, `cell_id`, `cycle_index`, `soh`
- 其它列作为候选特征列参与特征方案筛选

当前 7 个有效数据集是：

| 数据集 | 说明 |
|---|---|
| `calce_cs2` | CALCE CS2 电池数据 |
| `hust` | HUST 电池数据 |
| `nasa` | NASA PCoE 电池数据 |
| `oxford` | Oxford 电池数据 |
| `rwth_isea_public` | 新增 RWTH ISEA 公开数据 |
| `snl_lfp` | SNL LFP 数据 |
| `uconn` | UConn 电池数据 |

注意：只有 `*_features.pkl` 会被 `DatasetLoader.discover_datasets()` 自动发现。类似 `rwth_id*_events*.pkl` 的事件文件不会直接进入评测，除非先转换成统一特征表。

## 4. 启动前后端

在项目根目录 `D:\SOH_eval` 下执行。

启动前端静态服务：

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory frontend/app
```

启动后端 API：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m sohbench.api_server `
  --host 127.0.0.1 `
  --port 8001 `
  --data-dir D:\SOH_eval\data `
  --work-dir work\trusted_plugins `
  --runs-dir runs `
  --frontend-data-dir frontend\app\data
```

浏览器访问：

```text
http://localhost:8000
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/health
```

## 5. 前端使用流程

前端界面主要面向三类操作：选择评测范围、上传新资源、查看结果。

1. 打开 `http://localhost:8000`。
2. 在数据集区域勾选需要参与评测的数据集。
3. 在算法区域选择内置算法或上传的新算法。
4. 在特征方案区域选择内置特征或上传的新特征方案。
5. 确认协议、seed 和预计组合数。
6. 点击评测相关按钮触发任务。
7. 在结果区域查看全局榜单、覆盖范围、每个 held-out 数据集的表现和 QC 摘要。

前端展示的是 `frontend/app/data` 下的 JSON 产物。若后端任务成功完成，产物会自动同步到该目录。

## 6. 评测协议

| 协议 | 名称 | 含义 | 适用场景 |
|---|---|---|---|
| `loco` | Leave One Cell Out | 每次留出一个电芯测试 | 同数据集内泛化 |
| `lodo` | Leave One Dataset Out | 每次留出一个数据集测试 | 跨数据集泛化，当前公平比较主协议 |
| `lcho` | Leave One Chemistry Out | 每次留出一种化学体系测试 | 跨化学体系泛化 |

当前 7 数据集公平测评使用 `lodo`。这意味着每个数据集都会轮流作为未知测试域，其它数据集用于训练。

## 7. 算法测评到底测什么

当前榜单不是只测推理时间，而是完整的“训练 + 推理 + 指标计算”。

每个组合都会执行：

```text
算法 × 特征方案 × 协议折 × seed
```

在每个折中，评测器会：

1. 根据协议划分训练集和测试集。
2. 实例化一个新的算法对象。
3. 在训练折执行 `fit(...)`。
4. 在 held-out 测试折执行 `predict(...)`。
5. 执行 `predict_interval(...)` 或 conformal wrapper，用于区间校准指标。
6. 计算 MAE、RMSE、PICP、单调性、值域违规等指标。

因此，深度模型长跑会很慢。当前深度矩阵是：

```text
4 个深度算法 × 6 个特征方案 × 7 个 LODO 折 × 1 个 seed = 168 次独立训练+推理
```

本机 PyTorch 当前为 CPU 版，`CUDA=False`，训练全部发生在 CPU 上。深度模型默认参数较重：`cnn1d/lstm/transformer` 默认 `epochs=30, n_max=20000`，`pinn` 默认 `epochs=80, n_max=10000`。

如果要测部署时的推理速度，需要另做 inference-only benchmark：固定已训练模型，只统计 `predict(...)` 时间。

## 8. 命令行评测

不经过前端时，可以直接用 CLI 启动评测。

示例：运行 7 数据集、LODO、Ridge 基准：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m sohbench.cli run `
  --data-dir D:\SOH_eval\data `
  --datasets all `
  --estimators ridge `
  --protocols lodo `
  --featuresets full `
  --seeds 0 `
  --out-dir runs
```

示例：运行 CPU 公平矩阵：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m sohbench.cli run `
  --data-dir D:\SOH_eval\data `
  --datasets all `
  --estimators ridge gbdt svr gpr mlp byo_skb_ridge phys_resid `
  --protocols lodo `
  --featuresets full voltage ic_dva partial_window capacity_time `
  --seeds 0 `
  --out-dir runs
```

注意：CLI 只加载内置资源。若要同时使用本机可信插件（例如 `trusted_huber_decay`、`rwth_cycle_shape`），推荐使用 `EvaluationService` 或前端/后端 API。

## 9. 后端 API 使用

后端提供以下主要接口：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/catalog` | 查询当前可用数据集、算法、特征方案、任务 |
| GET | `/api/jobs` | 查询任务列表 |
| GET | `/api/jobs/<job_id>` | 查询单个任务状态 |
| POST | `/api/upload/dataset` | 上传新数据集 |
| POST | `/api/upload/algorithm` | 上传新算法插件 |
| POST | `/api/upload/featureset` | 上传新特征方案插件 |
| POST | `/api/jobs` | 创建并启动新评测任务 |

创建评测任务的 JSON 示例：

```json
{
  "datasets": ["all"],
  "estimators": ["ridge", "gbdt", "trusted_huber_decay"],
  "featuresets": ["full", "capacity_time", "rwth_cycle_shape"],
  "protocols": ["lodo"],
  "seeds": [0]
}
```

## 10. 上传新资源

### 10.1 上传新数据集

数据集插件最终应形成一个符合规范的 `*_features.pkl` 文件。最小字段要求：

```text
dataset, cell_id, cycle_index, soh
```

建议额外提供：

```text
voltage_mean, voltage_std, voltage_min, voltage_max,
ic_peak_ah_per_v, ic_peak_voltage_v,
charge_capacity_max, time_s_max,
cycle_index_feature, cycle_fraction
```

### 10.2 上传新算法

算法插件需要提供一个 estimator 类，并满足：

- 有唯一 `name`
- 有 `feature_mode`，通常是 `unified`
- 有 `backend`，例如 `sklearn`, `pytorch`, `external`
- 实现 `fit(X, y, meta=None)`
- 实现 `predict(X, meta=None)`

最小结构示例：

```python
class Estimator:
    name = "my_algorithm"
    feature_mode = "unified"
    backend = "sklearn"

    def fit(self, X, y, meta=None):
        return self

    def predict(self, X, meta=None):
        return predictions
```

### 10.3 上传新特征方案

特征方案可以是列选择，也可以包含 transform。当前 `rwth_cycle_shape` 就会从循环序号和电芯进度派生新特征。

特征方案应声明：

- `name`
- 候选列或 transform 逻辑
- 输出列名

## 11. 结果产物说明

每轮评测会在 `runs/<timestamp>` 下生成：

| 文件 | 说明 |
|---|---|
| `results.csv` | 最细粒度结果，每个 metric 一行 |
| `results.parquet` | 与 CSV 等价的二进制结果，便于后续分析 |
| `leaderboard.json` | 全局榜单，按算法/特征/协议汇总 |
| `dataset_leaderboard.json` | 按 held-out 数据集展开的榜单 |
| `qc_report.json` | 数据质量检查报告 |
| `run_config.json` | 本轮评测配置 |

前端读取的文件位于：

```text
frontend/app/data
```

该目录通常包含：

```text
leaderboard.json
dataset_leaderboard.json
qc_report.json
run_config.json
```

## 12. 指标解释

| 指标 | 含义 | 越小/越大 |
|---|---|---|
| `mae` | 平均绝对误差 | 越小越好 |
| `rmse` | 均方根误差 | 越小越好 |
| `mape` | 平均绝对百分比误差 | 越小越好 |
| `mean_bias_error` | 平均偏差 | 越接近 0 越好 |
| `overestimation_rate` | 高估比例 | 视风险偏好解释 |
| `max_error` | 最大误差 | 越小越好 |
| `picp` | 区间覆盖率 | 越接近目标置信水平越好 |
| `monotonicity_violation_rate` | SOH 单调性违规率 | 越小越好 |
| `value_range_violation` | 预测值越界率 | 越小越好 |
| `eol_region_error` | EOL 区域误差 | 越小越好，部分 held-out 数据集可能不可定义 |

当前前端会显示“覆盖范围”，用于区分真正覆盖全部数据集的全局结果和局部数据集结果。公平比较时，应优先看覆盖 `7/7` 的组合。

## 13. 当前公平基准结果

已完成的 CPU 公平 run：

```text
runs\20260629T151929
```

配置：

- 数据集：7 个全量数据集
- 协议：`lodo`
- seed：`0`
- 算法：`ridge`, `gbdt`, `svr`, `gpr`, `mlp`, `byo_skb_ridge`, `phys_resid`, `trusted_huber_decay`
- 特征方案：`full`, `voltage`, `ic_dva`, `partial_window`, `capacity_time`, `rwth_cycle_shape`

审计结果：

- 43 个可比组合全部覆盖 `7/7` 个 held-out 数据集。
- 失败折为 `0`。
- RWTH 已真实进入评测，`rwth_isea_public` 保留 `2707` 行。
- QC 输入 `235242` 行，仅 `18` 行因 `soh_nonpositive` 剔除。

## 14. 当前深度模型长跑

已启动的深度模型后台长跑：

```text
Job ID: 20260629T161004-3806120d
日志目录: D:\SOH_eval\work\deep_long_runs\20260629T160959
状态文件: D:\SOH_eval\work\deep_long_runs\20260629T160959\status.json
```

矩阵：

- 数据集：7 个全量数据集
- 协议：`lodo`
- seed：`0`
- 深度算法：`cnn1d`, `lstm`, `transformer`, `pinn`
- 特征方案：6 个全部特征方案

查看状态：

```powershell
Get-Content -Raw D:\SOH_eval\work\deep_long_runs\20260629T160959\status.json
Get-Content -Tail 40 D:\SOH_eval\work\deep_long_runs\20260629T160959\deep_lodo.out.log
Get-Content -Tail 40 D:\SOH_eval\work\deep_long_runs\20260629T160959\deep_lodo.err.log
```

如果任务成功，脚本会自动：

1. 生成 deep-only run。
2. 与 CPU 公平 run `runs\20260629T151929` 合并。
3. 生成新的合并 run。
4. 同步到 `frontend/app/data`。

## 15. 合并多个 run 并发布前端

当需要把多个评测结果合并到前端时，使用：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\build_frontend_artifacts.py `
  runs\20260629T151929 `
  runs\<deep_run_timestamp> `
  --phase fair7_lodo_cpu_plus_deep `
  --frontend-data-dir frontend\app\data
```

该脚本会生成一个新的 `runs/<timestamp>`，并同步前端 JSON。

## 16. 公平性注意事项

公平比较必须满足：

- 同一轮结果使用同一组数据集。
- 同一轮结果使用同一协议，例如都使用 `lodo`。
- 同一轮结果使用同一 seed 设置。
- 同一轮结果必须覆盖相同 held-out 范围，例如 `7/7`。
- 不能把 6 数据集旧结果和 RWTH 局部结果直接混成全局榜单。
- 对异构数据集的结构性缺失特征，应使用训练折内统计量处理，避免 held-out 测试集泄漏。

当前实现中，结构性缺失特征通过训练折中位数插补处理。这样 RWTH 这类数据集不会因为缺少其它数据集的特征列被整集误删。

## 17. 常见问题

### 17.1 为什么选择不同数据集后结果不变化？

如果前端只读取静态 `frontend/app/data`，而没有触发新的后端评测，显示的就是当前发布产物，不会随勾选立即重新计算。需要通过后端创建新 job，或使用前端的评测触发功能。

### 17.2 为什么深度模型跑很久？

因为评测包含训练过程，并且每个 LODO 折都要重新训练。深度长跑共有 168 次独立训练+推理，当前又是 CPU 版 PyTorch。

### 17.3 可以只测推理速度吗？

可以，但需要新增 inference-only benchmark。流程是先固定训练好的模型，再只统计 `predict(...)` 时间。当前全局榜单不是推理延迟榜单。

### 17.4 为什么有些指标是空值？

某些 held-out 数据集可能没有满足 EOL 区域定义的数据点，因此 `eol_region_error` 可能不可定义。这不是模型失败，需要结合 `error` 列和覆盖范围判断。

### 17.5 新数据集没有出现在前端怎么办？

依次检查：

1. 文件是否在 `D:\SOH_eval\data\processed`。
2. 文件名是否为 `<dataset_name>_features.pkl`。
3. 是否包含必需列 `dataset`, `cell_id`, `cycle_index`, `soh`。
4. 后端 `/api/catalog` 是否能看到该数据集。
5. 是否已经重新生成并同步 `frontend/app/data`。

## 18. 推荐工作流

日常快速验证：

1. 新数据集或新算法上传。
2. 先跑 `ridge + full + lodo + seed 0` smoke。
3. 再跑小矩阵，例如 `ridge gbdt trusted_huber_decay` × `full capacity_time rwth_cycle_shape`。
4. 通过后再加入更多算法和特征。

正式公平比较：

1. 固定数据集范围。
2. 固定协议为 `lodo`。
3. 固定 seed。
4. 跑完整算法/特征矩阵。
5. 审计覆盖范围必须一致。
6. 发布前端产物。

深度模型：

1. 优先后台运行。
2. 记录状态文件和日志目录。
3. 完成后合并 CPU run。
4. 前端只发布合并后的统一榜单。

## 19. 快速排障命令

查看 Python 进程：

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,StartTime,Path |
  Format-Table -AutoSize
```

查看后端任务：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/api/jobs
```

查看前端当前数据集配置：

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/data/run_config.json |
  Select-Object -ExpandProperty Content
```

检查前端文件是否同步：

```powershell
Get-ChildItem frontend\app\data
```

检查 run 目录：

```powershell
Get-ChildItem runs\<timestamp>
```

## 20. 交付与归档建议

建议每次正式评测后记录：

- run 目录
- 数据集清单
- 算法清单
- 特征方案清单
- 协议和 seed
- QC 摘要
- 覆盖范围审计
- Top-K MAE/RMSE 结果
- 是否已同步前端

这样可以避免后续把不同范围、不同协议、不同数据版本的结果误作同一公平榜单比较。
