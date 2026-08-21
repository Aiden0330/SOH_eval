# SOH 评测案例工作台：阶段 2.2 真实黄金案例

> 完成日期：2026-08-12  
> 定位：把阶段 2.1 的受控复测能力落到一条真实、可复验、可对账并已签署的案例链路。这里的签署是本地工作台审计签署，不代表法规认证或第三方认证。

## 交付结论

阶段 2.2 已完成第一条测量数据黄金案例：`phase22-golden-nasa`。它使用真实 NASA 处理数据、NASA/B0018 原始循环曲线以及按 `nasa/B0018` 留出的不可变 LOCO Ridge 模型产物，不使用演示预测或旧看板预测回放。

```text
processed/nasa_features.pkl
  + nasa_curves.json（B0018，102 个可回溯 Cycle）
  + model@sha256:fe67…493cd（LOCO held-out=nasa/B0018）
  → 目标列从 predictor 输入强制剔除
  → 102 条 PredictionRecord
  → 102 条 MetricContribution
  → 102 条 CurveSegmentRef
  → 8 / 8 必需证据
  → 真实模型重新推理
  → 问题闭环
  → KPI 对账一致
  → 本地人工评审批准
  → EvidencePack 签署
```

当前案例 ID、复测 ID 和包 ID 为本地持久化实体，可从 `work/trusted_plugins/phase22_golden_summary.json` 读取；固定的模型、数据、协议、比较和包摘要不依赖界面状态。

## 真实运行结果

黄金案例仅纳入同时存在处理特征和原始曲线的 B0018 Cycle 21–132，共 102 条记录。

| 指标 | 观测值 | `exploratory` 门限 | 结果 |
|---|---:|---:|---|
| MAE | 0.042922 | ≤ 0.12 | 通过 |
| PICP | 0.990196 | ≥ 0.80 | 通过 |
| 单调性违反率 | 0.227723 | ≤ 0.25 | 通过 |
| 值域违反率 | 0.000000 | ≤ 0.00 | 通过 |

`exploratory` 是现有版本化判据集，不是为本案例临时改出的阈值。该案例可作为算法发现阶段的已签署基线；它不应被解释成严格工程准入或法规认证。Cycle 56 存在显著离群残差，仍保留在聚合结果和逐 Cycle 事实表中，没有被静默删除。

## 新事实合同

### PredictionRecord

每条记录固定：`prediction_record_id`、`sample_id`、run、comparison、candidate、model artifact、dataset snapshot、split assignment、cell、cycle、真值、预测、区间、绝对误差和曲线引用。

### MetricContribution

每条记录提供 MAE 分子及贡献、RMSE 平方分子及贡献、PICP 分子及贡献和有符号残差。聚合 KPI 可由逐条贡献重新求和，不再只能信任单独的汇总 JSON。

### CurveSegmentRef

每条记录固定原始曲线键、源文件 SHA-256、信号列表和可用状态。黄金案例的 102 条预测均有匹配原始曲线。

API：

```http
GET /api/v2/cases/{case_id}/prediction-record
GET /api/v2/cases/{case_id}/prediction-record?sample_id=nasa:B0018:77
```

## 受控复测与发布门禁

黄金案例的代表样本被加入回归集后，控制面重新加载冻结的模型 pickle 和处理数据快照，重新调用 `predict` 与 `predict_interval`。完成结果必须同时包含：

- `execution_mode=model_reinference`；
- `prediction_recomputed=true`；
- 原协议引用保持不变；
- 单样本复测 MAE/PICP 检查通过；
- `result_digest` 写入内容寻址存储。

随后问题单置为 `resolved`，案例进入人工评审并批准；服务端将案例事实与 `battery_dashboard_nasa.json` 的同名 KPI 对账，MAE、PICP 和单调性差值均为 0。

EvidencePack manifest 升级到 `1.1.0`，除实体 ID 外还固定以下产物摘要：RunSpec、聚合指标、PredictionRecord、MetricContribution、CurveSegmentRef、原始曲线源和模型产物。签署后包保持不可变。

## 可重复执行

必须让 `sohbench` 指向当前仓库，脚本发现旧 editable install 或旧 worktree 时会拒绝执行：

```powershell
$env:PYTHONPATH='D:\SOH_eval\src'
$env:PYTHONUTF8='1'
python scripts/build_phase22_golden_case.py
```

同一黄金比较已存在并签署时，脚本返回现有案例摘要；如果发现同键但未完成的案例，则失败关闭并要求先检查工作区，不会生成第二份假“通过”。浏览器也使用 profile manifest 中的 `comparison_id` 生成同一幂等键，因此打开黄金 profile 会恢复已签署案例，不会旁路生成空案例。

## 工作台入口

```text
http://127.0.0.1:8765/SOH_eval/frontend/app/index.html?profile=phase22-golden-nasa&mode=audit&v=phase22-golden-20260812b
```

工作台恢复已签署案例并显示：真实黄金案例、8/8 证据、探索判据通过、真实复测完成、问题已闭环、KPI 一致、人工评审已批准和 EvidencePack 已签署。点击轨迹 Cycle 可查看记录主键、真值/预测、残差和该点 MAE 贡献；新浏览点先标记为“待固定”，不会覆盖案例中已经签署的证据点。未签署案例可明确执行“固定当前 Cycle”更新上下文；已签署案例会提示创建新案例版本，不允许原位改写。
