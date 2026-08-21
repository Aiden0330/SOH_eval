# SOH 测评与详情看板深度融合重构方案

## 1. 结论

当前产品不应继续维持“测评页给结论、SOH 完整看板查曲线”的双页结构。目标应是一个以 `Comparison -> Asset -> Cycle -> Evidence` 为主线的 **比较与证据工作台**：排行榜、跨域矩阵、逐电芯误差、SOH 轨迹、充放电曲线、IC/DVA、特征窗口、判据和谱系共享同一选择上下文。

SOH 完整看板不再作为与训练/推理平级的顶层页签，而成为以下入口共同打开的证据视图：

- 比较结果中的候选算法行；
- 跨域矩阵中的 held-out 数据集单元格；
- 最坏电芯或最坏循环；
- 失败分析中的具体 PredictionRecord；
- 数据集资产中的 data-only 浏览入口。

北极星交互指标：**从任一正式结论到逐循环原始证据不超过 3 次交互。**

## 2. 文档盘点与关键输入

本次盘点覆盖 `D:\SOH_V&V` 中 22 份 Markdown、17 份 HTML、1 份 DOCX 及其配套界面图片和 mockup。DOCX 与同目录 Markdown 为同一设计文档的不同载体，分析以可检索的 Markdown 为准。

| 材料组 | 关键结论 | 对本项目的影响 |
|---|---|---|
| `Claude/phm-vv-tooling-survey-bess-blueprint-zh.html` | 产品的主要输出应是版本化证据包，而非单个算法或单一准确率 | 页面必须把结论、证据、运行域和可信度放在一起 |
| `Claude/SPEC.md`、`DECISIONS.md` | 算法契约、黄金指标、快照、manifest、摄取硬门和复现门 | 前端所有结果都应可回到 RunSpec、Snapshot 和 Artifact digest |
| `Claude/phm-vv-tooling-ui-mockup-zh.html` | 运营型工具成熟，但证据型界面仍是空白；借鉴检测到验证的闭环 | 不复制运营大屏，构建面向审查的证据工作台 |
| `Kimi_Agent_PHM测评工具综述/research/*` | PHM 工具缺少统一测评中间层；指标、V&V、算法与平台架构需共同设计 | 测评协议、执行、统计和证据不可拆成孤立模块 |
| `SOH测评工具方案设计/research/*` | SOH 排名强依赖数据划分、真值、特征预算和迁移协议 | 排行榜必须显示 comparison key、CI、coverage 和失败范围 |
| `competitor_ui/competitor_ui_analysis.md` | TWAICE 强于分布与层级钻取，ACCURE 强于判据、任务和审计 | 采用层级钻取、实测/预测/偏差、阈值区、问题闭环和审计链 |
| `docs/superpowers/specs/2026-07-14-...design.md` | 已定义 Verification Fabric、证据中心和样本曲线契约 | 本次方案将其落为可实施页面、状态和 API 边界 |

## 3. 当前割裂的根因

### 3.1 信息架构重复

当前左侧导航和顶部页签重复表达训练、推理、模型、诊断和完整看板。用户把“页面”当成工作流阶段，但同一测评任务的上下文在页签切换时被弱化。

### 3.2 只传筛选条件，没有共享证据对象

现有 `openSohDashboardDrilldown()` 能传递 dataset、model、featureset 等字段，但它本质上是页面跳转参数。系统缺少一个可持久化、可恢复的统一对象来同时标识：

```text
comparison_id + run_id + candidate_id + dataset_snapshot_id
+ asset_id + sample_id + cycle_index + evidence_kind
```

因此看板知道“看哪个数据集”，却不知道“这个循环点对当前 MAE、判据和结论贡献了什么”。

### 3.3 聚合结果与原始证据属于两条数据链

测评页主要消费排行榜和任务结果，SOH 看板主要消费同步后的 dashboard 数据。两者没有以 `PredictionRecord` 和 `MetricContribution` 建立强关联，无法证明某个汇总数字由哪些样本构成。

### 3.4 页面术语和用户任务不一致

训练/推理页以执行方式命名，完整看板以展示形态命名。评审者真正关心的是：

1. 哪个结论可比较；
2. 哪些数据域支持或反驳结论；
3. 最坏样本发生了什么；
4. 原始信号和特征是否可信；
5. 证据是否足够发布。

## 4. 目标信息架构

采用单一左侧导航，移除重复顶部页签。

| 顶层页面 | 唯一职责 |
|---|---|
| 可信总览 | 发布准备度、阻断项、签署结论和运行健康 |
| 实验设计 | 数据、候选、特征、协议、政策、预检和 RunSpec 冻结 |
| 运行中心 | 队列、attempt、事件时间线、取消、失败复跑 |
| 比较与证据 | 排行榜、跨域矩阵、逐样本误差和同步曲线证据 |
| 数据与协议 | Snapshot、QC、许可、真值、split 和 fit-scope |
| 模型资产 | ModelArtifact、Pickle manifest、schema 和冻结推理 |
| 发布与审计 | EvidencePack、签署、导出和审计日志 |

“SOH 完整看板”降级为 `比较与证据` 的专家视图，不再占用顶层导航。训练与推理仍是不同执行语义，但在 UI 中分别落到实验设计/运行中心和模型资产/跨域比较，而不是维持两个结构雷同的大页面。

## 5. 核心页面：比较与证据工作台

### 5.1 布局

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 上下文栏：Comparison / 协议 / Snapshot / Seed / 状态 / 导出        │
├──────────────┬──────────────────────────────┬───────────────────────┤
│ 候选与数据域 │ SOH 真值/预测/区间主轨迹      │ 结论与判据             │
│ 排行榜       │ 点击 cycle 联动               │ 当前误差、覆盖、QC      │
│ 跨域切片     ├──────────────────────────────┤ 谱系与审计             │
│ 电芯列表     │ V/I/T/Q、IC、DVA、特征窗口    │ 创建问题/固定证据       │
├──────────────┴──────────────────────────────┴───────────────────────┤
│ 样本贡献表：cycle / truth / prediction / error / evidence / status │
└─────────────────────────────────────────────────────────────────────┘
```

- 左栏固定宽度并内部滚动，负责比较范围和对象选择。
- 中栏是主证据区，SOH 轨迹占最大面积；下方曲线高度受控。
- 右栏是 Evidence Inspector，只解释当前选择，不重复全局 KPI。
- 底部样本贡献表固定高度、表头吸顶，支持跳转到具体循环。

### 5.2 联动语义

| 用户动作 | 同步结果 |
|---|---|
| 点击候选算法行 | 更新预测曲线、CI、逐域指标、判据和模型谱系 |
| 点击 held-out 数据域 | 限定资产列表、coverage、最坏样本和迁移结论 |
| 点击电芯 | 更新该电芯 SOH 全轨迹和样本贡献表 |
| 点击 SOH 轨迹点 | 更新 V/I/T/Q、IC/DVA、特征高亮、误差贡献和 QC |
| Shift 点击多个周期 | 最多并列 3 个循环，用于退化前后对照 |
| 点击特征高亮段 | 显示特征定义、raw/processed、平滑参数和 FeatureState |
| 点击失败判据 | 创建问题单或按相同上下文生成复跑草案 |

### 5.3 默认叙事顺序

1. **结论**：候选是否通过，和基线相比是否显著。
2. **边界**：在哪些数据域、电芯和周期失效。
3. **证据**：原始信号、派生特征、预测和真值是否自洽。
4. **谱系**：由哪个数据快照、特征状态和模型产物产生。
5. **行动**：接受、阻断、创建问题、补采数据或复跑。

## 6. 统一状态与数据契约

### 6.1 前端 EvaluationContext

```json
{
  "workspace_id": "sohbench-local",
  "comparison_id": "cmp_loco_7ds_001",
  "run_id": "run_20260730_1042",
  "candidate_id": "ridge_voltage_loco",
  "dataset_snapshot_id": "rwth_isea_public@sha256:...",
  "asset_id": "cell_04",
  "sample_id": "rwth:cell_04:cycle_287",
  "cycle_index": 287,
  "evidence_kind": "ic",
  "compare_cycles": [40, 160, 287]
}
```

该对象同步到 URL、页面 store 和离线包 manifest。刷新、分享链接和离线打开后应恢复同一证据视图。

### 6.2 证据链

```text
RunSpec
  -> DatasetSnapshot
  -> SplitAssignment
  -> FeatureState
  -> ModelArtifact
  -> PredictionRecord
  -> MetricContribution
  -> Verdict
  -> EvidencePack
```

曲线数据通过下列链接加入同一链：

```text
PredictionRecord.sample_id
  -> RawSegmentRef
  -> ProcessedSegmentRef
  -> FeatureProvenance
```

### 6.3 建议 API v2

| API | 用途 |
|---|---|
| `GET /api/v2/comparisons/{id}` | comparison key、候选、统计结论和发布状态 |
| `GET /api/v2/comparisons/{id}/domains` | 数据域、coverage、逐域指标和最坏资产 |
| `GET /api/v2/runs/{run}/assets` | 可钻取资产及其误差摘要 |
| `GET /api/v2/predictions/{sample}` | truth、prediction、interval、metric contribution |
| `GET /api/v2/samples/{sample}/curves` | 按 curve kind 和视窗分段读取原始/处理后曲线 |
| `GET /api/v2/samples/{sample}/features` | 特征窗口、定义、参数和 provenance |
| `GET /api/v2/evidence/{record}/lineage` | 完整谱系和 digest |

原始曲线按 asset/cycle 分段读取并缓存；无曲线时必须返回覆盖原因，不允许生成演示曲线冒充真实证据。

## 7. 后端融合策略

### 7.1 先建 Evidence Adapter，不重写执行器

第一阶段保留当前训练、推理和 dashboard 同步链，在任务完成后增加只读适配器：

- 将现有 run JSON 映射为 RunSpec/PredictionRecord；
- 将 dashboard 曲线索引映射为 RawSegmentRef；
- 以 dataset + cell + cycle 建立 deterministic sample_id；
- 生成 comparison summary 和 evidence coverage；
- 对缺失 manifest 的历史结果标记 `legacy_evidence`。

### 7.2 再把证据生成前移到执行阶段

新运行直接双写 v2 schema，指标引擎以 PredictionRecord 为输入，聚合值不再是独立 JSON。SOH 看板由同一证据 API 驱动，逐步停止复制静态 dashboard 数据。

### 7.3 运行状态与证据状态分离

- `run.status`：queued/running/succeeded/partial/failed/cancelled；
- `evidence.status`：draft/complete/reviewed/signed；
- `verdict`：pass/warn/fail/not_assessed。

运行成功不等于证据完整，证据签署也不等于 verdict 通过。

## 8. 渐进迁移计划

### Phase A：只读融合壳层（1 个迭代）

- 新增 `比较与证据` 页面和 EvaluationContext。
- 复用现有排行榜、推理记录和 dashboard 曲线。
- 候选、数据域、电芯和循环实现前端联动。
- 旧 `SOH 完整看板` 保留为专家模式和回退入口。

退出门：任一结果行 3 次内到曲线；刷新可恢复选择；历史结果有明确 coverage。

### Phase B：Evidence API 与双写（1–2 个迭代）

- 增加 sample_id、PredictionRecord、MetricContribution 和曲线索引。
- 新旧结果自动核对；失败、partial 和缺失证据语义固定。
- 支持最坏样本、失败样本和特征窗口钻取。

退出门：汇总指标可由逐样本贡献重算；缺失曲线原因可量化；无伪证据。

### Phase C：比较实验室收口（1 个迭代）

- 算法推荐并入同口径比较视图。
- 训练和推理页只负责设计/执行，结果统一回到比较与证据。
- 移除重复顶部页签，完整看板退出顶层导航。

退出门：每页一个主决策；导航无重复；训练/推理结果使用同一证据视图。

### Phase D：可信发布（后续）

- EvidencePack 签署、SBOM、审阅批注、问题单和复测闭环。
- 离线包保存聚合结果、逐样本索引、固定证据曲线和完整 manifest。

## 9. SDD/TDD 验收规格

### Contract tests

- 相同 dataset/cell/cycle 始终生成相同 sample_id。
- PredictionRecord 必须引用 DatasetSnapshot、FeatureState 和 ModelArtifact。
- 聚合 MAE/RMSE 可由 MetricContribution 重算，误差在容差内。

### Interaction tests

- 点击候选行后，轨迹、判据和谱系 candidate_id 一致。
- 点击轨迹点后，cycle、曲线、特征和样本表高亮一致。
- URL 重载后恢复 comparison/run/asset/cycle。
- 无曲线时显示原因、覆盖率和可用替代证据。

### Visual tests

- 1440×900 首屏同时看见候选列表、SOH 主轨迹和证据检查器。
- 1280×720 无横向溢出；底部区域高度不挤压主轨迹。
- 768px 下右侧检查器变为抽屉；375px 下按结论、轨迹、证据顺序堆叠。
- 长候选和样本列表固定高度、内部滚动、表头吸顶。

### Scientific integrity tests

- 按 asset 分组切分，不允许同一电芯跨训练/测试泄漏。
- comparison key 不一致的候选不得进入正式排名。
- raw/processed 曲线和特征高亮可追溯到处理参数。
- 演示数据、快照数据和实时 API 数据有不可混淆的来源标识。

## 10. 优先级建议

建议立即实施 Phase A，而不是继续扩展独立看板。Phase A 改动集中在前端状态、只读适配器和路由，能以较低风险验证新的核心工作流；其交互和数据缺口会直接反推 Phase B 的证据 API，避免先建设一套无法服务评审任务的后端。

## 11. 第二轮 Mockup 验证结果

`docs/mockups/soh-evaluation-evidence-workbench.html` 已补充下列竞品对标能力：

- 候选算法 × held-out 数据域跨域热力矩阵，点击单元格同步全部证据上下文；
- 当前电芯与全部 held-out 电芯的误差分布对比；
- 最多三个循环的 IC、电压、温度和特征窗口叠加对照；
- 逐样本覆盖、原始曲线覆盖和数据快照新鲜度提示；
- 发现、指派、复测、关闭的问题闭环；
- EvidencePack 固定和加入动作。

交互验收覆盖矩阵联动、分布切换、多周期上限、问题状态推进和证据谱系更新。响应式验收覆盖 1440、1280、1024、768 和 375px，页面无横向溢出；窄屏时命令栏与主工作区按任务顺序堆叠，矩阵和周期选择在各自面板内滚动。
