# Agent 引导的算法/特征/参数搜索 · 设计（训练期 + 适配期）

> 2026-08-26 · 状态：设计定稿 + 训练期原型已交付（`sohbench.agent_search`）
> 关联：`SOH adaption multi-layer method.md`（L1–L6 分层）· 标签安全 v1.1 文档 §6 ·
> B3 FastAPI/OpenAPI 契约（agent 工具面）

## 0. 动机与核心论点

固定全局设置（如部分窗口电压界）对多数据集必然失配：同为 LFP，hust（5C 快充）
的合规窗口是 [3.30, 3.52] V，snl_lfp（0.5C）是 [3.30, 3.38] V —— 本轮窗口重建
就是一次"人肉 agent"逐数据集扫描的产物。把这类选择交给搜索循环是正确方向。

**核心约束（决定成败的一条）：搜索必须在验证织物内部进行，而不是绕过它。**
反例即本仓库的教训：旧 hust 窗口覆盖 98% 容量、验证 MAE 近乎完美 —— 一个
无约束的优化器会**选中**泄漏窗口，因为"泄漏"正是最优验证分数的形态。因此：

1. **门禁是硬约束**：仿射重构检测、DOD < 90% 窗口规则、标签来源列结构性禁用、
   组合预检 —— agent 不可将其作为可交换的代价项，被拒绝的候选直接出局。
2. **嵌套评估**：内环（搜索可见）与外环（结论声明）严格分离。外环数据集在
   搜索全程不可见，仅在最终 top-k 上一次性评测（缓解 winner's curse；top-k
   全量披露，不只报冠军）。
3. **适配期动作受限**：推理/适配阶段仅允许使用部署时合法可得的信息 ——
   目标域**无标签**特征统计（CORAL 对齐、重归一化）、原始曲线形态
   （按目标覆盖率 40–50% 重定窗口界）、以及少量 RPT 标注下的两参数输出重标定。
   适配决策与特征同样过标签安全审计（窗口重定所参考的容量类聚合是低容量
   信息通道，但不是零通道 —— 预登记，产物审计）。
4. **全部决策入 manifest 溯源**：与 `training_featuresets` 同型 —— agent 选定的
   窗口界、特征组合、适配算子写入模型/运行 manifest，供证据工作台审计与
   将来的失效清算（invalidation sweep）。

## 1. 搜索空间

| 阶段 | 维度 | 说明 |
|---|---|---|
| 训练期 | 算法 | 注册表内 estimator（ridge / extra_trees / gbdt / mlp / …） |
| 训练期 | 特征组合 | 无泄漏候选池（partial_window / ic_dva / degradation_kinetics / …）的子集组合 |
| 训练期 | 超参数 | 各算法小网格；连续参数交给经典内层优化器（贝叶斯/逐次减半） |
| 训练期 | 窗口界 | `q_win` 电压窗口 [lo, hi] —— 需先完成"窗口参数化改造"（§3） |
| 适配期 | 表征对齐 | CORAL 开/关；特征重归一化（除以 C_rated、逐电芯首循环基准） |
| 适配期 | 窗口重定 | 在目标域原始曲线上按覆盖率带（40–50%）重求 [lo, hi]，无标签 |
| 适配期 | 输出重标定 | 有少量 RPT 时的 slope/intercept 两参数校正 |

**分工**：LLM 型 policy 负责结构性决策（读 QC 报告、诊断"预测 SOH 129% →
尺度漂移 → 提议归一化"、LFP 弃 ICA 取 DVA）；连续超参交给经典优化器作为
policy 可调用的工具。

## 2. 架构

```
Policy.propose(history) ──候选──▶ 约束 oracle（training_preflight）
                                      │ blocked → 出局（原因入报告）
                                      ▼
                            内环评分（inner datasets · LOCO）
                                      │ 按最差数据集 MAE 排名
                                      ▼
                            Policy 迭代（可提新候选/收敛）
                                      ▼
                            外环声明（outer dataset · LODO，单次）
                                      ▼
                            报告 + manifest 溯源
```

- **约束 oracle** = `EvaluationService.training_preflight`：一次调用同时给出
  标签安全违规、窗口 DOD 违规/警告、列覆盖与协议可行性。
- **agent 工具面** = B3 的 FastAPI/OpenAPI 契约：preflight、jobs、artifacts、
  verdicts 即 LLM agent 的 typed tools；本原型直连服务层等价物。
- **Policy 协议**：`propose(history) -> list[Candidate]`，返回空列表即收敛。
  原型交付 `GridPolicy`（一轮全网格）；LLM policy 为后续接入点。

## 3. 窗口参数化改造（后续工作）

现状 `q_win_ah` 在数据预处理期固化。改造：窗口积分下沉为**参数化 transform**，
训练时读取原始曲线（复用证据侧 on-demand raw 提取器机制）按 [lo, hi] 现算；
窗口界随 `training_featuresets` 一并写入 manifest；推理期按 manifest 复算。
在此之前，训练期搜索的窗口维度以"预生成窗口栅格列"近似（q_win@W1, q_win@W2 …）。

## 4. 原型交付范围（本次）

`src/sohbench/agent_search.py` + `tests/test_agent_search.py`：

- `GridPolicy` 全网格提案；preflight 硬门禁（blocked 候选带原因出局）。
- 内环：逐 inner 数据集 LOCO，按 `mae_worst`（最差数据集均值）排名。
- 外环：top-k 在 inner+outer 池化表上 LODO，仅提取 held_out == outer 的折 ——
  外环数据在搜索中从未参与评分。
- 报告 JSON：admitted / excluded(+原因) / ranking / outer_claims / integrity 声明。
- LLM policy、超参内层优化器、适配期动作为后续迭代。

## 4.5 窗口参数化 v1 交付与实验结论（2026-08-27）

交付：`sohbench.raw_windows`（四数据集原始充电曲线统一迭代器，calce 指纹对齐内置）、
`sohbench.window_bank`（0.05 V 格点 78 窗/循环预积分，parquet 存储；
`select_bounds` 无标签选界；`apply_bank_window` 替换 + 复审计）、
`scripts/build_window_bank.py`、agent_search 的 `window_target` 维度
（候选形如 `ridge×partial_window_norm@w45%`，逐数据集选界写入报告溯源）。

**实测结论（内环 calce+snl · 外环 hust 单次）**：
1. 仅按覆盖率选界会选中拐点/边缘窗（同覆盖率窗口多解，snl@30% 选到平台以下
   [3.00,3.35]）→ 已加入**电荷密度偏好**（fraction/V，即平台区，仍无标签）。
2. 密度修正后调优窗仍逊于手工标定窗（worst 0.128 vs 0.078；外环 0.166 vs
   0.089）：0.05 V 格点无法表达实测最优的 0.07–0.08 V 亚格点窗
   （snl [3.30,3.38]、uconn [3.30,3.37]）。
**结论**：手工逐数据集标定窗保持当前最优；调优维度成为可用研究仪器但需
细化（平台邻域 0.01–0.02 V 格点，或直接以边界为搜索维度）后才可能反超。
负结果由嵌套协议诚实计量 — 这正是本装置的目的。

## 5. 验收与风险

- 验收：单测（tiny 数据集全流程 + 门禁出局路径）；一次真实 LFP 搜索
  （inner = calce_cs2 + snl_lfp，outer = hust）产出报告。
- 风险：
  - winner's curse —— 外环单次 + top-k 全披露 + 报告 integrity 字段缓解；
  - 适配期信息通道 —— 每类适配算子预登记其输入信息面，产物过推理门禁；
  - 计算预算 —— 内环逐次减半/早停，policy 有轮次上限。
