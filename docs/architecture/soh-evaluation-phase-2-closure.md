# SOH 评测案例工作台：阶段 2 闭环与对账

## 本阶段交付

阶段 2 已启动并落地第一条可执行竖切，目标不是增加一个独立的“审计页面”，而是把评测结论继续向发布闭环延伸：

```text
固定证据 / 判定
  → 问题单
  → 回归集
  → 同协议复测
  → KPI 对账
  → 人工评审
  → EvidencePack 草稿 / 受控签署
```

前端“发布与审计”现在是案例工作台的第 07 阶段和真实视图；它与设计、推理、比较与证据共享同一个 Evaluation Case ID、案例版本、候选、数据集、电芯、Cycle、判据和证据覆盖状态。

## 后端事实模型

阶段 2 在阶段 1.5 的 SQLite/WAL、内容寻址存储和案例 compare-and-swap 机制上新增：

| 实体 | 关键约束 |
|---|---|
| ReviewIssue | 创建时固定案例版本与证据上下文；不能静默覆盖 |
| RegressionSet | 必须引用问题单和原协议；成员必须来自已固定的精确预测记录 |
| RetestRequest | 必须引用原案例、问题单、回归集；协议变更必须说明原因 |
| KPIReconciliation | 服务端读取案例事实与旧 dashboard JSON，按字段与容差对账 |
| EvidencePack | 统一引用案例、候选、数据快照、协议、样本、问题、回归集、复测和对账 ID |
| WorkflowEvent | 追加写入；事件负载写入 SHA-256 内容寻址存储 |

案例事件与闭环事件在发布页合并显示，但仍保留各自事实源：案例版本变化进入 `case_events`，问题、复测、对账与证据包进入 `workflow_events`。

## API

```http
GET  /api/v2/cases/{case_id}/review-workspace
POST /api/v2/cases/{case_id}/review
POST /api/v2/review-actions/issues
POST /api/v2/review-actions/issues/{issue_id}/status
POST /api/v2/regression-sets
POST /api/v2/regression-sets/{set_id}/members
POST /api/v2/cases/{case_id}/retests
POST /api/v2/cases/{case_id}/reconcile
POST /api/v2/cases/{case_id}/evidence-packs
POST /api/v2/evidence-packs/{pack_id}/sign
```

所有写入都提交当前案例版本。版本不一致返回 409，前端重新读取服务端状态，不自动覆盖其他窗口的变更。

## 发布门禁

EvidencePack 草稿可以在门禁未通过时生成，用于暴露缺失项和留存审计快照；签署必须同时满足：

1. 来源为实测数据；
2. 执行已完成；
3. 必需证据完整；
4. 判定为通过或有条件通过；
5. 人工评审已批准；
6. 所有问题已闭环且没有未完成复测；
7. 案例事实表与旧 dashboard KPI 对账一致。

任一条件失败时，服务端拒绝签署。前端仅展示服务端返回的门禁状态，不自行推断可签署性。

## 当前真实案例状态

浏览器验收使用 `battery-soh-diagnostics` 案例。它正确显示为“受控阻断”：当前执行状态、精确预测记录、原始曲线、不可变模型产物和可对账 KPI 尚不完整，因此可以创建问题、回归集和 EvidencePack 草稿，但不能把案例签署为已验证。

这不是演示性禁用；问题单、回归集、对账结果、EvidencePack 草稿和事件摘要均已写入本地事实库。

## 已验证场景

- 完整夹具：问题单 → 回归成员 → 同协议复测 → 问题闭环 → KPI 一致 → 评审批准 → EvidencePack 签署；
- 缺失夹具：证据不完整或评审未批准时，签署被服务端拒绝；
- API：闭环工作区可读取，问题和回归集可创建，过早签署返回 400；
- 浏览器：第 07 阶段真实切换；问题单、回归集、不可对账结果和草稿状态在刷新后保留；
- 1280×720：三栏工作台可用，内容在列内滚动，表单无横向溢出。

## 阶段 2.1：受控执行器（已完成）

`POST /api/v2/cases/{case_id}/retests` 现在返回 `202 Accepted`，控制面自动分配 `RTR-*` 运行 ID，并按 `requested → running → completed/failed` 写入事件与结果。前端轮询工作区并自动回填，不再要求评审人输入既有 `run_id`，也不暴露手工“完成复测”入口。

执行器采用两条显式分支：

| 分支 | 条件 | 结果语义 |
|---|---|---|
| `model_reinference` | 问题单固定的 `model_artifact_id` 能在本地模型仓库按内容摘要复验，且固定数据行仍可定位 | 重新加载不可变模型与数据快照、重新调用预测、回填指标；允许完成复测门禁 |
| `evidence_replay` | 缺少或无法复验可执行模型产物 | 仅校验并重放冻结预测快照；结果带 `prediction_recomputed=false`，状态保持失败，不能闭环或签署 |

模型产物 ID 由模型清单与 pickle 字节联合计算为 `model@sha256:*`，加载时重新计算摘要；数据快照优先绑定 `processed/*_features.pkl`，证据重放另外固定 `prediction-snapshot@sha256:*`。因此“缓存预测重算”不会被包装成“模型复测成功”。

## 后续增量

下一步推进相邻样本批量加入回归集、原 run / retest 指标贡献对比，以及为历史案例补齐无歧义的模型产物绑定。
