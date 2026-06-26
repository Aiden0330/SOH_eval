# 路由日志 — SOH 方案 + soh_bench spec 重评估

**日期**：2026-06-26
**触发需求**：用最大分析深度重新评估并再设计「SOH泛化性增强分层方案.md」与 soh_bench 测评框架 spec。

## 任务画像（五问）

| # | 问题 | 答案 | 依据 |
|---|---|---|---|
| Q1 | 任务类型 | 预测（SOH 估计 / 泛化增强） | 方案 md 全篇为 SOH 预测 |
| Q2 | 数据源统一 | benchmark 范围已统一（processed 17 维特征表）；上游多厂商 BMS 未统一，需 L1/data_unify 前置 | data/README + processed/*.pkl 同 schema |
| Q3 | 标签可得性 | 退化标签（容量标定 SOH）；部分场站数据（RWTH/M5BAT）无容量标签 → 不可监督 | data/README §已登记未接入 |
| Q4 | 模型族/训练约束 | 限传统 ML（非 DNN）+ 目标场景全量重训 + 仅容量侧（禁内阻） | 方案 md 约束基线 |
| Q5 | 部署形态 | 边缘 C++ 纯推理（Python 训练 → C++ 推理） | 方案 md 约束基线 |

## 路由结论

- 主路由：**soh-adapt-v2**（Q4 限传统ML/全量重训 + Q5 纯推理C++）。
- 门控：复用 soh-adapt 1.0 的 `eval_harness`，按预测任务注入指标集（不自立第二套验收）。
- 部署：追加部署导出基座（cpp_export），门控追加 Python/C++ 对拍项。
- 版本冲突裁决：方案 L3 拟把 soh-adapt v1 的「冻结+微调」作为一种基学习器 → 同时命中 1.0 与 2.0。
  按裁决规则：数据统一 + 重训用 2.0，验证门控统一用 1.0 的 eval_harness；不允许两版各长一套验收。

## 关键发现（供评估展开）

1. **soh_bench == 验证门控基座**：phm-router 技能地图明确「验证门控暂由 eval_harness 提供，待升格 phm-eval-harness，按任务注入指标集」。soh_bench spec 就是这一基座，其契约应对齐 eval_harness 三门，而非另立标准。
2. **红线对齐**：phm-router 第 4 节统一红线（禁随机切分 / 不确定度不可省 / 未注册字段不入特征层 / 必过门控）必须在 spec 中逐条显式落位——当前 spec 命中「禁随机切分」「不确定度可选」，但「字段注册」「门控强制」需加强。
3. **标签可得性缺口**：方案与 spec 都未显式处理「无容量标签场站」的排除逻辑——会让 leave-one-station-out 在无标签场站上静默产出伪指标。

## 门控结果回填

- 重评估完成（2026-06-26）：产出 [`2026-06-26-soh-reeval-and-redesign.md`](../docs/superpowers/specs/2026-06-26-soh-reeval-and-redesign.md)。
- 设计审查结论：发现 P0×4 / P1×5 / P2×3。其中 P0 三项（CI 强制、真三门缺位、数据QC门缺位）违背基座红线 R5/R7，
  且 NASA/CALCE 标签越界（soh∈{0.0,1.22}）已实测坐实。
- 路由有效性自评：路由到 soh-adapt-v2 正确——正是其 eval_harness/data_unify/features 契约暴露了 spec 偏离，
  说明本次"先路由再评估"避免了在脱离基座的前提下空评。
- 修订项是否过三门设计审查：**待用户拍板第 6 节四项决策后**，据此修订 spec v2 再过审。
