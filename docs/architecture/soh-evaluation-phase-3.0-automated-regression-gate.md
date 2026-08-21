# SOH Evaluation 阶段 3.0：自动回归门禁

更新时间：2026-08-14

## 1. 阶段目标

阶段 3.0 不增加新的页面流程，而是把阶段 2.4 已贯通的真实案例流水线固化为发布前自动门禁。一次执行同时验证正向闭环与负向阻断；只有每项“实际结果”都符合“预期结果”，命令才返回成功。

## 2. 门禁矩阵

| 检查 | 预期 | 判定重点 |
| --- | --- | --- |
| 真实训练到签署闭环 | 通过 | 预检、训练、推理、逐 Cycle 事实、证据、判定、复测、KPI 对账和 EvidencePack 全部完成 |
| 刷新及服务重载 | 通过 | 重新实例化服务后仍解析到同一案例、同一任务、同一代表样本和同一 EvidencePack |
| EvidencePack 与审计事件完整性 | 通过 | manifest 摘要、内容寻址副本、必需事件和篡改检测均有效 |
| 不兼容范围 | 阻断 | 数据集不属于比较范围时，split、预测事实和发布签署均不能伪造为通过 |
| 必需证据缺失 | 阻断 | 原始曲线缺失时证据必须为 incomplete，EvidencePack 不可签署 |

负向场景复制真实 profile 到临时隔离目录运行，结束后自动清理；不会修改正向案例、模型包、签署包或工作台数据。

## 3. 一键执行

在仓库根目录运行：

```powershell
$env:PYTHONPATH='D:\SOH_eval\src'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_phase30_release_gate.py
```

默认使用 `NASA × ridge × full × LOCO × seed 0`，留出对象为 `B0018`。命令退出码为 `0` 表示整组门禁通过，`1` 表示至少一项与预期不符。

## 4. 输出

- 最新机器报告：`work/trusted_plugins/phase30_release_gate_report.json`
- 最新中文摘要：`work/trusted_plugins/phase30_release_gate_report.md`
- 历史归档：`work/trusted_plugins/release_gates/phase30/GATE-*.json|md`
- 工作台 Profile：`phase30-gate-<dataset>-<timestamp>`

报告包含门禁 ID、案例 ID、每项预期/实际结果、关键证据、运行用时、报告摘要指纹和可直接打开的审计页面。即使正向流水线失败，也会生成结构化失败报告，供本地或持续集成环境读取。

## 5. 发布规则

发布脚本只需读取 `status`：

- `passed`：所有正向检查通过，所有负向场景按预期被阻断；
- `failed`：至少一项正向检查失败、负向场景未被阻断，或门禁自身执行异常。

门禁不会把“预期阻断”记为失败。例如原始曲线缺失案例成功阻止签署时，该检查结果为通过。

## 6. 验证边界

- 当前 EvidencePack 验证的是本地审计 manifest、SHA-256 摘要与内容寻址副本；还不是外部私钥或证书签名。
- 阶段 3.0 保留现有工作台交互，不新增审批按钮；发布控制从人工点击前移到自动门禁。
- 去除剩余 iframe 已由阶段 3.1 完成；阶段 3.2 已根据产品优先级调整为“完成组合库与泛化验证”。正式密钥签名改为独立的后续安全阶段，不混入本门禁实现。

## 7. 本轮验收记录

2026-08-14 23:26（America/New_York）完成真实基线执行：

- 门禁 `GATE-20260815-3D5FF1A3`，状态 `passed`，5/5 检查符合预期，用时 6.294 秒；
- 案例 `EV-20260815-82F2263E` v4，Profile `phase30-gate-nasa-20260815-032644`；
- 训练 32 个模型，最终选择 `ridge_full_loco_nasa_B0018_seed0`；
- 推理产生 102 条 PredictionRecord，代表样本 `nasa:B0018:77`；
- EvidencePack `EPK-20260815-7917CDC2` 已完成本地摘要签署；
- 复测 `RET-20260815-9D087CD9` 与对账 `REC-20260815-89B51C31` 均通过，KPI 差异为 0；
- 报告摘要 `sha256:3e83cf23dcc5368299d53fc79e29b9ebbe3734e435800a7dcb5a5f7f69dc1b15` 已独立复算一致。

验收覆盖 48 项后端定向测试、4 组前端回归，以及浏览器中的训练、推理、评测三页同案例一致性检查；这不是整个仓库测试全集。浏览器还验证了 1280×720 布局无重叠，并将嵌入式原始曲线看板限制为数据集、电芯、Cycle 与原始曲线可用性的定位源。预测值、真值和区间只从当前案例的规范 PredictionRecord 读取，避免辅助看板的内置模型数值覆盖发布判定。

验收页面：`http://127.0.0.1:8765/SOH_eval/frontend/app/index.html?profile=phase30-gate-nasa-20260815-032644&mode=audit&v=phase30-regression-gate-20260814c`
