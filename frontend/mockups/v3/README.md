# SOH V3 · 三方向前端 mockup

2026-08-19 · 同一屏（案例 `EV-20260819-8C21A4` 的可信总览）在三个设计方向下的高保真静态 mockup，
用于 V3 方向抉择。方法学来源：`SafetyBench/survey` 下两份 survey（AutoML agents · 安全风险测评优化）
与 SafeBench PROTO v2.1；能力范围保持在 SOH 域内（高估率口径、技能分 vs 在役基线、段位/噪声带、
自评→私榜差列、代理起草-人签发、证据双签）。

## 文件

| 文件 | 内容 |
|---|---|
| `sohbench-v3-directions.html` | **直接双击打开** — 自包含双页画布：第 1 页三方向抉择，第 2 页方向二全站 10 页；可平移缩放、导出 PNG/PDF |
| `Main.dc.html` | 方向一 · 实验室续章（Lab Light V3，沿用现有 Evidence Bench token） |
| `OpsAmber.dc.html` | 方向二 · 控制室同源（SafeBench 控制台 Direction-A 设计系统原样应用） |
| `AuditHall.dc.html` | 方向三 · 审计厅签发流（安全论证文档：主张—论证—证据 / AMLAS） |
| `Overview/Design/AgentOrch/Runs/Compare` `.dc.html` | 方向二全站上排：01 可信总览 · 02 实验设计 · 03 代理编排 · 04 运行中心 · 05 比较实验室 |
| `Generalization/Failures/DataProto/Models/Evidence` `.dc.html` | 方向二全站下排：06 跨域泛化 · 07 失败分析 · 08 数据与协议 · 09 模型与推理 · 10 证据中心（内嵌方向三签发视图） |
| `canvas.json` | 双页画布布局、方向批注与全站导读 |

线上可编辑版本（同一内容，改动会保存为新版本）：
https://claude.ai/code/artifact/d6e3db85-c82b-46b0-8051-05e9c36a9a23

推荐：方向二作 V3 工作台外壳（SafeBench 侧已有交接规格书，设计风险最低、方法学可见度最高），
方向三的文档形态专用于证据中心的签发/报告视图；若延续性优先则选方向一。
