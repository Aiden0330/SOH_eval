# 阶段 3.1.1：生产稳定化

## 结论

阶段 3.1.1 已把案例生命周期、Profile 发布、运行中刷新和自动回归门禁收敛为可重复的生产默认行为。后端启动后，由另一个进程新增 Profile、训练任务、推理任务和案例时，无需重启即可被同一 API 进程读取；未进入正式流程的案例默认是草稿，归档后保留审计记录但不再污染默认工作列表。

## 稳定化边界

### Profile 与案例热刷新

- `GET /api/profiles` 每次合并静态索引和 `profiles/*/manifest.json`，新发布目录立即可见。
- Profile 产物、manifest 和 index 使用临时文件原子替换，避免页面读到半写入 JSON。
- Profile 发布使用跨进程锁串行更新索引，避免并发发布发生“最后写入者覆盖前一项”。
- 任务 JSON 和案例 SQLite 仍以磁盘为权威源；每次查询重新读取，因此训练、推理和案例都支持运行中发现。
- 前端优先读取动态 Profile API；API 不可用时回退静态 `profiles/index.json`，保留离线浏览能力。

### 生命周期默认策略

| 场景 | 默认状态 | 后续行为 |
| --- | --- | --- |
| 只有 Profile 和幂等键，尚未运行 | `draft` | 不进入正式评审，可在“更多”中归档 |
| 出现训练/推理任务、运行标识、候选模型或固定证据 | `active` | 草稿随同一次不可变事件自动激活 |
| 正式完成或失败 | `completed` / `failed` | 遵循既有受控状态转换 |
| 放弃草稿或失败案例 | `archived` | 保留案例、投影和事件，默认列表隐藏 |

`GET /api/v2/cases` 默认排除归档案例；审计或历史查询可显式传入 `include_archived=1`，也可按 `lifecycle_state=archived` 查询。归档不是删除，案例仍可按 ID 读取并核验事件摘要。

### 操作收敛

- 顶部仍只有一个主评审动作。
- “归档未完成案例”只在 `draft` 或 `failed` 状态显示，并收进既有“更多”菜单。
- 已完成、已归档或已签署案例不再显示可误触的评审主动作。
- 第三页继续直接显示数据集、算法和特征组合；本次真实案例显示“三页组合一致”。

## 自动门禁

本地和 CI 共用入口：

```powershell
$env:PYTHONPATH='D:\SOH_eval\src'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_phase311_production_stability.py
```

门禁覆盖：

1. 当前仓库 `sohbench` 导入边界；
2. 全量 Python 回归，其中包含案例生命周期、Profile 原子发布、运行中热刷新和 API；
3. Phase 3.0 发布门禁单元回归；
4. 前端核心状态、三页组合一致性和中文界面契约；
5. 原生证据画布与真实黄金案例契约。

CI 工作流位于 `.github/workflows/phase311-production-stability.yml`，使用 Windows、Python 3.11 和 Node 20。CI 上传 JSON 与 Markdown 门禁报告。该确定性门禁不会伪造 `D:\battery_SOH`；依赖实测数据的 Phase 3.0 发布门禁仍在受控本机或自托管发布环境单独运行。

## 2026-08-15 验收记录

### 确定性门禁

- 门禁：`P311-20260815T134355Z-6C501324`
- 结果：6 / 6 通过
- Python：313 项完整回归通过
- 前端：4 组契约测试通过
- 已知提示：6 条化学体系单数据集混杂警告，均来自既有协议防误判机制
- 报告：`work/trusted_plugins/phase311_production_stability_report.json`

### 实测发布门禁与热刷新

- 实测门禁：`GATE-20260815-009F727D`
- 结果：5 / 5 通过
- Profile：`phase30-gate-nasa-20260815-133424`
- 案例：`EV-20260815-50ACC664`，版本 4
- 训练：`20260815T093424-3c94b3de`，成功
- 推理：`20260815T093427-2514d8a8`，成功
- 证据：8 / 8 完整，EvidencePack 已签署
- 热刷新：Profile、案例、训练任务和推理任务均由已运行的同一 API 进程即时读取，未进行二次重启

### 浏览器可见验收

- 新 Profile 可直接打开，第三页显示 NASA / ridge / full。
- “设计与训练、推理验证、评测与证据”三页均显示完成，第三页显示“三页组合一致”。
- 原生 V / I / T / IC / DV 证据可见，无 iframe，控制台无错误。
- 已签署案例仅保留“更多”和禁用的“已签署发布”，草稿归档动作不会出现在正式签署案例上。

验收页面：

`http://127.0.0.1:8765/SOH_eval/frontend/app/index.html?profile=phase30-gate-nasa-20260815-133424&mode=audit&v=phase311-production-stability-20260815a`

## 未扩大的边界

- GitHub 托管 CI 工作流已创建并完成本地门禁验证，但本次没有声称远端 Actions 已实际运行。
- EvidencePack 当前仍是本地审计签署摘要；外部证书、密钥托管和法规签名属于后续阶段。
- 归档是可审计生命周期转换，不是数据删除或保留策略清理。
