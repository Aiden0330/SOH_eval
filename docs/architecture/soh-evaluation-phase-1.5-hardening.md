# SOH 评测案例工作台：阶段 1.5 加固说明

## 交付目标

阶段 1.5 不扩展新的评测算法，重点修复阶段 1 架构审查中会阻碍继续开发的边界问题：

1. 本地 API 不再接受任意浏览器来源；写操作必须携带本次服务会话令牌。
2. 上传 Python、反序列化 Pickle、训练、推理、导入和打包等旧执行入口默认关闭。
3. Evaluation Case 由 SQLite/WAL 和追加事件作为事实源，JSON 仅保留为兼容投影。
4. 案例通过稳定的 `idempotency_key` 标识一次评测执行或比较范围，不再与展示 profile 一一绑定。
5. Evidence Coverage 与 Verdict 只由服务端根据本地可信产物派生，前端不能提交或覆盖。
6. 数据集、候选模型、电芯、Cycle 和验收口径的浏览状态属于临时 ViewContext；用户必须显式点击“固定当前证据”才写入案例。
7. 并发更新必须提交 `expected_version`；冲突时返回 409，由用户确认后重新固定，不自动覆盖。

## 运行边界

默认启动即为安全模式：

```powershell
$env:PYTHONPATH='D:\SOH_eval\src'
python -m sohbench.api_server --host 127.0.0.1 --port 8001 --data-dir D:\SOH_eval\data --work-dir D:\SOH_eval\work\trusted_plugins --runs-dir D:\SOH_eval\runs --frontend-data-dir D:\SOH_eval\frontend\app\data
```

默认允许的界面来源只有：

- `http://127.0.0.1:8765`
- `http://localhost:8765`

如确需在完全可信的本机环境执行旧上传、训练、推理、导入和打包流程，必须显式增加：

```powershell
--enable-legacy-execution
```

也可以重复使用 `--ui-origin` 精确添加其他允许来源。不要把 API 监听地址改为公网地址。

## 案例数据架构

持久化目录位于 `work/trusted_plugins`：

- `evaluation_cases.sqlite3`：案例当前投影、追加事件、迁移错误。
- `artifacts/sha256/<前缀>/<摘要>`：每个案例版本的内容寻址快照。
- `cases/EV-*.json`：原有消费者继续可读的 JSON 投影，不再是事实源。

一次有效更新会在同一个 SQLite 事务中完成：

1. 校验 `expected_version`。
2. 生成新案例快照及 SHA-256 摘要。
3. 使用 compare-and-swap 更新当前投影。
4. 追加 `case.created`、`evidence.pinned` 或生命周期事件。
5. 事务提交后刷新 JSON 兼容投影。

旧 `1.0.0` JSON 会迁移为 `1.1.0` 并保留原案例 ID；无法识别的版本不会静默消失，而会写入迁移错误表并通过 `/api/v2/case-migration-errors` 查询。

## 证据与判定边界

前端固定证据时只提交引用：

- candidate、dataset、asset、cycle、evidence tab；
- criterion set；
- expected version、actor 和 reason。

服务端根据 run config、leaderboard、profile manifest、诊断场景、曲线索引和模型产物派生：

- 数据快照摘要；
- 切分协议摘要；
- 比较范围兼容性；
- 精确预测记录；
- 原始曲线摘要与可用信号；
- 8 项证据覆盖；
- strict / engineering / exploratory 判定。

如果模型产物、比较范围、精确 Cycle 或原始曲线缺失，服务端会返回 `incomplete` 或 `pending`，不会采纳前端预览结论。

## 前端行为

- 案例固定前显示“本地预览 · 未固定”。
- 浏览状态保存在当前浏览器标签页的 `sessionStorage` 中，不修改案例。
- 点击“固定当前证据”后显示服务端证据链与服务端判定。
- 刷新页面会恢复同一案例和已固定状态。
- 409 冲突会读取最新案例并提示用户确认，不会自动重试覆盖。
- 安全模式下，旧上传、训练、推理、导入、同步和打包按钮禁用；案例浏览与 v2 状态接口保持可用。

## 验证结果

- Python 全量测试通过，保留 2 项既有跳过项。
- 前端 3 组 Node 测试通过。
- Python 编译检查与 JavaScript 语法检查通过。
- 浏览器验证通过：安全模式、按钮禁用、显式固定、服务端派生、事件追加和刷新恢复均按预期工作。
- 实际案例验证显示 3/8 证据、Verdict 为 pending；这是当前数据边界的真实结果，而不是界面故障。
