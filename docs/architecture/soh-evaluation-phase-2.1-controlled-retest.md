# SOH 评测案例工作台：阶段 2.1 受控复测与 Mockup 融合

> 完成日期：2026-08-12  
> 目标：按 `soh-evaluation-deep-fusion-v3.html` 的三栏证据工作台更新阶段 2 界面，并让复测从人工登记升级为受控异步运行。

## 交付结论

阶段 2.1 已形成一条可验收竖切：

```text
案例上下文常驻
  → 比较范围 / SOH 轨迹 / 判定检查器同屏
  → 固定精确 Cycle
  → 问题单与回归集
  → 202 异步受控运行
  → 模型重新推理或证据重放
  → 结果自动回填
  → 闭环与发布门禁
```

## 与 V3 Mockup 的对齐

- 204 px 深色 `Evidence Bench` 侧栏；
- 案例面包屑、复制证据链接、EvidencePack 与评审动作常驻顶部；
- 5 阶段轨道与 6 项上下文条连续呈现；
- 左栏统一候选、工况切片与代表电芯；
- 中栏原生绘制 KPI、SOH 真值/预测/区间轨迹及逐 Cycle 详情；
- 右栏合并判定、证据覆盖、血缘、签署门禁和审计事件；
- 点击轨迹点同步当前证据上下文，固定 Cycle 后直接进入问题—回归—复测闭环。

旧 iframe 看板不再占据阶段 2 主视图，测评结论与详情证据使用同一 `EvaluationCase`、同一选择状态和同一服务端事实源。

## 受控运行合同

请求：

```http
POST /api/v2/cases/{case_id}/retests
202 Accepted
```

控制面自动生成 `RTR-*`，保存队列记录并在后台运行。工作流事件为：

```text
retest.requested → retest.running → retest.completed | retest.failed
```

结果固定包含：`runner`、`execution_mode`、`prediction_recomputed`、协议引用、模型产物、样本记录、指标、判据检查、来源产物和 `result_digest`。

### 模型重新推理

只有当问题单固定的 `model@sha256:*` 在仓库中存在且摘要复验一致时，执行器才加载原始 `processed/*_features.pkl`、应用模型特征变换并重新调用 `predict`。这条分支写入 `controlled_model_reinference/v1`，可以把复测状态置为 `completed`。

### 冻结证据重放

历史案例若没有无歧义的可执行模型产物，执行器只校验 `prediction-snapshot@sha256:*` 并重算冻结记录的 MAE/PICP。结果明确写入 `controlled_evidence_replay/v1` 与 `prediction_recomputed=false`，工作流状态保持 `failed`；界面显示“证据重放已完成 · 复测门禁仍阻断”。

## 验收边界

- 禁止评审人手工提交 `run_id` 或伪造完成结果；
- 数据或预测快照摘要变化时失败关闭；
- 缺少不可变模型产物时允许诊断性重放，但不允许问题闭环；
- 发布签署仍由服务端门禁统一裁决；
- 1280×720 下三栏无横向溢出，轨迹点支持鼠标与键盘定位。
