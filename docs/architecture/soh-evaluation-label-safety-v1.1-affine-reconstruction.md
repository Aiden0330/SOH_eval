# 标签安全 v1.1 · 仿射重构检测（affine label reconstruction）

> 2026-08-20 · 合约 `sohbench.label-safety.v1`（检查项扩展，合约名不变）
> 背景：byo_skb_ridge 与 gpr × capacity_time 在 hust 上出现 MAE 0.0003–0.0006 的"完美"成绩。
> 经查证为标签泄漏：`soh := 标定容量 / 额定容量`，而 `charge_capacity_max` 即该标定容量。

## 1. 证据（2026-08-20 实测）

| 数据集 | 重构列 | 仿射拟合 R² | 判定 |
|---|---|---|---|
| hust | charge_capacity_max | 0.99993 | 标签来源列 |
| oxford | charge_capacity_max | 0.99835 | 标签来源列 |
| uconn | charge_capacity_max | 0.99892 | 标签来源列 |
| rwth_isea_public | capacity_ah | 0.99976 | 标签来源列 |
| nasa / calce_cs2 / snl_lfp | —（最高 0.979） | — | 独立标定，无重构列 |
| 合法物理特征上限 | q_win_ah | ≤ 0.983 | 不误伤 |

阈值据此定为 **R² ≥ 0.995（逐数据集分组拟合）· 样本下限 64**。
已发布主结果（mlp × partial_window 等）不受此泄漏影响。

## 2. 检测与执行层（四层）

1. **入库扫描**（`trusted_plugins.upload_dataset`）：上传数据集时全列扫描，
   命中列记入目录字段 `label_source_columns` — 数据集不拒收（携带标签来源合法），
   但下游全程可见。
2. **训练门禁**（`evaluation_service.run_job_inline`）：对本次请求的直通特征方案
   （无 transform 的 featureset）与 BYO 全表逐列审计，命中即整体拒绝训练并给出精确原因 —
   覆盖此前绕过 transform 审计的两条路径。
3. **transform 审计**（`feature_safety.apply_transform_without_label`）：原有路径，
   新增 `affine_label_reconstruction` 违规类型。
4. **可信推理/评估**（`model_store.LoadedModel._ensure_label_safe`）：预测前审计
   manifest 特征列；`compatibility()` 同步报告 all-NaN 列。

## 3. 常驻回归（leak inventory）

`tests/test_label_leak_inventory.py` 对 `data/processed/*` 全量执行审计并锁定预期清单：
- 出现**新的**重构列 → 测试失败（新泄漏或未审核的新数据集）；
- 已知标签来源列**不再被检出** → 测试失败（检测回归）。
清单变更须随评审理由一并提交 — 该文件即审计痕迹。

## 4. 待办（按优先级）

- [ ] 历史工件清算：对既有 capacity_time / byo 模型工件执行
      `invalidate_label_leakage_results` 流程（沿用 smoke_fade_from_full 先例），
      榜单成绩标注 `invalidated · label-safety`。
- [ ] 多元重构探针（B 阶段）：k 特征线性探针 + 交叉验证 R²，阈值预注册后再启用 —
      防两列联合重构（如 容量 + 额定容量）。
- [ ] 前端呈现：数据集目录页展示 `label_source_columns`；训练页组合预检提示
      "该组合包含标签来源列，将被训练门禁拒绝"。
- [ ] 上传数据集的 datasheet 模板加入"标签构造"必填项（label anchor 声明）。

## 5. 验收记录

- 353 项后端测试 + 3 项仿射检测测试 + 泄漏清单测试通过（2026-08-20）。
- 训练门禁实测：byo/capacity_time × hust 请求被拒，错误信息含列名、分组、R² 与斜率。
- 推理门禁实测：`gpr_degradation_kinetics` 于 rwth 被 all-NaN 拦截；
  轨迹面板错误信息现透传服务端原因（fetchJson 修复）。
