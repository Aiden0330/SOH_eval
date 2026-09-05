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
| 合法物理特征上限 | q_win_ah | ≤ 0.983 | 不误伤（后续修订见 §6：R² 逃逸但窗口覆盖全量程） |

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

## 6. 修订（2026-08-21）：结构性禁用 + Ah 窗口 DOD 规则

### 6.1 标签来源列结构性禁用（commit a549470）

仿射检测是"逐数据集判定"，但标签来源列在任何数据集上都不该做模型输入。
`feature_safety.LABEL_SOURCE_FEATURE_COLUMNS` 静态封禁：
`charge_capacity_max`、`capacity_ah`、`rated_capacity(_ah)`，以及由其派生的
`charge_energy_proxy_wh`、`cumulative_capacity_proxy_ah`（transform 已不再生成）。
执行点：`FeatureSet.columns()` / `full` 选择器 / BYO（`byo.py` 输入过滤）/ runner。
候选列同步收缩：capacity_time 4→3、degradation_kinetics 6→4、
electrochemical_diagnostics 6→5。效果：hust × capacity_time 等组合以净化后
特征恢复可跑；仿射/数值审计保留为动态兜底（防换名混入）。

### 6.2 Ah 积分窗口覆盖率规则（本节为 §1 表末行的修订）

领域规则（评审确认）：**部分窗口 Ah 积分（`q_win_ah`）是合法健康特征，但窗口
DOD 必须 < 90%**；常规实践为 40–50% 窗口（LFP 取电压平台区）。窗口覆盖
（近）全量程时，其积分即标定容量 — 即使逐循环散布让仿射 R² 低于 0.995 逃逸。

实测中位覆盖率（q_win_ah / charge_capacity_max，逐数据集）：

| 数据集 | 中位覆盖率 | 判定 |
|---|---|---|
| hust | 97.7% | 违规（full_range_ah_window，阻断） |
| uconn | 95.2% | 违规（阻断） |
| snl_lfp | 94.4% | 违规（阻断） |
| oxford | 71.7% | 警告（wide_ah_window，可跑） |
| calce_cs2 | 2.8% | 真实部分窗口，通过 |
| nasa | —（无重叠有效行） | 跳过 |

实现：`feature_safety._window_coverage_findings`，接入 `audit_feature_frame`
（四层执行点自动继承）。阈值：**阻断 ≥ 90% · 警告 > 60% · 样本下限 64 ·
逐数据集中位数**（窗口是数据集级采集属性，不做池化拟合）。警告经
`_training_feature_profile` 透出到组合预检（黄色不阻断）。

泄漏清单更新：hust / uconn / snl_lfp 追加 `q_win_ah`（见
`tests/test_label_leak_inventory.py` 评审注记）。注意：v1.1 曾以
"q_win_ah R² ≤ 0.983 不误伤"作为合法特征上限证据 — 该结论仅对 calce/oxford
类真实部分窗口成立；hust/snl/uconn 的 q_win_ah 属全量程积分，按本规则封禁，
其 partial_window 历史成绩并入 §4 的历史工件清算范围。

验收（2026-08-21）：全量后端测试通过；预检实测三档行为 —
hust × partial_window 阻断（≈98%）、oxford 警告可跑（≈72%）、calce_cs2 通过。

### 6.3 长期修复（2026-08-25）：从原始曲线重建真正的部分窗口

`scripts/rebuild_partial_window_features.py` 从 `data/raw/` 原始曲线重新积分
三个 LFP 数据集的 `q_win_ah` / `t_win_s`，窗口取平台区并经扫描标定使
中位覆盖率落入 40–50% 档：

| 数据集 | 积分段 | 新窗口 | 覆盖率（改前→改后） | R²(q_win, 满容量) |
|---|---|---|---|---|
| hust | 5C CC 快充段 | [3.30, 3.52] V | 97.7% → 48.7% | 0.37 |
| snl_lfp | 0.5C 充电段 | [3.30, 3.38] V | 94.4% → 50.7% | ~0.01 |
| uconn | 1C RPT ref_chg CC 段 | [3.30, 3.37] V | 95.2% → 52.2% | 0.80 |

三数据集 100% 行匹配（hust 146,122 / snl_lfp 64,905 / uconn 2,855），
重建后审计零违规零警告。

**calce_cs2 修订（2026-08-27）**：v1.1 曾以"calce 覆盖率 2.8% = 真实部分窗口"
为例 —— 实为**分母失真**：calce 的 `charge_capacity_max` 来自 Arbin 文件级累计
`Charge_Capacity(Ah)`（均值 20.7 Ah、最大 67 Ah，对 1.1 Ah 电芯显然非单循环量），
旧 [3.6, 4.1] 窗口实际积分约 75% 循环充电量。已按 [3.85, 4.05] V（LCO 0.5C
中段）重建：CS2 中位 q_win/额定 = 0.434、CX2 = 0.337（CX2 额定 1.35 Ah，
metadata 增加逐电芯前缀覆盖）。行对齐采用 ccm 序列指纹匹配 + 占位防重
（错配不可能发生；未对齐行置 NaN）：12,612/15,411 对齐，其余为 txt 记录仪
电芯（CS2_8/CS2_21）与指纹不吻合文件。
**审计盲区（待办）**：窗口覆盖率审计以表内 `charge_capacity_max` 为参考 ——
参考列本身失真时审计"假干净"（calce 即此例）。TODO：参考量稳健化
（逐循环充电吞吐或额定容量兜底）。uconn 的 RPT↔cycle_index 映射 =
`cumsum(Num Cycles)`（逐 cell 校验，不吻合则拒写）。原表备份于
`data/processed/backup_pre_partial_window_fix/`；`data/curves/<ds>.json`
的 `feature_window` 与 `data/manifest.json` 的 processed sha256 同步更新。
泄漏清单相应回撤 q_win_ah 条目（评审注记在测试文件内）。
基于旧全量程 q_win 的历史模型工件仍属 §4 清算范围 — 新旧 q_win 语义不同，
旧模型不得在新表上继续使用。

> **2026-09-03 已修复（D5）**：覆盖率审计改以元数据中的额定容量为分母（含电芯前缀覆盖，见 `sohkernel.feature_safety._coverage_reference`），仅在无元数据的数据集上回退到表内容量列；calce 的失真 `charge_capacity_max` 不再影响审计。

### 7. 修订（2026-09-05, K12）：部分充电健康因子的窗口容量列

`sohkernel.health_indicators` 从原始充电曲线计算文献健康因子（四个固定电压子窗口与其跨度内的电量/时长、跨度内 IC 峰、V(Q) 形状、平均电流及其变异），写入 `data/processed/<dataset>_features.pkl`（`scripts/rebuild_partial_charge_hi.py`）。其中电量列 `hi_dq_w1_ah … hi_dq_w4_ah`、`hi_span_dq_ah` 属于 Ah 窗口积分，纳入 `feature_safety.PARTIAL_WINDOW_CAPACITY_COLUMNS`，受 §6.2 的 DOD 覆盖率规则约束（每次重建都审计）；LFP 跨度取 3.30–3.40 V（四个 25 mV 窗口），因为 uconn 的参考充电在 3.40 V 附近结束、hust 5C 充电有五分之一循环达不到 3.55 V，而 3.30–3.40 V 在 hust、snl_lfp、uconn 的每个循环都存在（重建报告 2026-09-05）。特征集 `partial_charge_hi` 只投影无量纲派生列（额定容量分数、充电时长分数、C 倍率、电压、斜率），派生列不进入 `full`。
