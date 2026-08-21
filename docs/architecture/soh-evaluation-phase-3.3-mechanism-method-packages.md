# 阶段 3.3：机理特征、灰盒方法与 PINN 方法包

日期：2026-08-16  
状态：已实现第一批可运行能力；严格电化学方法保留为数据契约，不伪造结果。

## 1. 结论

现有“算法 × 特征”矩阵只适合两类资源：

1. 目标无关、确定性计算的特征方案；
2. 接受普通二维特征表的纯数据驱动估计器。

以下资源不应继续作为自由笛卡尔积中的普通选项：

- 半经验老化先验 + 残差学习；
- ECM/SPM/SPMe/P2D 与 EKF、UKF、粒子滤波；
- 含控制方程、初边值条件和多项损失的严格 PINN；
- CORAL、DANN 等需要显式源域/目标域的迁移学习；
- 依赖脉冲、EIS、静置或专用温度协议的诊断方法。

这些方法应建模为 `MethodSpec` 方法包：估计器、特征处理、所需信号、数据粒度、约束、训练/推理策略和证据输出共同固定。阶段 3.3 已在目录中加入 `method_family`、`pairing_mode`、`compatible_featuresets`、`required_signals` 和 `physics_constraints`，并在预检与运行器中自动跳过不适用的笛卡尔组合。

## 2. 一手资料结论

| 方向 | 一手资料 | 对本工具的直接含义 |
|---|---|---|
| IC/DVA | [Energy 2018, DOI 10.1016/j.energy.2018.03.023](https://www.sciencedirect.com/science/article/pii/S0360544218304213)；[Journal of Energy Storage 2022, DOI 10.1016/j.est.2021.103669](https://doi.org/10.1016/j.est.2021.103669) | 峰位、峰高、面积和峰间距可作为机理诊断特征，但必须保存平滑、采样、电压窗和峰匹配规则。只有两个峰值列不足以覆盖完整诊断。 |
| 早期电压曲线 | [Nature Energy 2019, Severson et al.](https://doi.org/10.1038/s41560-019-0356-8) | 曲线差分、方差等特征可预测寿命；它们是固定预处理管线，不应只登记为几个无来源的表格列。 |
| 电池 PINN | [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48779-z)；[作者代码 PINN4SOH](https://github.com/wang-fujin/PINN4SOH) | 论文方法耦合经验退化/状态空间方程、解网络、动力学网络、方程残差、数据损失与单调损失。当前项目的 `pinn` 只是逐 Cycle MLP + 单调/范围正则，应标为 physics-regularized neural，而不是严格控制方程 PINN。 |
| SPM/PINN 混合 | [Batteries 2023, DOI 10.3390/batteries9060301](https://www.mdpi.com/2313-0105/9/6/301) | SPM 状态、参数、初边值条件和网络共同构成方法包，不能与任意特征方案自由组合。 |
| 贝叶斯 PINN | [Scientific Reports 2023](https://www.nature.com/articles/s41598-023-33018-0)；[NASA 复现代码](https://github.com/nasa/Li-ion-Battery-Prognosis-Based-on-Hybrid-Bayesian-PINN) | 不确定性是模型结构和推断过程的一部分，不只是事后 conformal 包装。 |
| 物理均值 GPR | [Journal of Power Sources 2017 / Oxford repository](https://ora.ox.ac.uk/objects/uuid%3A46a15fa2-6e51-47e6-afe4-c0223e254569) | GPR 可把物理模型放入均值函数并联合多个输出；这应成为独立灰盒方法包，区别于当前零均值通用 GPR。 |
| SPMe 状态估计 | [Moura 等 SPMe observer](https://its.berkeley.edu/publications/battery-state-estimation-single-particle-model-electrolyte-dynamics)；[SPMeT 代码](https://scott-moura.github.io/SPMeT/) | 需要连续 V/I/T、OCV/参数、初始状态和观测模型；应进入“状态估计”流程，不属于特征选择器。 |
| 半经验老化应力 | [Journal of Power Sources 2014, Schmalstieg et al.](https://www.sciencedirect.com/science/article/abs/pii/S0378775314001876) | 温度、电压、DOD、SOC、电流和吞吐量共同决定老化。当前只能构造应力代理，不能把代理系数解释为已辨识机理参数。 |
| EIS | [Journal of Energy Storage 2025, DOI 10.1016/j.est.2024.114715](https://www.sciencedirect.com/science/article/pii/S2352152X24043019) | EIS 特征依赖频谱、SOC 和温度条件；没有阻抗谱时不能补造欧姆/电荷转移/扩散特征。 |
| 脉冲内阻 | [Journal of Power Sources 2021, DOI 10.1016/j.jpowsour.2021.230519](https://www.sciencedirect.com/science/article/pii/S037877532101020X) | R0、极化与扩散代理需要定义清楚的电流脉冲和电压响应，逐 Cycle 均值不能替代。 |
| 跨域迁移 | [AAAI 2016 CORAL](https://ojs.aaai.org/index.php/AAAI/article/view/10306) | 推理页的多数据集验证是“泛化评测”；若启用无监督域适配，必须固定源域、无标签目标域和最终盲测域，不能在普通预测调用中暗中拟合目标数据。 |

## 3. 本轮新增的可运行能力

### 3.1 目标无关的机理代理量

`_mechanism_proxy_transform` 保持记录主键与行数不变，不读取 `soh`，新增：

| 字段 | 计算 | 边界 |
|---|---|---|
| `cycle_sqrt` | `sqrt(max(cycle, 0))` | 半经验老化坐标，不是时间或 Ah 吞吐本身 |
| `cycle_log1p` | `log(1 + cycle)` | 早期变化的压缩坐标 |
| `voltage_span_v` | `Vmax - Vmin` | 逐 Cycle 电压跨度，不是脉冲压降 |
| `charge_energy_proxy_wh` | `Vmean × Qcharge,max` | 仅为能量代理，非积分能量 |
| `cumulative_capacity_proxy_ah` | `cycle × Qcharge,max` | 无逐 Cycle 累加数据时的近似吞吐代理 |
| `current_rms_proxy_a` | `sqrt(Imean² + Istd²)` | 由聚合统计重建的 RMS 代理 |
| `current_variability_ratio` | `Istd / mean(|I|)` | 工况波动描述 |
| `temperature_over_25c` | `Tmean - 25°C` | 参考温差 |
| `arrhenius_stress_factor` | 固定 24 kJ/mol、25°C 参考 | 跨工况描述量，不是已辨识活化能 |

新增特征方案：

- `electrochemical_diagnostics`：IC/DVA、局部充电窗、电压跨度、能量代理；
- `operating_stress`：电流波动、温度、Arrhenius 应力和电压跨度；
- `degradation_kinetics`：平方根/对数老化坐标、容量吞吐和温度应力。

当源列缺失时派生列保留为 `NaN`，由预检显示数据集覆盖差异；不会使用常数伪造“可用机理特征”。

### 3.2 新增算法与方法

| 名称 | 类型 | 组合规则 | 用途 |
|---|---|---|---|
| `elastic_net` | 可解释数据驱动 | 自由组合 | 以稀疏线性基线检查新增特征是否真正增益 |
| `extra_trees` | 非线性集成 | 自由组合 | 检查非线性交互和特征重要性 |
| `pls` | 潜变量回归 | 可组合，推荐 `electrochemical_diagnostics` | 处理 IC/DVA 与窗口特征的强共线性 |
| `stress_resid` | 灰盒方法包 | 仅指定机理方案/`full` | 非负退化先验 + Arrhenius/电流应力 + GBDT 残差 |
| `phys_resid` | 既有灰盒方法包 | 已补组合约束 | `sqrt(cycle) + cycle` 先验 + GBDT 残差 |
| `pinn` | 既有物理正则网络 | 已补组合约束 | 明确标注为单调/范围正则，不冒充严格 PDE PINN |

新增算法实现依赖均为项目已有的 NumPy、SciPy 和 scikit-learn；PyTorch 仍只在可选深度学习运行时加载。

## 4. 方法分类与界面规则

建议将训练页从一个平面矩阵改为三个入口：

1. **自由比较**：Ridge、Elastic Net、GBDT、Extra Trees、SVR、GPR、MLP × 任意已通过预检的确定性特征；
2. **机理辅助预设**：PLS + electrochemical diagnostics、GPR + physics mean、半经验 prior + residual；用户选“方法”，系统展开固定构成；
3. **联合方法包**：SPMe observer、ECM-EKF、严格 PINN、Bayesian PINN、域适配；界面显示信号契约、参数/边界条件和可验证证据，不显示自由特征复选框。

目录建议统一使用：

```text
MethodSpec
  identity: id, version, family
  estimator: implementation, artifact contract
  feature_pipeline: transforms, fitted state, window definition
  data_contract: signals, granularity, units, protocol conditions
  physics: equations, parameters, initial/boundary conditions, constraints
  validation: LOCO/LODO/LCHO, source/target domains, leakage guards
  evidence: predictions, intervals, curves, residuals, parameter diagnostics
```

训练和推理不应各维护一套不同“组合列表”。推荐保存同一个不可变 `MethodArtifact`；训练记录其训练域与内部验证，推理记录一个或多个目标数据集的外部泛化结果。第 3 页按同一方法工件聚合这些验证记录，明确区分“训练内指标”和“目标域指标”。

## 5. 数据成熟度门禁

| 等级 | 当前可做 | 门禁 |
|---|---|---|
| L1 逐 Cycle 表格 | 本轮 9 个代理量、三组特征、PLS/灰盒残差 | 必须显示代理定义、源列和缺失率 |
| L2 原始 V/I/T 曲线 | IC/DVA 多峰、峰面积/间距、CV 时间、弛豫、曲线嵌入 | 统一时间轴、单位、阶段标签、滤波和窗口版本 |
| L3 脉冲/EIS/热测试 | R0/R1C1、扩散、DRT、熵/热力学特征 | 固定 SOC、温度、静置、脉冲/频率协议和设备元数据 |
| L4 控制方程联合模型 | ECM/SPM/SPMe/P2D、EKF/UKF/PF、严格/贝叶斯 PINN | 参数、初始/边界条件、可辨识性、求解器、残差和不确定性证据齐全 |

## 6. 下一批建议

优先级按“价值 / 数据准备成本 / 防止错误结论”排序：

1. 在原始曲线管线中增加多峰 IC/DVA 与窗口版本化，把第 3 页高亮窗口和模型实际输入绑定到同一个 `CurveSegmentRef`；
2. 新增 physics-mean GPR 方法包，输出物理均值、GP 残差和区间三条证据；
3. 为多数据集推理建立 source / adaptation / blind-test 三域契约，再实现 CORAL baseline；
4. 建立脉冲与 EIS 数据规范后再增加 ECM 参数、DRT 和阻抗特征；
5. 将现有 `pinn` 在界面重命名为“物理正则 MLP”，另建严格 PINN 方法包，只有控制方程和初边值条件完整时才解锁。

阶段 3.3 的核心门禁是：**宁可显示“不适用/缺少信号”，也不把没有物理含义的列组合包装成机理模型。**
