# SOH 方案 —— 外部研究锚定与设计决策（确定方案）

> **日期**：2026-06-26
> **目的**：按"clean-room、只认本目录 md + data + 第一性原理"的要求，用顶刊/顶会/高索引文献/标准/法规/工业界实践
> 把方案**确定下来**，替代此前对老 skill 契约的锚定。
> **检索纪律**（literature-research）：每个陌生基准/系统先联网到一手来源；一手（arXiv/官方仓库 fetch）与二手
> （搜索结果摘要）分别标注，二手数字定稿前需对照原文核验（标 ⚠️二手）。
> **关联**：[父方案 md](../../../SOH泛化性增强分层方案.md) · [重评估](2026-06-26-soh-reeval-and-redesign.md)

---

## 0. 结论先行：本研究确定/修正的 12 项决策

| # | 决策 | 取值 | 主要依据 |
|---|---|---|---|
| D1 | SOH 定义 | 容量侧 `SOH = C_now / C_rated`，显式记录；保留能量侧为备注 | 无国际统一定义，容量比是主流 |
| D2 | SOH 值域 QC 上界 | **< 1.3**（可配），>1.0 合法 | 容量再生/可逆锂、实测超标称真实存在 |
| D3 | EOL 阈值 | 80%（CALB 类工业大容量 90%）| IEEE 惯例 + BatteryML/BatteryLife |
| D4 | 不确定度 | **强制**，conformal 默认，跨域用加权/自适应 CP | EU 法规要求可审计 + CP 已成 SOH UQ 标配 |
| D5 | 防泄漏切分 | 留一电芯（cell-wise）为底线；跨域留出度量泛化 | 公认 battery-wise 协议 |
| D6 | 数据 QC 门 | 剔除越界/疑似设备故障突降/非降解电芯 | BatteryLife 同款 QC |
| D7 | 多团队比较架构 | BatteryML 式模块化 + 统一数据表示 + 模型接口插拔 | 既有基准平台验证范式 |
| D8 | 公平性机制 | 统一预处理 + **多种子平均（≥5）** + 同一输入契约 | "不公平比较源于预处理不一致" |
| D9 | 指标集 | RMSE/MAE/**MAPE**/EOL区/max/**α-accuracy(15%)**/PICP/单调违例 | BatteryML(MAE) + BatteryLife(MAPE,15%-acc) |
| D10 | 物理一致 | 单调（吞吐量维，留回跳容差）+ 值域 | 容量再生真实→必须留容差 |
| D11 | 自研算法路线 | 传统 ML + physics-residual + 异质集成（非 DNN）| 可审计性合规，且命中域挑战 |
| D12 | 安全预警扩展缝 | 算法无关接口 + 一致性/突降特征预留 | 电芯不一致→热失控是工业关切 |

> D2/D4 是对此前从老 skill 拿来的 `[0.6,1.05]` 上界与"区间可选"的**修正**，现以法规/文献重新锚定。

---

## 1. 既有基准平台（目标②的现成范式，一手 fetch）

三个公开基准平台直接对标"不同团队 SOH 算法比较"，是本项目脊柱 A 的设计参照：

| 平台 | 规模 | 模型接口 | 切分 | 指标 | 公平性机制 |
|---|---|---|---|---|---|
| **BatteryML**（微软, arXiv 2310.14714）| 7 源/8 变体, 14 模型 | 配置驱动 + sklearn/pytorch 后端插拔；5 模块（split/label/feature/preprocess/model）+ 统一 `BatteryData`(meta+cycles) | 随机**电芯级**或数据集标准划分 | MAE±std | **10 种子平均**消除初始化偏置 |
| **BatteryLife**（arXiv 2502.18807）| 990 电芯/16 数据集/59 化学体系/421 充放协议/9 温度 | 18 方法 | 6:2:2 **按域**随机 + 跨域迁移分析 | **MAPE + 15%-Accuracy**, mean±std/3 run | 3000+ 超参搜索于验证集；QC 见下 |
| **SOHbenchmark**（Wang Fujin, J. Energy Storage 2024）| 100 电芯, 5 DL, 3 输入型, 3 归一 | —（DL 为主）| 数据集标准划分 | — | 公开完整预处理+评测码 |

**对脊柱 A 的采纳**：
- 模块边界采 BatteryML 五模块思路（与重评估 v2 的 `data/qc/protocols/metrics/runner/report` 同构）。
- **统一数据表示**：除 17 维特征表外，保留 BatteryML 式 `meta + cycles` 原始表示，服务 raw 可选通道与未来 L1 特征测评。
- **模型接口**：sklearn/pytorch 双后端插拔——直接支撑"各团队算法接入"。但因 D11 非 DNN 约束，自研条目走 sklearn 侧；外部 DL 算法仍可接入比较（比较不受非 DNN 约束限制）。
- **公平性（D8）**：采 BatteryML 多种子平均（≥5）；统一预处理/特征，让所有算法见到**同一输入**——直接回应 SOHbenchmark/综述指出的"预处理不一致导致不公平比较"。
- **QC（D6）**：采 BatteryLife 实践——剔除"未降解到 80%（±2.5%）"、"疑似设备故障的 SOH 突降"、"<100 循环"的电芯。

> ⚠️ BatteryML 的 `BatteryData` 含 internal resistance 字段；本项目 D11 禁内阻，接入时**不取 IR 通道**。

---

## 2. SOH 定义与值域（D1/D2/D3）

- **无统一国际定义**：SOH 是"当前满充能力 / 全新满充能力"，IEEE 常用 **80% 退役阈值**；各 OEM 用容量保持/功率/内阻/专有模型计算，定义因厂商而异（⚠️二手，多源一致）。标准（IEC 62660-1:2018、ISO 12405-4:2018、GB/T 31486/34015 等）规定的是**容量测试条件**，不是单一 SOH 公式（⚠️二手）。
- **SOH 可 >100%**：`SOH = 最大放电容量 / 标称容量`，实测 > 标称即 >100%，是真实现象；静置后可逆锂重新嵌入会**升容**（容量再生），并非制造错误（⚠️二手，多源一致；另见 ECS 关于"超 80% 后退化"的报告）。
  - → **印证你给的事实**：厂商预留容量 ⇒ 初始可 >1.0；`<1.3` 作为经济上界是合理工程启发。
  - → **同时印证 D10**：NASA/CALCE 实测 10–13% 循环 >0.5% 回升中，**部分是真实再生**，故单调门必须留回跳容差（吞吐量维约束，非日历时间）。
- **本项目采用**：容量侧 `SOH=C_now/C_rated`；QC 值域 `[lower, 1.3)`，`lower` 可配、只拦明显垃圾（≤0/NaN）；EOL 区阈值 80%（工业大容量 90%）。

---

## 3. 不确定度：从"可选"修正为"强制"（D4）

- **法规驱动**：EU 2023/1542 第 14 条（2024-08-18 起）要求 BMS 含 SOH 与预期寿命参数（Annex VII）、可只读访问；电池护照（2027-02，>2 kWh）要求实时 SOH + 历史；"欧洲电池法规要求可靠、透明的 SOH 估计"（⚠️二手，多源一致）。→ **可审计 + 透明 = UQ 不可省**，与老 skill 无关。
- **方法已成熟**：Conformal Prediction 是 SOH UQ 的事实标准——分布无关、有限样本覆盖；单循环 ERT 集成给出 95% PI、实测覆盖 **94.1%（SoH）**（一手摘要，ScienceDirect S2590123026022462）；Conformalized Transfer Learning combine LSTM+MMD+CP（arXiv 2603.24475）；跨域 non-exchangeable 用 **SPCI**（arXiv 2507.00353）。
- **本项目采用**：CI 强制；点模型自动套 split-conformal；跨域折（chemistry/dataset-out）用**加权/自适应 conformal**（标准 CP 在跨域违反 exchangeability）；报告 PICP@90/95 并标 `ci_source`。

---

## 4. 泛化、域适配、验证协议（D5）

- **域适配 > 裸迁移**（印证 md L2）：TL meta 分析指出，显式域适配与表征级迁移在**强域偏移**下稳健性持续优于单纯 fine-tune（⚠️二手，Springer s11581-026-07030-z）。代表作：Domain-Adaptive Transformer 做跨域 SOH+RUL（Energy S0360544225049308）、Physics-Guided Test-Time Training（arXiv 2402.00068）、跨化学体系"可解释距离特征 + 动态权重"（Energy S0360544225028178）。
- **防泄漏**：battery-wise（留一电芯）是公认协议——防止同一电芯的循环跨训练/测试泄漏；分组列未纳入切分即 group leakage（⚠️二手，Wikipedia + 综述一致）。→ 印证 md L5 与重评估的留一域协议。
- **跨工况评测**：BatteryLife 用 421 充放协议 × 9 温度做"seen/unseen aging conditions"与 Li→其他域迁移——这正是你"测评考虑不同工况数据集"的成熟做法。
- **本项目采用**：留一 cell-out 为底线；dataset-out / chemistry-out（满足可分性，≥2 数据集/化学体系）度量跨域；前向链时序留出度量外推；指标含"误差 vs 域距离"。

---

## 5. 方法学路线与综述对照（D11）

2024–2025 综述（RSER S1364032125007981、Wiley acis/5516151 与 enc2.12125、Batteries 12020076）给出的统一分类与**公开挑战**：
- 特征：V/I/T、IC/ICA、弛豫/DRT；方法：SVM、GPR、LSTM、树、Transformer、PINN；策略：迁移学习、物理约束、鲁棒训练、效率设计。
- **明列挑战**：cell→pack 迁移、**适应性与泛化**、数据驱动模型的可靠性/可解释性、云平台、真实大数据、BMS 集成、**数据稀缺、域可迁移、物理一致**（⚠️二手，多综述一致）。

→ **md 的 L1–L6 与该挑战清单逐条吻合**（HI 特征 / 域适配 / 物理一致 / 不确定度 / 留一域 / 在线漂移），说明方案选题对路。
→ **非 DNN 取舍的正当性更强了**：综述与法规都强调**可解释/可审计**（EU 法规、OEM 透明性诉求）；传统 ML + physics-residual 在可审计性上优于黑箱 DL。代价是放弃端到端特征学习——这是 deployability/auditability 的自觉权衡，已在 md §3.0 论证。

---

## 6. 工业界与 BESS 现场（D12 + 现实约束）

- **现场数据稀缺**：缺乏大规模公开 BESS 现场数据，制约真实健康监测（⚠️二手，多源；另见 Nature Energy s41560-024-01620-9 户储多年现场容量估计、ScienceDirect S2590116824000511 EV 现场系统级 SOH）。
- **标定昂贵**：SOH 标定循环需让 BESS **离线数小时**，运营成本高 → 印证"机会性满充标定 + 协变量外推"的现实必要性（对应 md L6 与无内阻兜底）。
- **一致性→安全**：电芯不一致（电压/SOC/温度/SOH）传播到 pack，致功率失衡、加速退化乃至**热失控**（⚠️二手）→ 这是**安全预警扩展缝（D12）**的直接动机：A 的接口需算法无关，并预留一致性/突降类特征。
- **标准缺口**：现场容量衰减量化方法缺标准，EU 法规倒逼透明 SOH。

---

## 7. 对各子系统的影响

- **A（脊柱：统一接口 + 测评/比较 harness）**：按 §1 采纳 BatteryML 模块化 + 统一数据表示 + 模型接口；公平性用统一预处理 + 多种子平均；指标集 D9；协议 D5；QC 门 D6；UQ D4；值域 D2。
- **B（自研 SOH 估计+适配）**：md L1–L6 作为实现 A 接口的"我们队条目"；路线 D11；域适配采显式 DA（CORAL/MMD 等传统 ML 可实现版）。
- **C（前端三 mockup）**：消费 A 的 `results/leaderboard/gates/qc_report`；用 ui-ux-pro-max；可视化跨工况/跨域比较、误差-域距离、PICP 覆盖、QC 报告。
- **未来安全预警**：A 接口算法无关、协议可复用；本阶段只建 SOH，留缝不建。

---

## 8. 待核验 / 待用户确认

- ⚠️ 标 ⚠️二手 的数字（覆盖率、规模、法规条款细节）定稿前需对照一手原文/法规正文核验。
- 待确认：D8 公平性的"统一输入"边界——是否允许"自带特征"模式（bring-your-own-features）与"框架统一特征"模式并存？（影响目标②不同团队算法的接入自由度 vs 可比性）
- 待确认：A 第一版纳入哪些数据集（现有 processed：NASA/CALCE/HUST/Oxford；可补 SOHbenchmark/XJTU、MATR、RWTH 等以增工况/化学多样性）。

---

## 9. 主要来源（markdown 链接）

**基准平台（一手 fetch）**：[BatteryML arXiv 2310.14714](https://arxiv.org/html/2310.14714v4) · [BatteryLife arXiv 2502.18807](https://arxiv.org/html/2502.18807v7) · [SOHbenchmark GitHub](https://github.com/wang-fujin/SOHbenchmark) · [SOHbenchmark 论文(J. Energy Storage)](https://www.sciencedirect.com/science/article/abs/pii/S2352152X23032826)
**综述**：[RSER 2025 ML-SOH 综述](https://www.sciencedirect.com/science/article/abs/pii/S1364032125007981) · [Wiley 2025 AI-SOH](https://onlinelibrary.wiley.com/doi/10.1155/acis/5516151) · [Wiley 2024 cell-to-pack](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/enc2.12125) · [Batteries 2025 神经架构评述](https://doi.org/10.3390/batteries12020076) · [Nat. Mach. Intell. 2021 ML pipeline](https://www.nature.com/articles/s42256-021-00312-3)
**域适配/迁移**：[Domain-Adaptive Transformer (Energy 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0360544225049308) · [Physics-Guided TTT (arXiv 2402.00068)](https://arxiv.org/html/2402.00068v3) · [跨化学距离特征 (Energy 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0360544225028178) · [TL meta 分析 (Ionics 2026)](https://link.springer.com/article/10.1007/s11581-026-07030-z)
**不确定度/conformal**：[Conformalized TL (arXiv 2603.24475)](https://arxiv.org/pdf/2603.24475) · [单循环 ERT 95%PI (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2590123026022462) · [SPCI 物理集成 (arXiv 2507.00353)](https://arxiv.org/html/2507.00353v1) · [混合集成 UQ (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S026322412600237X)
**法规/标准**：[EU 2023/1542 要点](https://www.batteryregulation.eu/post/eu-battery-regulation-2023-1542-2024-requirements) · [TÜV SÜD EU 电池法规白皮书](https://www.tuvsud.com/en-us/-/media/regions/us/pdf-files/whitepaper-report-e-books/tuvsud_overview-of-eu-battery-regulations.pdf) · [IEC 62660-1:2018 样张](https://cdn.standards.iteh.ai/samples/23595/17e686b6e2c54ff69e9e8f8babca08f2/IEC-62660-1-2018.pdf) · [GB vs 国际标准对比](https://www.test-navi.com/eng/report/pdf/ComparisonOfGBAndInternationalStandardsForElectricVehicleSecondaryBatteries.pdf) · [npj SOH 测量标准化提案](https://www.nature.com/articles/s44406-025-00010-8)
**SOH>100%/再生**：[State of health (Wikipedia)](https://en.wikipedia.org/wiki/State_of_health) · [部分充电大容量 SOH (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2352152X21015243) · [ECS 超 80% 退化 Q&A](https://www.electrochem.org/ecsnews/qa-ecs-webinar-degradation-commercial-lithium-ion-cells)
**工业/BESS 现场**：[EV 现场系统级 SOH Part 2](https://www.sciencedirect.com/science/article/abs/pii/S2590116824000511) · [Nature Energy 户储现场容量](https://www.nature.com/articles/s41560-024-01620-9) · [可解释 BESS 运维 (arXiv 2601.03007)](https://arxiv.org/pdf/2601.03007)
**数据泄漏**：[Leakage (ML) Wikipedia](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
