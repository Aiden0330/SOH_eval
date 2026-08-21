# 阶段 3.4：引导式闭环与三类方法运行架构

日期：2026-08-16  
状态：交互闭环与统一方法契约已实现；严格电化学求解器仍需按本说明接入，不以代理结果冒充。

## 1. 本阶段解决的问题

第 3 页原来的主按钮直接等同于“签署 EvidencePack”。当候选判定不通过、存在问题单、尚未复测、尚未对账或尚未批准时，按钮只能显示为“等待证据闭环”，用户既不知道缺什么，也无法从当前页面继续。

本阶段把它改为**唯一的上下文主动作**。界面读取当前案例版本、服务端判定、问题单、回归集、复测、对账与证据包状态，只显示当前真正可执行的一步：

```text
不通过路径
创建问题 → 建立回归集 → 固定当前及邻近 Cycle → 受控复测
→ 确认问题闭环 → 确认不通过并结束

通过路径
提交评审 → 批准评审 → KPI 对账 → 生成 EvidencePack → 签署发布
```

“不通过”也是合法终态。确认后服务端先写入 `review=rejected`，再写入 `lifecycle=failed`，审计轨迹保留操作者、原因和案例版本，不再要求失败案例伪装成可签署结果。

## 2. 辅助区简化

侧栏辅助区由三个常驻入口压缩为两个：

| 入口 | 包含能力 | 设计原则 |
|---|---|---|
| 可信洞察 | 可信总览、诊断与洞察建议 | “看结果、找原因、定下一步”合并为一个阅读入口 |
| 资源管理 | 算法包、模型工件、数据与接入 | 所有上传、版本、依赖和包管理集中到一个管理入口 |

洞察建议面板仍可通过内部跳转和历史深链打开，但不再占用一个一级按钮。第 1—3 页只保留流程动作，资源安装和 Pickle/算法包上传不回流到训练页面。

## 3. 三类方法不是三套列表

训练与推理应使用同一个不可变方法身份；差别在于验证域：

- 训练记录训练域、折次、内部验证和模型工件；
- 推理为同一模型工件追加同源回放或多个外部数据集的泛化记录；
- 第 3 页按方法身份聚合训练内、同源和跨数据集结果。

目录同时保留两个正交维度：

1. `method_family`：科学属性，如 `data_driven`、`gray_box_degradation`、`physics_regularized`、`state_estimation`；
2. `method_category`：产品中的组合与执行方式，即下面三类。

| 方法类别 | 组合规则 | 执行契约 | 适用对象 |
|---|---|---|---|
| `free_combination` 自由组合 | 通过预检后参加特征 × 算法笛卡尔积 | `estimator/v1` | Ridge、GBDT、GPR、MLP 等普通表格估计器 |
| `recommended_preset` 推荐预设 | 默认绑定推荐特征，允许用户改选并记录偏离 | `estimator/v1` | PLS + IC/DVA、推荐工况特征的 GPR 等 |
| `joint_method` 联合方法 | 方法拥有特征处理、先验、损失或求解器；禁止任意笛卡尔积 | `joint_method/v1` 或 `embedded_pipeline/v1` | 灰盒残差、ECM/SPMe + 滤波器、严格 PINN、域适配 |

因此，“机理算法可以不区分算法和特征”的正确落点是 `joint_method`：界面选择的是完整方法，内部仍保存可审计的特征管线、参数、方程和模型工件，但不把这些内部构件暴露成任意复选框。

## 4. 已落地的统一契约

`src/sohbench/method_contracts.py` 是目录、预检和运行器共用的唯一规则源，核心字段为：

```text
MethodContract
  identity: name, method_family, method_category
  composition: pairing_mode, combination_policy, preferred_featureset
  execution: execution_contract, feature_mode, backend
  data: data_granularity, required_signals, compatible_featuresets
  physics: physics_constraints
  explanation: description
```

运行路径统一为：

```text
算法目录
  → MethodContract 归类
  → 预检数据列、信号、粒度和组合兼容性
  → 生成可执行实例 / not_applicable 证据
  → Runner 按契约投影特征或交给联合方法内部管线
  → 保存模型工件、预测、区间、曲线与判据
```

目录、预检结果和逐折结果都写入：

- `method_category`
- `combination_policy`
- `execution_contract`
- `pairing_mode`
- `method_family`

这避免前端显示“可运行”，后端却采用另一套组合规则。

## 5. 三类方法的接入方式

### 5.1 自由组合

实现普通 `fit / predict / predict_interval` 接口，不声明兼容白名单，默认进入 `free_combination + cartesian + estimator/v1`。特征投影、协议分折和 conformal 包装由通用 Runner 负责。

### 5.2 推荐预设

仍使用普通估计器接口，但声明：

```python
pairing_mode = "recommended"
preferred_featureset = "electrochemical_diagnostics"
```

界面优先选中推荐组合；用户改选时应把“偏离推荐预设”写入 RunSpec，而不是静默当成同一方法。

### 5.3 联合方法

声明 `pairing_mode="method_package"`，并给出 `compatible_featuresets`、`required_signals`、`physics_constraints`。短期内可以继续通过统一估计器接口执行，但特征构造和物理约束由方法包内部负责。

严格 PINN、ECM-EKF、SPMe observer 等下一步应实现独立 `joint_method/v1` 适配器，至少包含：

```text
prepare_data(context)        固定采样、单位、窗口和域角色
fit_or_calibrate(context)    训练网络、辨识参数或初始化滤波器
infer(context)               输出 SOH、状态、区间和诊断量
emit_evidence(context)       输出残差、参数、约束、求解器与曲线证据
```

严格 PINN 只有在控制方程、初始/边界条件、损失分项和求解器元数据齐全时才登记为 PINN；当前物理正则 MLP 继续归为 `physics_regularized`，不改名冒充。

## 6. 验收边界

本阶段已完成：

- 第 3 页主按钮按服务端状态给出可执行下一步；
- 失败案例有正式结束路径；
- 辅助导航减少为两个一级入口；
- 三类方法的目录、预检、组合过滤、运行记录使用同一契约；
- 现有自由组合、推荐/受限方法和联合/内嵌方法可保持兼容。

后续实现严格机理方法时，不应只增加一个算法名称。必须同时交付数据契约、执行适配器、版本化参数、失败原因和第 3 页可回放证据。
