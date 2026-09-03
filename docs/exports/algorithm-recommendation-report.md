# 算法推荐报告

- 默认推荐算法：`gbdt`
- 数据集范围：calce_cs2, hust, nasa, oxford, rwth_isea_public, snl_lfp, uconn
- 特征集：`battery_soh_operational`

## 推荐理由
- 跨数据集迁移能力中 gbdt 的平均 MAE 最低。
- 同数据集跨电芯泛化能力中 gbdt 的平均 MAE 最低。

## 注意事项
- 跨数据集迁移能力的最快推理模型是 ridge，不是默认推荐 gbdt。
- 跨数据集迁移能力的训练过程均值最佳是 battery_soh_tabular_ensemble，推理推荐仍以迁移结果为准。

## 跨数据集迁移能力

- 协议：`lodo`
- 成功记录：147/147
- 精度推荐：`gbdt`
- 速度最快：`ridge`
- 训练过程最佳：`battery_soh_tabular_ensemble`

| 排名 | 算法 | MAE | RMSE | ms/1000 |
| --- | --- | ---: | ---: | ---: |
| 1 | `gbdt` | 0.089604 | 0.116874 | 5.455111 |
| 2 | `battery_soh_tabular_ensemble` | 0.09337 | 0.115923 | 39.115461 |
| 3 | `ridge` | 0.116366 | 0.138166 | 0.895969 |

## 同数据集跨电芯泛化能力

- 协议：`loco`
- 成功记录：3080/3080
- 精度推荐：`gbdt`
- 速度最快：`gbdt`
- 训练过程最佳：`gbdt`

| 排名 | 算法 | MAE | RMSE | ms/1000 |
| --- | --- | ---: | ---: | ---: |
| 1 | `gbdt` | 0.080077 | 0.105995 | 8.530577 |
| 2 | `battery_soh_tabular_ensemble` | 0.085676 | 0.107146 | 45.918551 |
