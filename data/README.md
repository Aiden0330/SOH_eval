# SOH_eval 项目数据目录

本目录是 `SOH_eval` 的默认数据根目录。大体量原始数据、处理缓存和曲线资产不提交到 Git；
`manifest.json` 保留数据来源、迁移状态、规模、用途和校验信息。

## 目录结构

```text
data/
├── raw/          # 已接入监督数据的只读原始文件
├── processed/    # 训练、推理和评测直接读取的 *_features.pkl
├── curves/       # 前端原始 V/I/T/IC/DV 与 SOH 轨迹来源
├── registry/     # 本地字段/来源登记（如有）
├── checksums/    # 上游校验记录（如有）
└── manifest.json # 项目数据注册清单
```

默认运行路径应通过 `SOHBENCH_DATA_DIR` 覆盖；未设置时使用本目录。
原始数据仅用于重建与追溯，正常训练、推理和案例评测读取 `processed/`。

## 迁移边界

当前采用方案 B：迁移 NASA、CALCE、Oxford、HUST、SNL、UConn 等已接入监督数据，
以及全部处理缓存和轻量曲线资产。RWTH 户储现场大文件、无监督电站数据、历史工程和重复压缩包
暂不复制，但会在 `manifest.json` 中登记为外部或未接入资源。

迁移完成清单为 306 个文件、24,288,608,875 字节；9 个处理数据集共 235,320 条记录、
235,288 个唯一电池-Cycle、226 个电池。`raw_local=true` 只说明原始数据目录已迁移，
不代表每个 Cycle 都已转换成前端曲线；逐 Cycle 曲线覆盖由运行内 `CurveSegmentRef` 判定。

三页数据流、缓存/索引结构和真实案例复核见
`docs/architecture/soh-evaluation-phase-3.9-local-data-connectivity.md`。
