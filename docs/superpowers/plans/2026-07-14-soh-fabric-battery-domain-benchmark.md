# SOH Fabric Battery Domain & Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Battery SOH 从散落的静态映射和看板导入逻辑升级为版本化 Domain Pack，为当前七数据集生成可追溯 DatasetSnapshot、LabelSpec、曲线索引和公平基准，并把 LFP 诊断、跨域、二次寿命/现场和部署推理明确分赛道。

**Architecture:** `domain_packs/battery_soh` 保存可审查的领域配置，`sohbench.fabric.domain_packs` 负责通用装载，电池模块只实现规范化、标签、原始曲线索引和特征谱系。所有真实数据仍从可配置根目录读取，注册时生成 manifest/checksum，运行时只引用不可变 snapshot。前端和离线包消费同一 EvidencePack 与 SampleIndex。

**Tech Stack:** Python 3.11+、JSON Schema、pandas、NumPy、SciPy、PyArrow、scikit-learn、pytest。

## Global Constraints

- 默认数据根目录为 `D:\battery_SOH\data`，但单元测试全部使用临时合成数据。
- 当前核心七数据集固定为 `nasa`、`calce_cs2`、`hust`、`oxford`、`snl_lfp`、`uconn`、`rwth_isea_public`；snapshot 中必须记录实际发现状态，不伪造缺失数据。
- DatasetSpec、LabelSpec、ProtocolSpec、FeatureSpec 和 PolicySpec 均有独立版本和 digest。
- `uconn` 现有数据与计划引入的 `uconn_second_life` 分开注册，除非 checksum 和 LabelSpec 证明它们是同一资产集合。
- 无可信容量真值的数据只进入 OOD/HI/漂移赛道，不进入监督 SOH 总榜。
- 原始曲线缺失时返回明确不可用原因和 coverage，禁止补造可视化曲线。
- `RawSegmentRef` 和 `FeatureProvenance` 只声明来源；高亮窗口不宣称模型因果解释。
- 核心公平榜不使用只在部分数据集可得的 DCIR/IC 曲线特征；这些特征进入对应专项赛道。
- 真实全量矩阵只在小规模签署基线通过后启动。

---

### Task 1: Add A Versioned Domain Pack Loader

**Files:**
- Create: `src/sohbench/fabric/domain_packs.py`
- Create: `src/sohbench/fabric/schemas/domain-pack.schema.json`
- Create: `src/sohbench/fabric/schemas/task-spec.schema.json`
- Create: `src/sohbench/battery/__init__.py`
- Create: `domain_packs/battery_soh/domain.json`
- Create: `domain_packs/battery_soh/tasks/soh_regression.json`
- Test: `tests/fabric/test_domain_packs.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `DomainPackRef(domain_id, version, digest, root)`
- Produces: `DomainPackLoader.discover(root) -> list[DomainPackRef]`
- Produces: `DomainPackLoader.load(ref) -> DomainPack`
- Domain pack declares: task types, registries, track IDs, schema versions
- TaskSpec declares: target semantics, prediction unit, asset/cycle keys and allowed run types

- [ ] **Step 1: Write failing discovery and digest tests**

```python
def test_battery_domain_pack_is_versioned_and_digest_stable():
    loader = DomainPackLoader(Path("domain_packs"))
    ref = next(item for item in loader.discover() if item.domain_id == "battery_soh")
    pack = loader.load(ref)
    assert ref.version == "1.0.0"
    assert len(ref.digest) == 64
    assert pack.track_ids == (
        "core_fair", "lfp_curve_diagnostics", "cross_domain",
        "second_life_field", "deployment_inference",
    )
```

Add tests rejecting an unknown schema version, path traversal in registry references and a digest change when a referenced spec changes.

- [ ] **Step 2: Run tests and confirm the loader is missing**

Run: `python -m pytest tests/fabric/test_domain_packs.py -q`

Expected: FAIL because `DomainPackLoader` does not exist.

- [ ] **Step 3: Implement schema-validated recursive digesting**

`domain.json` references only paths below its pack root. Compute the pack digest from canonical JSON of the domain document plus ordered `(relative_path, sha256)` entries for every referenced spec. Return immutable dataclasses and reject duplicate IDs across a registry. Validate `soh_regression` as a TaskSpec whose prediction unit is one asset-cycle and whose target is resolved through LabelSpec rather than a hard-coded column.

```python
class DomainPackLoader:
    def load(self, ref: DomainPackRef) -> DomainPack:
        root = ref.root.resolve()
        document = read_json(root / "domain.json")
        validate_document("domain-pack", document)
        entries = []
        registries = {}
        for kind, relative_paths in sorted(document["registries"].items()):
            values = []
            for relative in sorted(relative_paths):
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise DomainPackError("registry reference escapes pack root")
                value = read_json(path)
                validate_document(value["schema_kind"], value)
                entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
                values.append(value)
            reject_duplicate_ids(kind, values)
            registries[kind] = tuple(values)
        digest = sha256_json({"domain": document, "entries": entries})
        return DomainPack.from_documents(document, registries, digest, root)
```

- [ ] **Step 4: Verify pack discovery from any working directory**

Run: `python -m pytest tests/fabric/test_domain_packs.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/domain_packs.py src/sohbench/fabric/schemas/domain-pack.schema.json src/sohbench/fabric/schemas/task-spec.schema.json src/sohbench/fabric/__init__.py src/sohbench/battery/__init__.py domain_packs/battery_soh/domain.json domain_packs/battery_soh/tasks/soh_regression.json tests/fabric/test_domain_packs.py
git commit -m "feat: add versioned battery domain pack"
```

### Task 2: Specify The Current Seven Datasets And Labels

**Files:**
- Create: `src/sohbench/fabric/schemas/dataset-spec.schema.json`
- Create: `src/sohbench/fabric/schemas/label-spec.schema.json`
- Create: `domain_packs/battery_soh/datasets/nasa.json`
- Create: `domain_packs/battery_soh/datasets/calce_cs2.json`
- Create: `domain_packs/battery_soh/datasets/hust.json`
- Create: `domain_packs/battery_soh/datasets/oxford.json`
- Create: `domain_packs/battery_soh/datasets/snl_lfp.json`
- Create: `domain_packs/battery_soh/datasets/uconn.json`
- Create: `domain_packs/battery_soh/datasets/rwth_isea_public.json`
- Create: `domain_packs/battery_soh/labels/reference_discharge_soh.json`
- Create: `domain_packs/battery_soh/labels/system_capacity_soh.json`
- Test: `tests/fabric/test_battery_dataset_specs.py`
- Modify: `src/sohbench/metadata.py`
- Modify: `tests/test_metadata.py`

**Interfaces:**
- DatasetSpec fields: ID, version, chemistry, form factor, source, license status, raw/canonical locators, asset key, cycle key, available signals
- LabelSpec fields: target column, definition, reference capacity rule, measurement procedure, interpolation policy, valid range, unit
- Produces: `battery_metadata_from_specs(pack) -> dict[str, DatasetMetadata]`

- [ ] **Step 1: Write failing completeness and metadata tests**

```python
CORE_DATASETS = {
    "nasa", "calce_cs2", "hust", "oxford",
    "snl_lfp", "uconn", "rwth_isea_public",
}


def test_every_core_dataset_has_dataset_and_label_specs(battery_pack):
    assert set(battery_pack.datasets) == CORE_DATASETS
    for spec in battery_pack.datasets.values():
        assert spec.label_spec_id
        assert spec.chemistry
        assert spec.asset_key and spec.cycle_key
        assert spec.source["url"]
```

Add assertions that `rwth_isea_public` is no longer absent from central metadata and that unknown rated capacity is represented as `null`, not a guessed number.

- [ ] **Step 2: Run tests and observe static-map gaps**

Run: `python -m pytest tests/fabric/test_battery_dataset_specs.py tests/test_metadata.py -q`

Expected: FAIL because the seven versioned specs and RWTH metadata are incomplete.

- [ ] **Step 3: Write reviewed specs and derive compatibility maps from them**

Populate values only from existing project documents and source manifests. Use `license_status="needs_review"` when the local corpus does not include auditable license evidence. `metadata.py` loads the pack and exposes legacy dictionaries for existing callers; it must not retain a second conflicting source of truth.

```python
CORE_DATASET_IDS = (
    "nasa", "calce_cs2", "hust", "oxford",
    "snl_lfp", "uconn", "rwth_isea_public",
)


def load_battery_dataset_specs(pack_root: Path):
    specs = {}
    for dataset_id in CORE_DATASET_IDS:
        document = read_json(pack_root / "datasets" / f"{dataset_id}.json")
        validate_document("dataset-spec", document)
        if document["dataset_id"] != dataset_id:
            raise ValueError(f"dataset ID mismatch for {dataset_id}")
        if not document["source"]["url"]:
            raise ValueError(f"auditable source URL required for {dataset_id}")
        specs[dataset_id] = DatasetSpec.from_dict(document)
    return specs


def battery_metadata_from_specs(pack):
    return {
        dataset_id: DatasetMetadata(
            chemistry=spec.chemistry,
            c_rated_ah=spec.c_rated_ah,
            label_source=pack.labels[spec.label_spec_id].measurement_procedure,
        )
        for dataset_id, spec in pack.datasets.items()
    }


_BATTERY_METADATA = battery_metadata_from_specs(load_battery_domain_pack())
DATASET_CHEMISTRY = {key: value.chemistry for key, value in _BATTERY_METADATA.items()}
DATASET_C_RATED_AH = {key: value.c_rated_ah for key, value in _BATTERY_METADATA.items()}
```

- [ ] **Step 4: Validate all specs and legacy metadata behavior**

Run: `python -m pytest tests/fabric/test_battery_dataset_specs.py tests/test_metadata.py -q`

Expected: all tests PASS; all seven IDs resolve through both the pack and compatibility API.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/schemas/dataset-spec.schema.json src/sohbench/fabric/schemas/label-spec.schema.json domain_packs/battery_soh/datasets domain_packs/battery_soh/labels src/sohbench/metadata.py tests/fabric/test_battery_dataset_specs.py tests/test_metadata.py
git commit -m "feat: specify seven battery datasets and labels"
```

### Task 3: Build Immutable Dataset Snapshots And Data Cards

**Files:**
- Create: `src/sohbench/fabric/dataset_registry.py`
- Create: `src/sohbench/fabric/data_cards.py`
- Create: `src/sohbench/fabric/schemas/canonical-cycle-summary.schema.json`
- Create: `src/sohbench/fabric/schemas/canonical-trajectory.schema.json`
- Create: `scripts/build_dataset_snapshots.py`
- Test: `tests/fabric/test_dataset_registry.py`
- Test: `tests/fabric/test_build_dataset_snapshots.py`
- Modify: `src/sohbench/data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Produces: `DatasetRegistry.register(spec_ref, source_root) -> DatasetSnapshot`
- Produces: `DatasetRegistry.verify(snapshot_id) -> SnapshotVerification`
- Produces: `build_data_card(snapshot, qc_report) -> dict`
- CLI example: `python scripts/build_dataset_snapshots.py --data-root D:\battery_SOH\data --datasets nasa calce_cs2 hust oxford snl_lfp uconn rwth_isea_public --output artifacts\snapshots`
- Canonical keys: `dataset_id`, `snapshot_id`, `asset_id`, `cycle_id`, `phase_id`, `sample_id`

- [ ] **Step 1: Write failing snapshot and mutation tests**

```python
def test_snapshot_digest_changes_when_source_file_changes(tmp_path, dataset_spec):
    source = write_canonical_fixture(tmp_path)
    registry = registry_at(tmp_path / "registry")
    first = registry.register(dataset_spec, source)
    (source / "features.parquet").write_bytes(b"changed")
    second = registry.register(dataset_spec, source)
    assert first.snapshot_id != second.snapshot_id


def test_data_card_reports_discovered_not_declared_counts(snapshot):
    card = build_data_card(snapshot, snapshot.qc_report)
    assert card["observed"]["assets"] == 2
    assert card["observed"]["cycles"] == 8
```

The same registry suite must cover source file list ordering, duplicate asset/cycle keys, missing target, invalid SOH range, checksum verification and a missing optional raw directory.

- [ ] **Step 2: Run tests and verify the registry is absent**

Run: `python -m pytest tests/fabric/test_dataset_registry.py tests/fabric/test_build_dataset_snapshots.py -q`

Expected: FAIL because registry and snapshot CLI are missing.

- [ ] **Step 3: Implement streaming checksums, QC manifest and data cards**

Validate cycle-summary tables and any raw trajectory index against the two canonical schemas before snapshot creation. Snapshot manifest contains DatasetSpec/LabelSpec refs, every source relative path, byte size and digest, canonical table digest, row/asset/cycle counts, feature columns, missingness, SOH range, raw-curve availability and creation tool version. Do not copy large raw files by default; record verified content refs. Store the manifest in CAS and make the snapshot ID its canonical digest.

```python
class DatasetRegistry:
    def register(self, spec_ref: SpecRef, source_root: Path) -> DatasetSnapshot:
        spec = self.specs.resolve(spec_ref)
        files = tuple(sorted(discover_source_files(spec, source_root)))
        source_entries = tuple({
            "path": path.relative_to(source_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        } for path in files)
        table = load_canonical_cycle_summary(spec, source_root)
        validate_cycle_summary(table, spec)
        qc = run_snapshot_qc(table, spec)
        manifest = snapshot_manifest(spec_ref, source_entries, table, qc)
        manifest_ref = self.cas.put_bytes(
            canonical_json(manifest), "application/json", "dataset-snapshot.json"
        )
        return DatasetSnapshot.from_manifest(manifest, snapshot_id=manifest_ref.digest)
```

- [ ] **Step 4: Run synthetic CLI and legacy loader regressions**

Run: `python -m pytest tests/fabric/test_dataset_registry.py tests/fabric/test_build_dataset_snapshots.py tests/test_data.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/dataset_registry.py src/sohbench/fabric/data_cards.py src/sohbench/fabric/schemas/canonical-cycle-summary.schema.json src/sohbench/fabric/schemas/canonical-trajectory.schema.json scripts/build_dataset_snapshots.py src/sohbench/data.py tests/fabric/test_dataset_registry.py tests/fabric/test_build_dataset_snapshots.py tests/test_data.py
git commit -m "feat: register immutable battery dataset snapshots"
```

### Task 4: Index Raw Curve Segments By Sample

**Files:**
- Create: `src/sohbench/fabric/raw_segments.py`
- Create: `src/sohbench/fabric/schemas/sample-index.schema.json`
- Create: `src/sohbench/battery/raw_curve_adapters.py`
- Test: `tests/fabric/test_raw_segments.py`
- Test: `tests/test_raw_curve_adapters.py`
- Modify: `src/sohbench/battery_dashboard_import.py`

**Interfaces:**
- Produces: `RawSegmentRef(dataset_id, snapshot_id, asset_id, cycle_id, phase_id, source_ref, row_start, row_end, columns, digest)`
- Produces: `SampleIndex.build(snapshot, adapter) -> SampleIndex`
- Produces: `SampleIndex.find(asset_id, cycle_id, phase=None) -> list[RawSegmentRef]`
- Adapters normalize: timestamp, voltage, current, temperature, charge/discharge capacity, phase and step

- [ ] **Step 1: Write failing segmentation and unavailable-reason tests**

```python
def test_segments_keep_asset_cycle_and_phase_boundaries(raw_curve_fixture):
    index = SampleIndex.build(raw_curve_fixture.snapshot, raw_curve_fixture.adapter)
    refs = index.find("cell-a", 12)
    assert [ref.phase_id for ref in refs] == ["charge", "discharge"]
    assert all(ref.row_end > ref.row_start for ref in refs)


def test_dataset_without_raw_curves_reports_reason(summary_only_snapshot):
    index = SampleIndex.build(summary_only_snapshot, NoRawCurveAdapter())
    assert index.coverage == 0.0
    assert index.unavailable_reason == "raw_curves_not_present"
```

- [ ] **Step 2: Run tests and observe missing canonical curve index**

Run: `python -m pytest tests/fabric/test_raw_segments.py tests/test_raw_curve_adapters.py -q`

Expected: FAIL because raw segment types and adapters are missing.

- [ ] **Step 3: Implement lazy adapters and compact indexes**

Read only metadata/keys while building the index. Store source row groups or byte-range-compatible locators instead of duplicating full curves. Generate stable `sample_id` from snapshot, asset and cycle IDs. Keep charge/discharge phases separate and require monotonic time within each segment. Adapt the existing dashboard importer to consume SampleIndex previews rather than maintaining an independent curve identity scheme.

```python
@dataclass(frozen=True)
class RawSegmentRef:
    dataset_id: str
    snapshot_id: str
    asset_id: str
    cycle_id: str
    phase_id: str
    source_ref: ArtifactRef
    row_start: int
    row_end: int
    columns: tuple[str, ...]
    digest: str


class SampleIndex:
    @classmethod
    def build(cls, snapshot, adapter):
        refs = []
        for segment in adapter.scan_segments(snapshot):
            validate_monotonic_time(segment)
            refs.append(RawSegmentRef.from_segment(snapshot, segment))
        reason = None if refs else adapter.unavailable_reason(snapshot)
        return cls(snapshot.snapshot_id, tuple(refs), coverage_for(snapshot, refs), reason)

    def find(self, asset_id, cycle_id, phase=None):
        return [ref for ref in self.refs if ref.asset_id == asset_id
                and ref.cycle_id == str(cycle_id)
                and (phase is None or ref.phase_id == phase)]
```

- [ ] **Step 4: Run raw-index and dashboard import regressions**

Run: `python -m pytest tests/fabric/test_raw_segments.py tests/test_raw_curve_adapters.py tests/test_battery_dashboard_import.py -q`

Expected: all tests PASS; summary-only datasets expose an explicit unavailable reason.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/raw_segments.py src/sohbench/fabric/schemas/sample-index.schema.json src/sohbench/battery/raw_curve_adapters.py src/sohbench/battery_dashboard_import.py tests/fabric/test_raw_segments.py tests/test_raw_curve_adapters.py
git commit -m "feat: index battery raw curves by sample"
```

### Task 5: Record Feature Provenance And Highlight Windows

**Files:**
- Create: `src/sohbench/fabric/feature_provenance.py`
- Create: `src/sohbench/fabric/schemas/feature-provenance.schema.json`
- Create: `src/sohbench/fabric/schemas/feature-spec.schema.json`
- Create: `src/sohbench/battery/curve_features.py`
- Create: `domain_packs/battery_soh/features/core_cycle_summary.json`
- Create: `domain_packs/battery_soh/features/lfp_curve_diagnostics.json`
- Test: `tests/fabric/test_feature_provenance.py`
- Test: `tests/test_battery_curve_features.py`
- Modify: `src/sohbench/plugins/battery_soh_features.py`

**Interfaces:**
- Produces: `FeatureProvenance(feature_id, version, value, unit, missing_status, raw_segment_refs, windows, processing, feature_state_digest, fit_scope)`
- Produces: `extract_ic_dva(curve, config) -> CurveFeatureResult`
- Produces: `extract_window_features(curve, windows, config) -> CurveFeatureResult`
- Window types: time, voltage, SOC/capacity and step ranges
- FeatureSpec declares: required signals, fit scope, parameters, units, missing policy and permitted tracks

- [ ] **Step 1: Write failing feature-value/provenance correspondence tests**

```python
def test_capacity_window_value_points_to_exact_raw_segment(charge_curve):
    result = extract_window_features(
        charge_curve, [{"id": "q_3_6", "voltage": [3.0, 3.6]}],
        {"interpolation": "linear"},
    )
    evidence = result.provenance["q_3_6"]
    assert evidence.value == pytest.approx(result.features["q_3_6"])
    assert evidence.windows[0]["voltage"] == [3.0, 3.6]
    assert evidence.raw_segment_refs == (charge_curve.segment_ref,)
```

The same feature suite must cover IC smoothing parameters, missing voltage-window coverage, fit scope, non-finite outputs and the rule that provenance JSON contains no causal/importance claim.

- [ ] **Step 2: Run tests and confirm provenance is absent**

Run: `python -m pytest tests/fabric/test_feature_provenance.py tests/test_battery_curve_features.py -q`

Expected: FAIL because feature provenance types and extractors do not exist.

- [ ] **Step 3: Implement deterministic curve features with explicit processing metadata**

Record interpolation grid, smoothing method/window/polyorder, derivative direction, clipping and missing policy. Validate each extractor invocation against its FeatureSpec. Fit any learned normalization through Phase 0 `FeaturePipeline` only on training folds. Remove global `cycle.max()` and resistance-growth features from the compatibility plugin; represent relative-age or baseline values only when PolicySpec declares a causal, fold-safe reference.

```python
def extract_window_features(curve, windows, config):
    validate_feature_config("lfp_curve_diagnostics", config)
    features, provenance = {}, {}
    for window in windows:
        selected = select_curve_window(curve, window)
        if selected.coverage < config.get("minimum_coverage", 0.95):
            features[window["id"]] = np.nan
            provenance[window["id"]] = FeatureProvenance.missing(
                window["id"], "insufficient_window_coverage", curve.segment_ref, window, config
            )
            continue
        value = integrate_capacity(selected, interpolation=config["interpolation"])
        features[window["id"]] = value
        provenance[window["id"]] = FeatureProvenance(
            feature_id=window["id"], version="1.0.0", value=value, unit="Ah",
            missing_status="present", raw_segment_refs=(curve.segment_ref,),
            windows=(window,), processing=dict(config),
            feature_state_digest=config["feature_state_digest"], fit_scope=config["fit_scope"],
        )
    return CurveFeatureResult(features, provenance)
```

- [ ] **Step 4: Run feature, leakage and plugin regressions**

Run: `python -m pytest tests/fabric/test_feature_provenance.py tests/test_battery_curve_features.py tests/fabric/test_feature_leakage.py tests/test_trusted_plugins.py -q`

Expected: all tests PASS and every emitted curve feature has provenance or an explicit missing reason.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/feature_provenance.py src/sohbench/fabric/schemas/feature-provenance.schema.json src/sohbench/fabric/schemas/feature-spec.schema.json src/sohbench/battery/curve_features.py src/sohbench/plugins/battery_soh_features.py domain_packs/battery_soh/features tests/fabric/test_feature_provenance.py tests/test_battery_curve_features.py
git commit -m "feat: trace battery features to raw curve windows"
```

### Task 6: Encode Core And LFP-specific Policies

**Files:**
- Create: `src/sohbench/fabric/schemas/policy-spec.schema.json`
- Create: `domain_packs/battery_soh/policies/core_fair_v1.json`
- Create: `domain_packs/battery_soh/policies/lfp_curve_diagnostics_v1.json`
- Create: `domain_packs/battery_soh/policies/cross_domain_v1.json`
- Create: `src/sohbench/battery/policies.py`
- Test: `tests/fabric/test_battery_policies.py`

**Interfaces:**
- Produces: `BatteryPolicy.validate_run(run_spec, dataset_snapshots, features) -> PolicyDecision`
- Core policy: common cycle-summary feature budget, 3 seeds, LOCO/LODO, asset bootstrap, complete baseline set
- LFP policy: curve/window coverage, explicit DCIR availability, smoothing sensitivity and missingness report

- [ ] **Step 1: Write failing eligibility tests**

```python
def test_core_fair_rejects_dataset_specific_dcir_feature(core_context):
    decision = core_context.policy.validate_run(
        core_context.run_spec, core_context.snapshots, [feature("dcir_growth")]
    )
    assert decision.allowed is False
    assert decision.code == "FEATURE_NOT_COMMON_TO_TRACK"


def test_lfp_diagnostic_allows_dcir_with_coverage_report(lfp_context):
    decision = lfp_context.policy.validate_run(
        lfp_context.run_spec, lfp_context.snapshots, [feature("dcir_growth")]
    )
    assert decision.allowed is True
    assert "feature_coverage" in decision.required_evidence
```

The same policy suite must cover fewer than three seeds, missing classic baselines, fixed voltage windows with inadequate coverage and cross-chemistry claims with fewer than two sources per claimed chemistry.

- [ ] **Step 2: Run tests and verify policies are not yet enforced**

Run: `python -m pytest tests/fabric/test_battery_policies.py -q`

Expected: FAIL because policy specs and evaluator are absent.

- [ ] **Step 3: Implement stable policy decisions and preflight integration**

Return `allowed`, stable code, category `POLICY`, impact and required evidence. Policies evaluate declared availability and snapshot manifests before queueing; runtime evidence rechecks actual coverage before signing. Never silently remove a feature to make a run eligible.

```python
@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    category: str
    impact: str
    required_evidence: tuple[str, ...] = ()


class BatteryPolicy:
    def validate_run(self, run_spec, dataset_snapshots, features):
        if self.spec.policy_id == "core_fair_v1":
            unavailable = [feature.feature_id for feature in features
                           if not feature.available_in_all(dataset_snapshots)]
            if unavailable:
                return PolicyDecision(False, "FEATURE_NOT_COMMON_TO_TRACK", "POLICY",
                                      ",".join(unavailable))
            if tuple(run_spec.seeds) != (0, 1, 2):
                return PolicyDecision(False, "CORE_SEEDS_REQUIRED", "POLICY", "seeds=0,1,2")
        required = tuple(self.spec.required_evidence)
        return PolicyDecision(True, "ELIGIBLE", "POLICY", "none", required)
```

- [ ] **Step 4: Run policy and API preflight tests**

Run: `python -m pytest tests/fabric/test_battery_policies.py tests/fabric/test_api_v2.py -q`

Expected: all tests PASS and rejected runs never reach `QUEUED`.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/schemas/policy-spec.schema.json domain_packs/battery_soh/policies src/sohbench/battery/policies.py tests/fabric/test_battery_policies.py
git commit -m "feat: enforce core and lfp evaluation policies"
```

### Task 7: Define Five Independent Benchmark Tracks

**Files:**
- Create: `src/sohbench/fabric/schemas/track-spec.schema.json`
- Create: `src/sohbench/fabric/schemas/protocol-spec.schema.json`
- Create: `src/sohbench/fabric/schemas/metric-spec.schema.json`
- Create: `domain_packs/battery_soh/tracks/core_fair.json`
- Create: `domain_packs/battery_soh/tracks/lfp_curve_diagnostics.json`
- Create: `domain_packs/battery_soh/tracks/cross_domain.json`
- Create: `domain_packs/battery_soh/tracks/second_life_field.json`
- Create: `domain_packs/battery_soh/tracks/deployment_inference.json`
- Create: `domain_packs/battery_soh/protocols/loco_v1.json`
- Create: `domain_packs/battery_soh/protocols/lodo_v1.json`
- Create: `domain_packs/battery_soh/protocols/cross_domain_v1.json`
- Create: `domain_packs/battery_soh/protocols/frozen_inference_v1.json`
- Create: `domain_packs/battery_soh/metrics/soh_core_v1.json`
- Create: `domain_packs/battery_soh/metrics/transfer_v1.json`
- Create: `domain_packs/battery_soh/metrics/deployment_v1.json`
- Create: `src/sohbench/battery/tracks.py`
- Test: `tests/fabric/test_battery_tracks.py`

**Interfaces:**
- Produces: `TrackRegistry.get(track_id) -> TrackSpec`
- Produces: `TrackRegistry.comparison_scope(track_id, run_spec) -> dict`
- Each TrackSpec binds: allowed run types, labels, datasets/query, protocols, metrics, policies and release gates
- ProtocolSpec declares: split unit, holdout scope, calibration scope, fit boundaries and minimum assets
- MetricSpec declares: formula/version, direction, aggregation unit, uncertainty and required prediction fields

- [ ] **Step 1: Write failing separation tests**

```python
def test_zero_shot_and_supervised_adaptation_have_different_scopes(track_registry):
    zero = track_registry.comparison_scope("cross_domain", zero_shot_run())
    adapted = track_registry.comparison_scope("cross_domain", adaptation_run())
    assert zero["comparison_key"] != adapted["comparison_key"]


def test_deployment_track_requires_frozen_inference(track_registry):
    with pytest.raises(TrackEligibilityError, match="inference"):
        track_registry.comparison_scope("deployment_inference", training_run())
```

Add tests that system-level RWTH capacity labels do not compare with cell-level reference-discharge labels and that OOD-only datasets cannot enter a supervised track.

- [ ] **Step 2: Run tests and confirm track contracts are missing**

Run: `python -m pytest tests/fabric/test_battery_tracks.py -q`

Expected: FAIL because TrackRegistry does not exist.

- [ ] **Step 3: Implement explicit track binding and comparison scopes**

Comparison scope includes TrackSpec, LabelSpec, ProtocolSpec, MetricSpec and PolicySpec digests. Validate every referenced protocol/metric document before the track is eligible. For cross-domain, encode `zero_shot`, `few_shot` and `supervised_adaptation` as distinct protocol variants. For deployment, require a ModelArtifact and report accuracy plus engineering metrics without a scalar composite score.

```python
class TrackRegistry:
    def comparison_scope(self, track_id: str, run_spec: RunSpec) -> dict:
        track = self.get(track_id)
        if run_spec.run_type not in track.allowed_run_types:
            raise TrackEligibilityError(f"{run_spec.run_type} is not allowed in {track_id}")
        protocol = self.protocols.resolve(run_spec.protocol)
        metric = self.metrics.resolve(run_spec.metric)
        policy = self.policies.resolve(run_spec.policy)
        label = self.labels.resolve(track.label_spec_id)
        scope = {
            "track": track.ref.to_dict(),
            "label": label.ref.to_dict(),
            "protocol": protocol.ref.to_dict(),
            "metric": metric.ref.to_dict(),
            "policy": policy.ref.to_dict(),
            "run_type": run_spec.run_type,
            "adaptation_mode": protocol.adaptation_mode,
        }
        return {**scope, "comparison_key": sha256_json(scope)}
```

- [ ] **Step 4: Run track, RunSpec and policy tests**

Run: `python -m pytest tests/fabric/test_battery_tracks.py tests/fabric/test_specs.py tests/fabric/test_battery_policies.py -q`

Expected: all tests PASS; incompatible tracks never share a comparison key.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/schemas/track-spec.schema.json src/sohbench/fabric/schemas/protocol-spec.schema.json src/sohbench/fabric/schemas/metric-spec.schema.json domain_packs/battery_soh/tracks domain_packs/battery_soh/protocols domain_packs/battery_soh/metrics src/sohbench/battery/tracks.py tests/fabric/test_battery_tracks.py
git commit -m "feat: separate battery benchmark tracks"
```

### Task 8: Run A Signed Seven-dataset Core Baseline

**Files:**
- Create: `src/sohbench/fabric/schemas/algorithm-spec.schema.json`
- Create: `domain_packs/battery_soh/algorithms/ridge_v1.json`
- Create: `domain_packs/battery_soh/algorithms/random_forest_v1.json`
- Create: `domain_packs/battery_soh/algorithms/gradient_boosting_v1.json`
- Create: `domain_packs/battery_soh/algorithms/hist_gradient_boosting_v1.json`
- Create: `domain_packs/battery_soh/experiments/seven_dataset_core_v1.json`
- Create: `scripts/run_signed_core_benchmark.py`
- Create: `src/sohbench/battery/core_benchmark.py`
- Test: `tests/fabric/test_core_benchmark.py`
- Test: `tests/fabric/test_core_benchmark_cli.py`
- Modify: `src/sohbench/cli.py`

**Interfaces:**
- CLI inputs: data root, registry root, output root, candidate IDs, `--smoke` or `--full`
- Core matrix: seven DatasetSnapshots, LOCO and LODO, declared common features, classic baselines, seeds 0/1/2
- Outputs: signed EvidencePack, comparable leaderboard, per-domain/worst-domain metrics, coverage/failures and asset-level CI
- AlgorithmSpec declares: package ref, hyperparameters, seed injection path, resource class and supported feature schema

- [ ] **Step 1: Write a small seven-dataset matrix test**

Build seven two-cell synthetic snapshots and two deterministic candidates. Assert the matrix contains all dataset holdouts, three seeds, attempted/succeeded counts, baseline rows, a stable comparison key and a verifiable EvidencePack. Deliberately fail one unit and assert `outcome="partial"`, coverage below one and no denominator loss.

```python
def test_seven_dataset_core_matrix_preserves_failures(tmp_path, seven_snapshot_fixture):
    services = fabric_services(tmp_path, snapshots=seven_snapshot_fixture)
    run = run_core_benchmark(CoreBenchmarkConfig(
        mode="full", candidate_ids=("ridge_v1", "fails_one_fold"),
        reviewer="test-reviewer", review_reason="synthetic acceptance",
    ), services)
    assert set(run["coverage"]["dataset_ids"]) == set(CORE_DATASET_IDS)
    assert run["seeds"] == [0, 1, 2]
    assert {row["protocol"] for row in run["leaderboard"]} == {"loco_v1", "lodo_v1"}
    assert run["outcome"] == "partial"
    assert run["execution"]["attempted"] > run["execution"]["succeeded"]
    assert run["coverage"]["fraction"] < 1.0
    assert verify_evidence_pack(run["evidence_path"])["ok"] is True
```

- [ ] **Step 2: Run the synthetic benchmark test before the command exists**

Run: `python -m pytest tests/fabric/test_core_benchmark.py tests/fabric/test_core_benchmark_cli.py -q`

Expected: FAIL because the signed core benchmark command is missing.

- [ ] **Step 3: Implement smoke/full modes and signing preconditions**

Validate every classic baseline against AlgorithmSpec and require its seed injection path before queueing. `--smoke` runs one classic candidate with one fold per protocol while retaining all seven dataset identities; it cannot be labelled a formal leaderboard. `--full` runs the frozen experiment spec and may transition `REVIEW -> SIGNED` only after EvidencePack verification, complete required statistics, policy pass and reviewer identity. Never auto-sign partial runs when the policy requires full coverage.

```python
def run_core_benchmark(config: CoreBenchmarkConfig, services) -> dict:
    experiment = services.experiments.load("seven_dataset_core_v1")
    snapshots = [services.datasets.resolve(dataset_id) for dataset_id in experiment.dataset_ids]
    candidates = services.algorithms.resolve_many(config.candidate_ids or experiment.candidate_ids)
    for candidate in candidates:
        validate_algorithm_seed_path(candidate)
    spec = experiment.to_run_spec(
        snapshots=snapshots,
        candidates=candidates[:1] if config.mode == "smoke" else candidates,
        seeds=(0,) if config.mode == "smoke" else (0, 1, 2),
        smoke=config.mode == "smoke",
    )
    decision = services.preflight.validate(spec)
    if not decision.allowed:
        raise PolicyError(decision.code)
    run = services.runs.execute_and_wait(spec)
    if config.mode == "full" and services.release_gate.eligible(run):
        run = services.signer.sign_run(run, config.reviewer, config.review_reason)
    return run
```

- [ ] **Step 4: Run the synthetic acceptance and a local metadata-only preflight**

Run: `python -m pytest tests/fabric/test_core_benchmark.py tests/fabric/test_core_benchmark_cli.py -q`

Run: `python scripts/run_signed_core_benchmark.py --data-root D:\battery_SOH\data --preflight --output artifacts\preflight`

Expected: tests PASS. Preflight exits `0` only when all seven sources resolve; otherwise it exits `2` and prints the exact missing/invalid dataset IDs without starting a run.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/schemas/algorithm-spec.schema.json domain_packs/battery_soh/algorithms domain_packs/battery_soh/experiments/seven_dataset_core_v1.json scripts/run_signed_core_benchmark.py src/sohbench/battery/core_benchmark.py src/sohbench/cli.py tests/fabric/test_core_benchmark.py tests/fabric/test_core_benchmark_cli.py
git commit -m "feat: add signed seven dataset core benchmark"
```

### Task 9: Register The UConn Second-life LFP Dataset Separately

**Files:**
- Create: `domain_packs/battery_soh/datasets/uconn_second_life.json`
- Create: `domain_packs/battery_soh/labels/second_life_reference_capacity.json`
- Create: `src/sohbench/battery/uconn_second_life.py`
- Create: `scripts/register_uconn_second_life.py`
- Test: `tests/fabric/test_uconn_second_life.py`
- Create: `tests/fixtures/uconn_second_life/manifest.json`

**Interfaces:**
- Source page: `https://digitalcommons.lib.uconn.edu/reil_datasets/3/`
- Produces: `inspect_uconn_archive(path) -> SourceInspection`
- Produces: `register_uconn_second_life(path, registry) -> DatasetSnapshot`
- CLI requires: local archive path, expected SHA-256 and explicit license acknowledgement

- [ ] **Step 1: Write failing archive identity and label tests**

```python
def test_second_life_dataset_is_not_aliased_to_existing_uconn(specs):
    assert specs["uconn_second_life"].dataset_id != specs["uconn"].dataset_id
    assert specs["uconn_second_life"].label_spec_id == "second_life_reference_capacity"


def test_registration_requires_matching_digest(tmp_path, fixture_archive):
    with pytest.raises(SourceDigestMismatch):
        register_uconn_second_life(fixture_archive, registry_at(tmp_path), expected_sha256="0" * 64)
```

The same adapter suite must cover the 108-asset declaration versus observed count, source URL, CC BY 4.0 metadata, required columns and no automatic network access in the registration function.

- [ ] **Step 2: Run tests and verify no separate adapter exists**

Run: `python -m pytest tests/fabric/test_uconn_second_life.py -q`

Expected: FAIL because the dataset spec and adapter are missing.

- [ ] **Step 3: Implement explicit local-archive onboarding**

The script may print the authoritative source page but does not download unless the user passes `--download` and confirms the estimated size/license. Registration streams the archive, checks the user-supplied digest, inspects without extracting outside a temporary root, validates observed assets and writes DatasetSnapshot/QC manifests. Keep any discrepancy between declared and observed asset count as a blocking QC issue.

```python
def register_uconn_second_life(path, registry, expected_sha256, license_acknowledged=False):
    if not license_acknowledged:
        raise LicenseAcknowledgementRequired("CC BY 4.0 acknowledgement required")
    observed_digest = sha256_file(path)
    if observed_digest != expected_sha256:
        raise SourceDigestMismatch(expected_sha256, observed_digest)
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise UnsafeArchivePath(member.filename)
            archive.extractall(root)
        inspection = inspect_uconn_archive(root)
        if inspection.observed_assets != 108:
            raise DatasetQCError(f"expected 108 assets, observed {inspection.observed_assets}")
        return registry.register("uconn_second_life", root)
```

- [ ] **Step 4: Run fixture registration and source-metadata tests**

Run: `python -m pytest tests/fabric/test_uconn_second_life.py -q`

Expected: all tests PASS; no network is accessed by default.

- [ ] **Step 5: Commit**

```powershell
git add domain_packs/battery_soh/datasets/uconn_second_life.json domain_packs/battery_soh/labels/second_life_reference_capacity.json src/sohbench/battery/uconn_second_life.py scripts/register_uconn_second_life.py tests/fabric/test_uconn_second_life.py tests/fixtures/uconn_second_life/manifest.json
git commit -m "feat: register uconn second life lfp dataset"
```

### Task 10: Guard Field And OOD Datasets From Supervised Ranking

**Files:**
- Create: `domain_packs/battery_soh/datasets/real_ess.json`
- Create: `domain_packs/battery_soh/datasets/m5bat.json`
- Create: `domain_packs/battery_soh/protocols/ood_drift.json`
- Create: `src/sohbench/battery/field_policy.py`
- Test: `tests/fabric/test_field_ood_policy.py`

**Interfaces:**
- Produces: `classify_label_readiness(dataset_spec, snapshot) -> LabelReadiness`
- Produces: `FieldTrackPolicy.validate(dataset, requested_track) -> PolicyDecision`
- OOD evidence: health indicators, drift, operating envelope, missingness and abstention; no supervised SOH leaderboard without a valid LabelSpec

- [ ] **Step 1: Write failing label-readiness tests**

```python
@pytest.mark.parametrize("dataset_id", ["real_ess", "m5bat"])
def test_unverified_field_labels_cannot_enter_core_rank(dataset_id, field_context):
    decision = field_context.policy.validate(field_context.dataset(dataset_id), "core_fair")
    assert decision.allowed is False
    assert decision.code == "SUPERVISED_LABEL_NOT_VERIFIED"
```

Add a test that the same snapshot is allowed in `second_life_field` with protocol `ood_drift` and verdict `not_applicable` for supervised accuracy.

- [ ] **Step 2: Run tests and demonstrate the current missing guard**

Run: `python -m pytest tests/fabric/test_field_ood_policy.py -q`

Expected: FAIL because field label readiness is not modeled.

- [ ] **Step 3: Implement auditable readiness and track gating**

Label readiness values are `verified`, `intermittent_reference`, `proxy_only`, and `unlabeled`. Only `verified` may enter supervised core/transfer leaderboards; intermittent labels require a dedicated protocol; proxy/unlabeled datasets report OOD/HI evidence. A later LabelSpec version can unlock ranking without rewriting the old snapshot.

```python
class LabelReadiness(str, Enum):
    VERIFIED = "verified"
    INTERMITTENT_REFERENCE = "intermittent_reference"
    PROXY_ONLY = "proxy_only"
    UNLABELED = "unlabeled"


class FieldTrackPolicy:
    SUPERVISED_TRACKS = frozenset({"core_fair", "cross_domain"})

    def validate(self, dataset, requested_track):
        readiness = classify_label_readiness(dataset.spec, dataset.snapshot)
        if requested_track in self.SUPERVISED_TRACKS and readiness is not LabelReadiness.VERIFIED:
            return PolicyDecision(False, "SUPERVISED_LABEL_NOT_VERIFIED", "POLICY", readiness.value)
        if requested_track == "second_life_field" and readiness is not LabelReadiness.VERIFIED:
            return PolicyDecision(True, "OOD_ONLY", "POLICY", "supervised verdict not applicable",
                                  ("drift", "operating_envelope", "abstention"))
        return PolicyDecision(True, "ELIGIBLE", "POLICY", "none")
```

- [ ] **Step 4: Run field and track tests**

Run: `python -m pytest tests/fabric/test_field_ood_policy.py tests/fabric/test_battery_tracks.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add domain_packs/battery_soh/datasets/real_ess.json domain_packs/battery_soh/datasets/m5bat.json domain_packs/battery_soh/protocols/ood_drift.json src/sohbench/battery/field_policy.py tests/fabric/test_field_ood_policy.py
git commit -m "feat: separate field ood evidence from soh ranking"
```

### Task 11: Expose Sample And Curve Evidence APIs

**Files:**
- Create: `src/sohbench/fabric/evidence_queries.py`
- Create: `src/sohbench/battery/curve_queries.py`
- Test: `tests/fabric/test_evidence_queries.py`
- Test: `tests/fabric/test_curve_api.py`
- Modify: `src/sohbench/api_v2.py`

**Interfaces:**
- Adds: `GET /api/v2/evidence/{evidence_id}/samples`
- Adds: `GET /api/v2/evidence/{evidence_id}/samples/{sample_id}/curves?cycle=<id>&max_points=<n>`
- Sample fields: true/predicted SOH, interval, error, metric contribution, QC, raw availability and provenance refs
- Curve fields: raw/processed V/I/T/Q, IC/DVA/DCIR when available, highlight windows, processing parameters and coverage

- [ ] **Step 1: Write failing query, pagination and downsampling tests**

```python
def test_curve_response_links_prediction_and_provenance(api_client, signed_evidence):
    sample = api_client.get(f"/api/v2/evidence/{signed_evidence}/samples?limit=1").json()["items"][0]
    response = api_client.get(
        f"/api/v2/evidence/{signed_evidence}/samples/{sample['sample_id']}/curves?max_points=500"
    ).json()
    assert response["sample_id"] == sample["sample_id"]
    assert len(response["curves"]["voltage"]["x"]) <= 500
    assert response["lineage"]["prediction_record"]
    assert response["feature_highlights"][0]["raw_segment_ref"]
```

The same query suite must cover stable cursor pagination, largest-triangle downsampling preserving endpoints, max-points limits, no-raw availability reasons and attempts to query a sample outside the evidence pack.

- [ ] **Step 2: Run API tests and observe missing endpoints**

Run: `python -m pytest tests/fabric/test_evidence_queries.py tests/fabric/test_curve_api.py -q`

Expected: FAIL with 404 or missing query service.

- [ ] **Step 3: Implement indexed, bounded evidence reads**

Resolve only files listed in a verified EvidencePack. Join PredictionRecord, MetricContribution, RawSegmentRef and FeatureProvenance by stable IDs. Read a single asset/cycle segment, cache by source digest and query shape, and apply deterministic downsampling after preserving highlighted window boundaries. Return `raw_curves_not_present`, `phase_not_recorded` or `feature_not_computed` explicitly.

```python
class EvidenceQueryService:
    def sample_curves(self, evidence_id, sample_id, cycle_id=None, max_points=2000):
        if not 100 <= max_points <= 10_000:
            raise QueryContractError("max_points must be between 100 and 10000")
        pack = self.evidence.verify_and_open(evidence_id)
        sample = self.index(pack).require_sample(sample_id)
        if not sample.raw_available:
            return curve_unavailable(sample, "raw_curves_not_present")
        refs = self.raw_segments.find(sample.asset_id, cycle_id or sample.cycle_id)
        curves = [self.curves.read_segment(pack, ref) for ref in refs]
        highlights = self.provenance.for_sample(pack, sample_id)
        return build_curve_response(
            sample, curves, highlights,
            downsample=lambda series: largest_triangle(series, max_points, highlights),
        )
```

- [ ] **Step 4: Run curve APIs, evidence integrity and memory-bound tests**

Run: `python -m pytest tests/fabric/test_evidence_queries.py tests/fabric/test_curve_api.py tests/fabric/test_evidence.py -q`

Expected: all tests PASS and a 100,000-point fixture response contains at most the requested points without loading other cycles.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/evidence_queries.py src/sohbench/battery/curve_queries.py src/sohbench/api_v2.py tests/fabric/test_evidence_queries.py tests/fabric/test_curve_api.py
git commit -m "feat: expose sample and curve evidence api"
```

### Task 12: Close The Battery Benchmark Acceptance Gate

**Files:**
- Create: `tests/fabric/test_phase_battery_gate.py`
- Create: `docs/verification/battery-domain-benchmark-report.md`
- Modify: `README.md`

**Interfaces:**
- Verifies: seven specs/snapshots, label and policy separation, sample lineage, fair comparison and signing gates
- Produces: a reproducible small signed baseline report

- [ ] **Step 1: Add an end-to-end synthetic seven-dataset acceptance test**

The test registers seven snapshots, runs LOCO/LODO with seeds 0/1/2, verifies no target reaches candidate inputs, checks per-asset CI and failures, signs the eligible EvidencePack, then follows one worst sample to its raw voltage/IC feature window. Include one summary-only dataset and assert its curve reason and aggregate metrics both remain correct.

```python
def test_battery_gate_links_signed_result_to_curve_evidence(tmp_path, battery_gate_fixture):
    result = battery_gate_fixture.run_signed_core(tmp_path)
    assert result["execution_state"] == "SIGNED"
    assert result["comparison_key"]
    assert result["statistics"]["bootstrap_unit"] == "asset"
    worst = result["worst_samples"][0]
    curves = battery_gate_fixture.api.sample_curves(
        result["evidence_id"], worst["sample_id"], worst["cycle_id"], max_points=500
    )
    assert curves["lineage"]["prediction_record"]
    assert curves["feature_highlights"][0]["raw_segment_ref"]
    summary_only = battery_gate_fixture.summary_only_sample(result)
    unavailable = battery_gate_fixture.api.sample_curves(
        result["evidence_id"], summary_only["sample_id"], summary_only["cycle_id"]
    )
    assert unavailable["reason"] == "raw_curves_not_present"
```

- [ ] **Step 2: Run the battery acceptance gate**

Run: `python -m pytest tests/fabric/test_phase_battery_gate.py -q`

Expected: all tests PASS; the signed evidence verifies and all seven dataset IDs appear in coverage.

- [ ] **Step 3: Run domain and regression suites**

```powershell
python -m pytest tests/fabric/test_battery_dataset_specs.py tests/fabric/test_dataset_registry.py tests/fabric/test_raw_segments.py tests/fabric/test_feature_provenance.py tests/fabric/test_battery_policies.py tests/fabric/test_battery_tracks.py tests/fabric/test_core_benchmark.py tests/fabric/test_curve_api.py -q
python -m pytest -q
```

Expected: both commands exit `0`.

- [ ] **Step 4: Write the battery-domain verification report**

Record spec digests, observed local data readiness for each of seven datasets, smoke EvidencePack ID, track eligibility table, label gaps, raw-curve coverage, LFP-specific availability and the exact conditions before starting the full matrix. Do not present preflight or smoke results as a formal SOTA ranking.

- [ ] **Step 5: Commit**

```powershell
git add tests/fabric/test_phase_battery_gate.py docs/verification/battery-domain-benchmark-report.md README.md
git commit -m "test: close battery domain benchmark gate"
```

## Battery Review Gate

- [ ] Seven current datasets have schema-valid DatasetSpec, LabelSpec, source metadata and immutable snapshot manifests.
- [ ] Core, LFP curve, cross-domain, second-life/field and deployment runs have distinct comparison scopes.
- [ ] The core benchmark uses only common feature budget, three seeds, LOCO/LODO, classic baselines, asset-level CI and explicit failure coverage.
- [ ] Every available feature highlight resolves through `sample_id -> RawSegmentRef -> FeatureProvenance -> PredictionRecord -> MetricContribution`.
- [ ] Missing raw curves and unverified labels are visible limitations, never synthesized or silently ranked.
- [ ] UConn second-life is registered as a distinct dataset with a verified digest and label definition.
- [ ] A small seven-dataset signed baseline passes before any full deep-model matrix starts.

## Plan Verification Commands

```powershell
python -m pytest tests/fabric/test_domain_packs.py tests/fabric/test_battery_dataset_specs.py tests/fabric/test_dataset_registry.py -q
python -m pytest tests/fabric/test_raw_segments.py tests/fabric/test_feature_provenance.py tests/fabric/test_battery_policies.py tests/fabric/test_battery_tracks.py -q
python -m pytest tests/fabric/test_core_benchmark.py tests/fabric/test_uconn_second_life.py tests/fabric/test_field_ood_policy.py -q
python -m pytest tests/fabric/test_evidence_queries.py tests/fabric/test_curve_api.py tests/fabric/test_phase_battery_gate.py -q
python -m pytest -q
```

Expected: every command exits `0`; synthetic acceptance produces a verifiable signed EvidencePack and no unverified field dataset enters a supervised leaderboard.
