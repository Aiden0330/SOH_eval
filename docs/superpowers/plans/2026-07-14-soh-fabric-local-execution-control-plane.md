# SOH Fabric Local Execution & Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Phase 0 的可信测评内核迁移到可恢复的本机控制面，使 API 不再加载上传源码或 Pickle，并为训练、适配和冻结推理提供可取消、可超时、可重试、可审计的独立 Worker。

**Architecture:** SQLite 以追加事件保存运行生命周期，ContentAddressedStore 以 SHA-256 原子保存输入和产物；Scheduler 只调度资源引用，LocalProcessBackend 通过受控子进程执行 T1/T2 Worker。API v2 只读写契约、状态和引用，v1 通过适配器调用同一服务。T3 在容器后端落地前明确拒绝。

**Tech Stack:** Python 3.11+、sqlite3、subprocess、resource/psutil（按平台能力）、jsonschema、pytest。

## Global Constraints

- API 进程不得 `import` 上传模块，也不得调用 `pickle.load` 或 `pickle.loads`。
- SQLite 开启 WAL、foreign keys 和 busy timeout；每个状态转换与 attempt 在同一事务提交。
- CAS 路径只能由摘要派生，禁止用户输入参与目标路径拼接。
- Worker 请求只含资源引用、摘要和无目标测试视图；测试真值只进入 Metric Worker。
- `execution_state`、`outcome` 和 `verdict` 分开持久化。
- 失败重试创建新 attempt，不修改或删除旧 attempt 和事件。
- T2 是可信本机隔离，不宣称抵御恶意系统调用；T3 必须在未配置容器后端时返回 `POLICY` 拒绝。
- Windows 与 Linux 都必须通过核心状态、超时和原子存储测试；平台资源限制差异进入 capability 字段。

---

### Task 1: Add The SQLite Event Store

**Files:**
- Create: `src/sohbench/fabric/db.py`
- Create: `src/sohbench/fabric/sql/__init__.py`
- Create: `src/sohbench/fabric/sql/001_initial.sql`
- Test: `tests/fabric/test_event_store.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `EventStore(path: Path)`
- Produces: `EventStore.create_run(run_id, spec, comparison_key, idempotency_key) -> dict`
- Produces: `EventStore.append_event(run_id, event_type, payload, actor, attempt, expected_version) -> dict`
- Produces: `EventStore.get_run(run_id) -> dict`
- Produces: `EventStore.list_events(run_id, after_sequence=0) -> list[dict]`

- [ ] **Step 1: Write failing persistence and concurrency tests**

```python
def test_run_and_events_survive_reopen(tmp_path):
    path = tmp_path / "fabric.sqlite3"
    first = EventStore(path)
    first.create_run("RUN-a", {"run_type": "training"}, "cmp", "idem-a")
    first.append_event("RUN-a", "VALIDATED", {}, "system", 1, expected_version=0)
    second = EventStore(path)
    assert second.get_run("RUN-a")["execution_state"] == "VALIDATED"
    assert [item["sequence"] for item in second.list_events("RUN-a")] == [1, 2]


def test_idempotency_key_returns_existing_run(tmp_path):
    store = EventStore(tmp_path / "fabric.sqlite3")
    first = store.create_run("RUN-a", {"run_type": "training"}, "cmp", "same")
    second = store.create_run("RUN-b", {"run_type": "training"}, "cmp", "same")
    assert second["run_id"] == first["run_id"]
```

Add a stale `expected_version` test expecting `ConcurrentUpdateError` and a foreign-key test that rejects an event for an unknown run.

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `python -m pytest tests/fabric/test_event_store.py -q`

Expected: FAIL because `sohbench.fabric.db` does not exist.

- [ ] **Step 3: Implement schema migration and transactional methods**

`001_initial.sql` must create `schema_migrations`, `runs`, `run_attempts`, `run_events`, `artifact_refs`, and `idempotency_keys`. Store canonical JSON text, timestamps in UTC ISO-8601, and a monotonically increasing per-run `sequence` and `version`. Configure every connection with:

```python
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("PRAGMA foreign_keys=ON")
connection.execute("PRAGMA busy_timeout=5000")
```

Use `BEGIN IMMEDIATE` for append and compare the stored version before inserting the event and updating the run snapshot.

- [ ] **Step 4: Verify persistence, idempotency and concurrent update behavior**

Run: `python -m pytest tests/fabric/test_event_store.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/db.py src/sohbench/fabric/sql src/sohbench/fabric/__init__.py tests/fabric/test_event_store.py
git commit -m "feat: add persistent run event store"
```

### Task 2: Enforce The Run State Machine

**Files:**
- Create: `src/sohbench/fabric/state.py`
- Test: `tests/fabric/test_run_state.py`
- Modify: `src/sohbench/fabric/db.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `ExecutionState` enum
- Produces: `RunStateMachine.transition(current, requested) -> ExecutionState`
- Produces: `RunStateMachine.begin_attempt(previous_state, previous_attempt, requested_attempt, retryable) -> ExecutionState`
- Produces: `replay_run(events) -> RunSnapshot`
- Extends event payload with: `actor`, `reason`, `attempt`, `checkpoint`, `occurred_at`

- [ ] **Step 1: Write the transition-table and replay tests**

```python
@pytest.mark.parametrize("left,right", [
    ("DRAFT", "VALIDATED"), ("VALIDATED", "QUEUED"),
    ("QUEUED", "RUNNING"), ("RUNNING", "AGGREGATING"),
    ("AGGREGATING", "REVIEW"), ("REVIEW", "SIGNED"),
    ("RUNNING", "TIMED_OUT"), ("REVIEW", "INVALIDATED"),
])
def test_valid_transitions(left, right):
    assert RunStateMachine.transition(left, right).value == right


def test_signed_run_cannot_return_to_running():
    with pytest.raises(InvalidTransitionError):
        RunStateMachine.transition("SIGNED", "RUNNING")


def test_retry_starts_a_new_attempt_without_rewriting_failed_attempt():
    assert RunStateMachine.begin_attempt("FAILED", 1, 2, retryable=True).value == "QUEUED"
    with pytest.raises(InvalidTransitionError):
        RunStateMachine.begin_attempt("FAILED", 1, 1, retryable=True)
```

Add a replay test that rebuilds the same snapshot twice and asserts the three independent fields: `execution_state="REVIEW"`, `outcome="partial"`, `verdict="warn"`.

- [ ] **Step 2: Verify tests fail before the state model exists**

Run: `python -m pytest tests/fabric/test_run_state.py -q`

Expected: FAIL because state types are missing.

- [ ] **Step 3: Implement exact lifecycle transitions**

Allow only the transitions in the approved design: `DRAFT -> VALIDATED -> QUEUED -> RUNNING -> AGGREGATING -> REVIEW -> SIGNED`; cancellation from `QUEUED/RUNNING`; timeout and failure from `RUNNING`; failure from `AGGREGATING`; invalidation from `REVIEW`. `SIGNED`, `FAILED`, `CANCELLED`, `TIMED_OUT`, and `INVALIDATED` are terminal for their attempt. A retry is not a transition of the failed attempt: `begin_attempt` requires an incremented attempt number and retry-policy approval, appends `ATTEMPT_CREATED`, and projects the new attempt as `QUEUED` while preserving all prior events. Make `EventStore.append_event` call the state machine before committing.

```python
class ExecutionState(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AGGREGATING = "AGGREGATING"
    REVIEW = "REVIEW"
    SIGNED = "SIGNED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    INVALIDATED = "INVALIDATED"


ALLOWED_TRANSITIONS = {
    ExecutionState.DRAFT: {ExecutionState.VALIDATED},
    ExecutionState.VALIDATED: {ExecutionState.QUEUED},
    ExecutionState.QUEUED: {ExecutionState.RUNNING, ExecutionState.CANCELLED},
    ExecutionState.RUNNING: {ExecutionState.AGGREGATING, ExecutionState.CANCELLED,
                             ExecutionState.TIMED_OUT, ExecutionState.FAILED},
    ExecutionState.AGGREGATING: {ExecutionState.REVIEW, ExecutionState.FAILED},
    ExecutionState.REVIEW: {ExecutionState.SIGNED, ExecutionState.INVALIDATED},
}


class RunStateMachine:
    @staticmethod
    def transition(current, requested):
        left, right = ExecutionState(current), ExecutionState(requested)
        if right not in ALLOWED_TRANSITIONS.get(left, set()):
            raise InvalidTransitionError(f"{left.value} -> {right.value}")
        return right

    @staticmethod
    def begin_attempt(previous_state, previous_attempt, requested_attempt, retryable):
        terminal = ExecutionState(previous_state)
        if terminal not in {ExecutionState.FAILED, ExecutionState.TIMED_OUT}:
            raise InvalidTransitionError("only failed or timed-out attempts are retryable")
        if not retryable or requested_attempt != previous_attempt + 1:
            raise InvalidTransitionError("retry must create exactly the next approved attempt")
        return ExecutionState.QUEUED
```

- [ ] **Step 4: Run state and database tests**

Run: `python -m pytest tests/fabric/test_run_state.py tests/fabric/test_event_store.py -q`

Expected: all tests PASS and replay is deterministic.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/state.py src/sohbench/fabric/db.py src/sohbench/fabric/__init__.py tests/fabric/test_run_state.py
git commit -m "feat: enforce auditable run lifecycle"
```

### Task 3: Add Atomic Content-addressed Storage

**Files:**
- Create: `src/sohbench/fabric/artifacts.py`
- Test: `tests/fabric/test_artifact_store.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `ArtifactRef(digest, size, media_type, logical_name)`
- Produces: `ContentAddressedStore.put_bytes(data, media_type, logical_name) -> ArtifactRef`
- Produces: `ContentAddressedStore.put_file(path, media_type, logical_name) -> ArtifactRef`
- Produces: `ContentAddressedStore.open(ref, mode="rb")`
- Produces: `ContentAddressedStore.verify(ref) -> None`

- [ ] **Step 1: Write failing deduplication, tamper and traversal tests**

```python
def test_equal_content_is_deduplicated(tmp_path):
    store = ContentAddressedStore(tmp_path / "cas")
    one = store.put_bytes(b"model", "application/octet-stream", "a.pkl")
    two = store.put_bytes(b"model", "application/octet-stream", "../../b.pkl")
    assert one.digest == two.digest
    assert len(list((tmp_path / "cas" / "sha256").rglob("*.blob"))) == 1


def test_open_detects_tampering(tmp_path):
    store = ContentAddressedStore(tmp_path / "cas")
    ref = store.put_bytes(b"evidence", "application/json", "result.json")
    store.path_for(ref).write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        store.verify(ref)
```

- [ ] **Step 2: Run tests and observe the missing CAS implementation**

Run: `python -m pytest tests/fabric/test_artifact_store.py -q`

Expected: FAIL because `ContentAddressedStore` is missing.

- [ ] **Step 3: Implement atomic writes and safe logical metadata**

Write to a random temporary file inside the digest shard, flush and fsync, then use `os.replace`. If the digest already exists, verify size and digest before discarding the temporary file. Keep `logical_name` as metadata only after `Path(name).name` normalization; never use it to locate the blob.

```python
@dataclass(frozen=True)
class ArtifactRef:
    digest: str
    size: int
    media_type: str
    logical_name: str


class ContentAddressedStore:
    def put_bytes(self, data: bytes, media_type: str, logical_name: str) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / "sha256" / digest[:2] / f"{digest}.blob"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{digest}.{uuid.uuid4().hex}.tmp"
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            temporary.unlink()
        else:
            os.replace(temporary, target)
        ref = ArtifactRef(digest, len(data), media_type, Path(logical_name).name)
        self.verify(ref)
        return ref
```

- [ ] **Step 4: Run CAS tests twice to cover the existing-blob path**

Run: `python -m pytest tests/fabric/test_artifact_store.py -q`

Run again: `python -m pytest tests/fabric/test_artifact_store.py -q`

Expected: both runs PASS without extra blobs.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/artifacts.py src/sohbench/fabric/__init__.py tests/fabric/test_artifact_store.py
git commit -m "feat: add atomic content addressed storage"
```

### Task 4: Define The Worker Wire Protocol

**Files:**
- Create: `src/sohbench/fabric/worker_protocol.py`
- Create: `src/sohbench/fabric/worker_main.py`
- Create: `src/sohbench/fabric/workers/__init__.py`
- Test: `tests/fabric/test_worker_protocol.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `WorkerRequest`, `WorkerEvent`, `WorkerResult`
- Produces: `encode_message(value) -> bytes`
- Produces: `decode_message(line) -> WorkerRequest | WorkerEvent | WorkerResult`
- CLI example: `python -m sohbench.fabric.worker_main --request work/request.json --result work/result.json`

- [ ] **Step 1: Write failing round-trip and truth-isolation tests**

```python
def test_worker_request_round_trip_has_only_refs():
    request = WorkerRequest(
        protocol_version="1.0.0", run_id="RUN-a", attempt=1,
        worker_kind="training", input_refs=(artifact_ref(),),
        output_root="OUTPUT", limits={"wall_seconds": 30},
    )
    restored = decode_message(encode_message(request))
    assert restored == request
    assert "y_test" not in encode_message(request).decode("utf-8")
    assert "soh" not in encode_message(request).decode("utf-8")
```

Add schema rejection tests for unknown `worker_kind`, absolute output paths, protocol version mismatch and an input reference with a non-SHA-256 digest.

- [ ] **Step 2: Verify the protocol tests fail**

Run: `python -m pytest tests/fabric/test_worker_protocol.py -q`

Expected: FAIL because the wire types do not exist.

- [ ] **Step 3: Implement versioned JSON-lines messages**

Serialize only primitive JSON values. Worker requests reference CAS blobs and a worker-private output directory; they never embed DataFrames or test labels. `worker_main` validates the request before dispatch, emits structured progress to stdout, and writes exactly one atomic result document. Unknown worker kinds return a structured `INPUT_CONTRACT` failure with a nonzero process exit.

```python
@dataclass(frozen=True)
class WorkerRequest:
    protocol_version: str
    run_id: str
    attempt: int
    worker_kind: Literal["training", "adaptation", "inference", "metric", "ingest"]
    input_refs: tuple[ArtifactRef, ...]
    output_root: str
    limits: dict[str, int]


def encode_message(value) -> bytes:
    return canonical_json({"message_type": type(value).__name__, "payload": asdict(value)}) + b"\n"


def decode_message(line: bytes):
    document = json.loads(line)
    if document["message_type"] != "WorkerRequest":
        raise WorkerProtocolError("unexpected message type")
    request = WorkerRequest(**normalize_request_payload(document["payload"]))
    validate_worker_request(request)
    return request
```

- [ ] **Step 4: Run protocol and module-entry smoke tests**

Run: `python -m pytest tests/fabric/test_worker_protocol.py -q`

Expected: all tests PASS; malformed requests never dispatch a worker.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/worker_protocol.py src/sohbench/fabric/worker_main.py src/sohbench/fabric/workers src/sohbench/fabric/__init__.py tests/fabric/test_worker_protocol.py
git commit -m "feat: define isolated worker protocol"
```

### Task 5: Implement The Local Process Backend

**Files:**
- Create: `src/sohbench/fabric/execution.py`
- Create: `src/sohbench/fabric/process_limits.py`
- Test: `tests/fabric/test_local_process_backend.py`
- Create: `tests/fixtures/workers/echo_worker.py`
- Create: `tests/fixtures/workers/hanging_worker.py`

**Interfaces:**
- Produces: `ExecutionBackend` protocol
- Produces: `LocalProcessBackend.start(request) -> ProcessHandle`
- Produces: `ProcessHandle.wait(timeout) -> WorkerResult`
- Produces: `ProcessHandle.cancel(grace_seconds=2) -> WorkerResult`
- Produces: `ProcessCapabilities(network_isolation, memory_limit, cpu_limit)`

- [ ] **Step 1: Write failing success, timeout and cancellation tests**

```python
def test_hanging_worker_is_terminated_without_killing_caller(tmp_path):
    backend = LocalProcessBackend(work_root=tmp_path)
    handle = backend.start(worker_request("hanging", wall_seconds=1))
    result = handle.wait(timeout=3)
    assert result.outcome == "timed_out"
    assert result.failure_category == "RESOURCE"
    assert handle.is_alive() is False


def test_cancel_is_idempotent(tmp_path):
    backend = LocalProcessBackend(work_root=tmp_path)
    handle = backend.start(worker_request("hanging", wall_seconds=30))
    assert handle.cancel().outcome == "cancelled"
    assert handle.cancel().outcome == "cancelled"
```

- [ ] **Step 2: Run tests and verify no backend exists**

Run: `python -m pytest tests/fabric/test_local_process_backend.py -q`

Expected: FAIL because `LocalProcessBackend` is missing.

- [ ] **Step 3: Implement process groups, clean environments and capability reporting**

Launch `sys.executable -I -m sohbench.fabric.worker_main` with a minimal environment, dedicated working directory and redirected pipes. On Windows use a new process group and Job Object when available; on POSIX use a new session and `resource.setrlimit`. Timeout first requests graceful termination, then kills the full process tree. Do not claim network isolation when the host backend cannot enforce it; record `network_isolation=false` in capabilities.

```python
class LocalProcessBackend:
    def start(self, request: WorkerRequest) -> ProcessHandle:
        work_dir, request_path, result_path = self._prepare_request(request)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "sohbench.fabric.worker_main",
             "--request", str(request_path), "--result", str(result_path)],
            cwd=work_dir,
            env=minimal_worker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
        limiter = attach_process_limits(process, request.limits)
        return ProcessHandle(process, result_path, limiter, request.limits["wall_seconds"])
```

- [ ] **Step 4: Run backend tests and check no child remains**

Run: `python -m pytest tests/fabric/test_local_process_backend.py -q`

Expected: all tests PASS and every handle reports `is_alive() is False` after completion.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/execution.py src/sohbench/fabric/process_limits.py tests/fabric/test_local_process_backend.py tests/fixtures/workers
git commit -m "feat: isolate evaluation in local workers"
```

### Task 6: Move Training And Adaptation Execution Behind Workers

**Files:**
- Create: `src/sohbench/fabric/workers/training.py`
- Create: `src/sohbench/fabric/workers/adaptation.py`
- Create: `src/sohbench/fabric/workers/metric.py`
- Create: `src/sohbench/fabric/worker_io.py`
- Test: `tests/fabric/test_training_worker.py`
- Test: `tests/fabric/test_adaptation_worker.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `tests/test_evaluation_service.py`

**Interfaces:**
- Training Worker consumes: train feature view, train labels, target-free test view, AlgorithmPackage ref
- Training Worker produces: prediction ref, fold model refs, resource trace ref
- Adaptation Worker consumes: immutable base ModelArtifact ref, declared calibration feature/label view and target-free held-out view
- Adaptation Worker produces: derived ModelArtifact ref, adaptation cost/trace and held-out prediction ref; it never overwrites the base model
- Metric Worker consumes: prediction ref, truth ref, MetricSpec ref
- Metric Worker produces: metric records and trajectory records

- [ ] **Step 1: Write failing cross-process and truth-boundary tests**

Create a fixture estimator that records its process ID and received columns. Assert the Training Worker PID differs from the test/API PID, `soh` is absent from every estimator input, and Metric Worker receives predictions plus truth only after training exits.

```python
assert result.worker_pid != os.getpid()
assert result.debug_columns["test"] == ["dataset", "cell_id", "cycle_index", "f1"]
assert metric_result.metrics["mae"] == pytest.approx(0.05)
```

Create an adaptation fixture that records calibration and held-out columns. Assert calibration labels are available only to the adapter fit step, held-out labels never enter the Adaptation Worker, the base model digest is unchanged and the derived model has a new lineage/digest.

- [ ] **Step 2: Run worker and service tests before integration**

Run: `python -m pytest tests/fabric/test_training_worker.py tests/fabric/test_adaptation_worker.py tests/test_evaluation_service.py -q`

Expected: new training/adaptation worker tests FAIL; existing service tests remain green.

- [ ] **Step 3: Implement CAS-based frame exchange and worker adapters**

Use Parquet for tabular exchanges and JSON for manifests. `worker_io` verifies every digest before reading. Training returns prediction rows keyed by `dataset`, `cell_id`, `cycle_index`, `fold`, `seed`, `candidate_id`; Metric Worker performs the truth join internally and rejects duplicate or missing keys. Adaptation receives separate calibration/held-out references, clones or derives from the immutable base model, records calibration asset/cycle counts and returns a new ModelArtifact lineage. Add an `execution_backend` dependency to `EvaluationService`, defaulting to the inline compatibility backend only while `SOHBENCH_FABRIC_WORKERS` is false during migration.

```python
def run_training(request: WorkerRequest, io: WorkerIO) -> WorkerResult:
    train_x = io.read_frame(request.ref("train_features"))
    train_y = io.read_series(request.ref("train_labels"))
    test_x = io.read_frame(request.ref("test_features"))
    reject_target_columns(train_x, test_x)
    candidate = load_algorithm_package(request.ref("algorithm"))
    model = candidate.fit(train_x, train_y, seed=request.seed)
    predictions = prediction_records(request, test_x, model.predict(test_x))
    return io.result(predictions=io.write_frame(predictions), model=io.write_model(model))


def run_adaptation(request: WorkerRequest, io: WorkerIO) -> WorkerResult:
    base = io.load_verified_model(request.ref("base_model"))
    calibration_x = io.read_frame(request.ref("calibration_features"))
    calibration_y = io.read_series(request.ref("calibration_labels"))
    held_out_x = io.read_frame(request.ref("held_out_features"))
    reject_target_columns(calibration_x, held_out_x)
    derived = clone_model(base).adapt(calibration_x, calibration_y)
    predictions = prediction_records(request, held_out_x, derived.predict(held_out_x))
    return io.result(
        predictions=io.write_frame(predictions),
        model=io.write_derived_model(derived, base_digest=request.ref("base_model").digest),
    )
```

- [ ] **Step 4: Enable the local backend in focused service tests**

Run: `$env:SOHBENCH_FABRIC_WORKERS='1'; python -m pytest tests/fabric/test_training_worker.py tests/fabric/test_adaptation_worker.py tests/test_evaluation_service.py -q`

Expected: all tests PASS; the API process never imports either fixture and adaptation cannot read held-out truth.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/workers/training.py src/sohbench/fabric/workers/adaptation.py src/sohbench/fabric/workers/metric.py src/sohbench/fabric/worker_io.py src/sohbench/evaluation_service.py tests/fabric/test_training_worker.py tests/fabric/test_adaptation_worker.py tests/test_evaluation_service.py
git commit -m "feat: execute training and adaptation in workers"
```

### Task 7: Register Plugin Packages And Pickle Models Without API Execution

**Files:**
- Create: `src/sohbench/fabric/asset_registry.py`
- Create: `src/sohbench/fabric/model_assets.py`
- Create: `src/sohbench/fabric/schemas/package-manifest.schema.json`
- Create: `src/sohbench/fabric/schemas/model-manifest.schema.json`
- Create: `src/sohbench/fabric/schemas/model-artifact.schema.json`
- Create: `src/sohbench/fabric/workers/inference.py`
- Test: `tests/fabric/test_asset_registry.py`
- Test: `tests/fabric/test_model_assets.py`
- Test: `tests/fabric/test_inference_worker.py`
- Modify: `src/sohbench/model_store.py`
- Modify: `src/sohbench/trusted_plugins.py`
- Modify: `tests/test_model_store.py`

**Interfaces:**
- Produces: `AssetRegistry.register(kind, fileobj, manifest) -> RegisteredAsset`
- Accepted package kinds: `dataset_adapter`, `algorithm`, `feature`
- Produces: `RegisteredAsset(asset_id, kind, payload_ref, manifest_ref, trust_tier, status)`
- Produces: `ModelAssetRegistry.register(fileobj, manifest) -> ModelAsset`
- Produces: `ModelAsset(asset_id, pickle_ref, manifest_ref, trust_tier)`
- ModelArtifact records: immutable model/pickle refs, source run/base model, algorithm/feature refs, runtime, schema, metrics scope and digest
- Frozen Inference Worker consumes: ModelAsset refs and target-free DatasetSnapshot view
- Frozen Inference Worker produces: predictions, cold/warm latency, throughput, peak memory, abstentions

- [ ] **Step 1: Write import-side-effect and malicious-reduce registration tests**

```python
def test_source_package_registration_does_not_import_payload(tmp_path):
    marker = tmp_path / "imported.txt"
    payload = source_package_that_writes_on_import(marker)
    registry = asset_registry(tmp_path)
    asset = registry.register("algorithm", io.BytesIO(payload), valid_package_manifest(payload))
    assert asset.status == "registered_unvalidated"
    assert marker.exists() is False


def test_registration_does_not_execute_pickle_reduce(tmp_path):
    marker = tmp_path / "executed.txt"
    payload = build_pickle_that_writes(marker)
    registry = model_registry(tmp_path)
    asset = registry.register(io.BytesIO(payload), valid_manifest(payload))
    assert asset.pickle_ref.digest == hashlib.sha256(payload).hexdigest()
    assert marker.exists() is False
```

The package suite must require entry point, runtime, dependency lock, trust tier and payload digest without importing the entry point. The model suite must require `training_run_id`, `algorithm_id`, `feature_schema`, `runtime`, `python_version`, `dependencies`, and `pickle_sha256`. The worker suite must prove deserialization occurs in another PID and a digest mismatch is rejected before load.

- [ ] **Step 2: Confirm the current model store is unsafe under the test**

Run: `python -m pytest tests/fabric/test_asset_registry.py tests/fabric/test_model_assets.py tests/fabric/test_inference_worker.py tests/test_model_store.py -q`

Expected: source-package registration/import boundary is missing and the malicious-reduce test FAILS against the current API-side `pickle.loads` behavior.

- [ ] **Step 3: Store opaque bytes in CAS and deserialize only in the inference worker**

Validate manifest JSON and stream SHA-256 while registering every package. The API stores source archives as opaque CAS bytes and does not import entry points; validation that requires code runs in the corresponding Worker. The API may inspect a model manifest but must not inspect Python object attributes. Inference Worker verifies both refs, checks declared feature names and runtime policy, then loads the Pickle. Preserve v1 asset/model listing through adapters that read manifests. Mark old unmanifested packages and Pickles `legacy_unverified` and block them from formal runs until a manifest is attached.

```python
class AssetRegistry:
    def register(self, kind: str, fileobj, manifest: dict) -> RegisteredAsset:
        validate_package_manifest(kind, manifest)
        payload_ref = self.cas.put_stream(fileobj, "application/zip", manifest["filename"])
        if payload_ref.digest != manifest["payload_sha256"]:
            raise ArtifactIntegrityError("package digest mismatch")
        manifest_ref = self.cas.put_bytes(canonical_json(manifest), "application/json", "manifest.json")
        return self.store.insert_asset(kind, payload_ref, manifest_ref, manifest["trust_tier"])


def run_inference(request: WorkerRequest, io: WorkerIO) -> WorkerResult:
    manifest = io.read_verified_json(request.ref("model_manifest"))
    model_path = io.materialize_verified(request.ref("model_pickle"))
    features = io.read_frame(request.ref("features"))
    reject_target_columns(features)
    validate_feature_schema(features, manifest["feature_schema"])
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    return benchmark_predictions(request, model, features, io)
```

- [ ] **Step 4: Run model, inference and API regressions**

Run: `python -m pytest tests/fabric/test_asset_registry.py tests/fabric/test_model_assets.py tests/fabric/test_inference_worker.py tests/test_model_store.py tests/test_api_server.py -q`

Expected: all tests PASS; source-import and Pickle marker files remain absent in the API process.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/asset_registry.py src/sohbench/fabric/model_assets.py src/sohbench/fabric/schemas/package-manifest.schema.json src/sohbench/fabric/schemas/model-manifest.schema.json src/sohbench/fabric/schemas/model-artifact.schema.json src/sohbench/fabric/workers/inference.py src/sohbench/model_store.py src/sohbench/trusted_plugins.py tests/fabric/test_asset_registry.py tests/fabric/test_model_assets.py tests/fabric/test_inference_worker.py tests/test_model_store.py
git commit -m "fix: keep uploaded assets opaque in the api process"
```

### Task 8: Add The Recoverable Scheduler

**Files:**
- Create: `src/sohbench/fabric/scheduler.py`
- Test: `tests/fabric/test_scheduler.py`
- Modify: `src/sohbench/fabric/db.py`
- Modify: `src/sohbench/evaluation_service.py`

**Interfaces:**
- Produces: `LocalScheduler.submit(run_id) -> None`
- Produces: `LocalScheduler.cancel(run_id, actor, reason) -> dict`
- Produces: `LocalScheduler.retry(run_id, actor, reason) -> dict`
- Produces: `LocalScheduler.recover() -> RecoveryReport`
- Configuration: max workers, per-run timeout, stale heartbeat threshold

- [ ] **Step 1: Write failing queue, retry and restart-recovery tests**

```python
def test_retry_creates_attempt_two_and_preserves_attempt_one(tmp_path):
    scheduler = scheduler_with_failing_once_backend(tmp_path)
    run_id = scheduler.create_and_submit(valid_spec())
    scheduler.wait_idle()
    scheduler.retry(run_id, "reviewer", "infrastructure retry")
    scheduler.wait_idle()
    events = scheduler.store.list_events(run_id)
    assert {event["attempt"] for event in events} == {1, 2}
    assert any(event["type"] == "FAILED" and event["attempt"] == 1 for event in events)
    assert scheduler.store.get_run(run_id)["execution_state"] == "REVIEW"
```

The same test module must cover FIFO with `max_workers=1`, cancellation of queued and running runs, and recovery that marks a stale `RUNNING` attempt failed before queuing a new attempt only when policy allows.

- [ ] **Step 2: Run scheduler tests and verify the missing implementation**

Run: `python -m pytest tests/fabric/test_scheduler.py -q`

Expected: FAIL because `LocalScheduler` is absent.

- [ ] **Step 3: Implement bounded dispatch and heartbeat recovery**

Use one coordinator thread and a bounded set of `ProcessHandle`s. Never keep canonical state only in memory. Persist `QUEUED` before dispatch, `RUNNING` with PID and heartbeat, then terminal/aggregation transitions. Retry only `INFRASTRUCTURE` and configured `RESOURCE` failures; algorithm and policy failures require a new RunSpec. Cancellation is idempotent and produces a reasoned event.

```python
class LocalScheduler:
    def submit(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        self.store.append_event(run_id, "QUEUED", {}, "scheduler", run["attempt"], run["version"])
        self.queue.put(run_id)

    def _dispatch_one(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        request = self.requests.build(run)
        handle = self.backend.start(request)
        self.store.append_event(
            run_id, "RUNNING", {"pid": handle.pid, "heartbeat": utc_now()},
            "scheduler", run["attempt"], run["version"],
        )
        result = handle.wait(timeout=request.limits["wall_seconds"] + 2)
        self._record_result(run_id, run["attempt"], result)

    def retry(self, run_id: str, actor: str, reason: str) -> dict:
        previous = self.store.latest_attempt(run_id)
        if previous.failure_category not in self.retryable_categories:
            raise RetryNotAllowed(previous.failure_category)
        attempt = self.store.create_attempt(run_id, previous.attempt + 1, actor, reason)
        self.queue.put(run_id)
        return attempt
```

- [ ] **Step 4: Run scheduler and worker tests together**

Run: `python -m pytest tests/fabric/test_scheduler.py tests/fabric/test_local_process_backend.py tests/fabric/test_training_worker.py -q`

Expected: all tests PASS without leaked worker processes.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/scheduler.py src/sohbench/fabric/db.py src/sohbench/evaluation_service.py tests/fabric/test_scheduler.py
git commit -m "feat: add recoverable local run scheduler"
```

### Task 9: Expose API v2 And Preserve v1 Adapters

**Files:**
- Create: `src/sohbench/api_v2.py`
- Create: `src/sohbench/fabric/errors.py`
- Create: `src/sohbench/fabric/signing.py`
- Create: `src/sohbench/fabric/schemas/evidence-attestation.schema.json`
- Test: `tests/fabric/test_api_v2.py`
- Test: `tests/fabric/test_evidence_signing.py`
- Modify: `src/sohbench/api_server.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `tests/test_api_server.py`

**Interfaces:**
- Produces: `EvidenceSigner` protocol
- Produces: `LocalDigestAttestationSigner.sign(evidence_ref, actor, verdict, reason) -> AttestationRef`
- Adds: `GET /api/v2/registry/{kind}`
- Adds: `POST /api/v2/assets/{datasets|algorithms|features|models}`
- Adds: `POST /api/v2/preflight`
- Adds: `POST /api/v2/experiments`
- Adds: `POST /api/v2/runs`
- Adds: `GET /api/v2/runs/{run_id}`
- Adds: `GET /api/v2/runs/{run_id}/events?after=<sequence>`
- Adds: `POST /api/v2/runs/{run_id}/cancel`
- Adds: `POST /api/v2/runs/{run_id}/retry`
- Adds: `GET /api/v2/comparisons`
- Adds: `GET /api/v2/evidence/{evidence_id}`
- Adds: `POST /api/v2/evidence/{evidence_id}/review`
- Adds: `POST /api/v2/evidence/{evidence_id}/sign`
- Error envelope: `code`, `category`, `message`, `impact`, `correlation_id`, `details`

- [ ] **Step 1: Write failing API contract tests**

```python
def test_create_run_is_idempotent(api_client):
    headers = {"Idempotency-Key": "browser-action-1"}
    first = api_client.post("/api/v2/runs", json=valid_run_spec(), headers=headers)
    second = api_client.post("/api/v2/runs", json=valid_run_spec(), headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]


def test_event_resume_uses_sequence_cursor(api_client, completed_run):
    events = api_client.get(f"/api/v2/runs/{completed_run}/events?after=2").json()["items"]
    assert all(item["sequence"] > 2 for item in events)


def test_local_signing_freezes_verified_evidence(api_client, reviewed_evidence):
    response = api_client.post(
        f"/api/v2/evidence/{reviewed_evidence}/sign",
        json={"actor": "local-reviewer", "verdict": "warn", "reason": "accepted limitation"},
        headers={"Idempotency-Key": "sign-once"},
    )
    assert response.status_code == 200
    assert response.json()["execution_state"] == "SIGNED"
    assert response.json()["attestation"]["assurance"] == "local_digest_attestation"
```

The same API contract suite must cover registry listing, opaque package/model upload, experiment persistence, comparison filtering, correlation IDs, invalid transition error envelopes, `Content-Type: application/json; charset=utf-8`, and v1 training endpoints mapping to the same run ID. The signing suite must reject unverified/tampered evidence, require actor/verdict/reason, make signing idempotent and reject any in-place artifact mutation after `SIGNED`.

- [ ] **Step 2: Run API tests and observe 404 responses**

Run: `python -m pytest tests/fabric/test_api_v2.py tests/fabric/test_evidence_signing.py tests/test_api_server.py -q`

Expected: v2/signing tests FAIL because routes and signer are missing; v1 tests remain green.

- [ ] **Step 3: Implement thin route handlers over the scheduler and registries**

Route handlers validate JSON before service calls and never access plugin files directly. Asset writes stream opaque payloads into the registries from Task 7. `preflight` checks refs, trust tier, feature schema, resource estimate and comparison key without executing code. `EvidenceSigner` is a provider interface; Phase 1 implements only `LocalDigestAttestationSigner`, which verifies the EvidencePack, records actor/verdict/reason and the frozen digest, then transitions `REVIEW -> SIGNED`. Its attestation declares `assurance="local_digest_attestation"` and must not be described as cryptographic organizational identity. Return `202` for accepted runs and retries, `200` for reads/cancel/sign already completed, `409` for invalid state, and `422` for contract errors. Keep v1 response shapes through explicit serializers.

```python
class EvidenceSigner(Protocol):
    def sign(self, evidence_ref: ArtifactRef, actor: str, verdict: str, reason: str) -> AttestationRef: ...


class LocalDigestAttestationSigner:
    def sign(self, evidence_ref, actor, verdict, reason):
        verified = self.evidence.verify(evidence_ref)
        document = {
            "schema_version": "1.0.0",
            "assurance": "local_digest_attestation",
            "evidence_sha256": verified.digest,
            "actor": require_text(actor),
            "verdict": validate_verdict(verdict),
            "reason": require_text(reason),
            "signed_at": utc_now(),
        }
        ref = self.cas.put_bytes(canonical_json(document), "application/json", "attestation.json")
        self.store.mark_signed(verified.run_id, ref, expected_state="REVIEW")
        return AttestationRef.from_artifact(ref, assurance=document["assurance"])


def create_run_endpoint(request, services):
    spec = validate_run_spec_document(request.json)
    run = services.runs.create(spec, idempotency_key=request.headers["Idempotency-Key"])
    services.scheduler.submit(run["run_id"])
    return json_response(run, status=202)
```

- [ ] **Step 4: Run API, service and contract regressions**

Run: `python -m pytest tests/fabric/test_api_v2.py tests/fabric/test_evidence_signing.py tests/test_api_server.py tests/test_evaluation_service.py -q`

Expected: all tests PASS; v1/v2 resolve to identical evidence IDs for the same run and signed evidence is immutable.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/api_v2.py src/sohbench/fabric/errors.py src/sohbench/fabric/signing.py src/sohbench/fabric/schemas/evidence-attestation.schema.json src/sohbench/api_server.py src/sohbench/evaluation_service.py tests/fabric/test_api_v2.py tests/fabric/test_evidence_signing.py tests/test_api_server.py
git commit -m "feat: expose verification fabric api v2"
```

### Task 10: Close The Phase 1 Security And Recovery Gate

**Files:**
- Create: `tests/fabric/test_phase1_security_gate.py`
- Create: `tests/fabric/test_phase1_recovery_gate.py`
- Create: `tests/fixtures/plugins/hang_forever.py`
- Create: `tests/fixtures/plugins/import_side_effect.py`
- Create: `tests/fixtures/plugins/path_escape.py`
- Create: `docs/verification/phase1-control-plane-report.md`
- Modify: `README.md`

**Interfaces:**
- Verifies: API process does not execute plugin import/Pickle payloads
- Verifies: timeout, cancel, restart recovery, attempt history and CAS integrity
- Verifies: T3 submission returns `POLICY/T3_BACKEND_UNAVAILABLE`

- [ ] **Step 1: Add the full adversarial acceptance tests**

Test one hanging source plugin, one import side effect, one path traversal output, one malicious Pickle, two same-second concurrent runs, API restart during `RUNNING`, and T3 submission. Start the API without feature flags and assert it selects `LocalProcessBackend`; the inline compatibility backend may run only with `SOHBENCH_LEGACY_INLINE=1` and must reject T2 assets. Assert the server still answers registry and run-status requests while the hanging worker is being terminated.

```python
def test_default_api_backend_contains_adversarial_plugins(fabric_server, tmp_path):
    assert fabric_server.capabilities()["backend"] == "local_process"
    run = fabric_server.submit_algorithm_fixture("hang_forever.py", timeout_seconds=1)
    assert fabric_server.get("/api/v2/registry/algorithms").status_code == 200
    completed = fabric_server.wait(run["run_id"])
    assert completed["execution_state"] == "TIMED_OUT"
    assert completed["outcome"] == "timed_out"
    assert not (tmp_path / "imported.txt").exists()


def test_t3_is_rejected_without_container_backend(fabric_server):
    response = fabric_server.post("/api/v2/assets/algorithms", t3_package_request())
    assert response.status_code == 422
    assert response.json()["category"] == "POLICY"
    assert response.json()["code"] == "T3_BACKEND_UNAVAILABLE"
```

- [ ] **Step 2: Run the gate and capture any remaining failures**

Run: `python -m pytest tests/fabric/test_phase1_security_gate.py tests/fabric/test_phase1_recovery_gate.py -q`

Expected before the safe-default change: FAIL only if the API still defaults to inline execution. No other security/recovery assertion may fail.

- [ ] **Step 3: Enable safe defaults and run the complete Python regression suite**

Make `LocalProcessBackend` the default for API-started runs. Keep inline execution behind `SOHBENCH_LEGACY_INLINE=1`, label it `legacy_unsafe`, and reject uploaded T2 assets in that mode. Re-run the gate, then the full suite.

```python
def configured_execution_backend(settings, work_root):
    if settings.legacy_inline:
        return InlineCompatibilityBackend(
            capability_label="legacy_unsafe",
            allowed_trust_tiers=frozenset({"T1"}),
        )
    return LocalProcessBackend(work_root=work_root)
```

Run: `python -m pytest tests/fabric/test_phase1_security_gate.py tests/fabric/test_phase1_recovery_gate.py -q`

Run: `python -m pytest -q`

Expected: both commands PASS; no marker/escaped artifact is created and no worker process remains after pytest exits.

- [ ] **Step 4: Write the Phase 1 verification report**

Record the exact commands, test counts, OS capability flags, timeout/cancel observations, API/Pickle boundary evidence, remaining T2 limitations, and the explicit T3 rejection. Update README with `SOHBENCH_FABRIC_WORKERS`, database/CAS paths and recovery behavior.

- [ ] **Step 5: Commit**

```powershell
git add tests/fabric/test_phase1_security_gate.py tests/fabric/test_phase1_recovery_gate.py tests/fixtures/plugins docs/verification/phase1-control-plane-report.md README.md
git commit -m "test: close local control plane acceptance gate"
```

## Phase 1 Review Gate

- [ ] API process source inspection and adversarial tests prove it does not import uploaded code or deserialize Pickle.
- [ ] Event replay yields the same `execution_state`, `outcome`, `verdict`, attempt and artifact refs as the stored snapshot.
- [ ] Hanging workers time out and full process trees terminate without blocking API reads.
- [ ] Retry preserves every prior attempt and only follows the declared retry policy.
- [ ] CAS rejects tampering and path traversal; same-second concurrent runs cannot overwrite each other.
- [ ] v1 behavior remains available through adapters and v2 errors use stable categories/correlation IDs.
- [ ] `EvidenceSigner` freezes verified evidence with a local digest attestation; the UI/API does not mislabel it as an organizational cryptographic signature.
- [ ] T2 capabilities are accurately displayed; T3 is rejected until a container backend exists.

## Plan Verification Commands

```powershell
python -m pytest tests/fabric/test_event_store.py tests/fabric/test_run_state.py tests/fabric/test_artifact_store.py -q
python -m pytest tests/fabric/test_worker_protocol.py tests/fabric/test_local_process_backend.py tests/fabric/test_training_worker.py tests/fabric/test_adaptation_worker.py tests/fabric/test_inference_worker.py -q
python -m pytest tests/fabric/test_scheduler.py tests/fabric/test_api_v2.py tests/fabric/test_evidence_signing.py -q
python -m pytest tests/fabric/test_phase1_security_gate.py tests/fabric/test_phase1_recovery_gate.py -q
python -m pytest -q
```

Expected: every command exits `0`; no marker file or escaped artifact exists; no child worker remains running.
