# Trusted Local Plugin Evaluation System Design

## Goal

Turn the static sohbench dashboard into a local front/back-end evaluation system where uploading any one of these resource types can start a fresh benchmark run:

- a prepared dataset table
- a trusted local Python estimator plugin
- a trusted local Python feature-set plugin

The first implementation is intentionally a trusted local mode. Uploaded Python code runs in the backend Python process with local filesystem permissions. This is suitable for lab-owned algorithms and feature scripts, not untrusted third-party code.

## Scope

In scope:

- Dataset upload for `.pkl`, `.parquet`, and `.csv` tables that already contain sohbench rows.
- Dataset validation requiring `dataset`, `cell_id`, `cycle_index`, and `soh`.
- Estimator plugin upload as a `.py` file exposing either `Estimator`, `get_estimator()`, or `PLUGIN["estimator"]`.
- Feature-set plugin upload as a `.py` file exposing a feature-set definition and optionally a `transform(table, meta=None)` function.
- API endpoints for catalog, upload, job creation, job status, and health.
- Background job execution that reuses existing sohbench evaluation, report writing, and frontend artifact sync.
- Chinese frontend controls for resource upload, dataset/estimator/featureset selection, job launch, and job status.
- Validation using one new public dataset, one new estimator plugin, and one new feature-set plugin.

Out of scope for this version:

- Sandboxing, containers, remote execution, multi-user authentication, or permission isolation.
- Raw cycler-file parsers for every public dataset format.
- Arbitrary dependency installation from uploaded plugins.
- Distributed job queues.

## Architecture

The backend is a small stdlib HTTP JSON service. This avoids adding FastAPI/uvicorn dependencies to the current project and keeps startup simple on the existing Windows machine.

The service has three layers:

- `trusted_plugins.py` owns resource validation, trusted plugin loading, catalog persistence, dataset normalization, and feature transforms.
- `evaluation_service.py` owns job creation, background execution, benchmark orchestration, and frontend artifact publishing.
- `api_server.py` exposes JSON endpoints and CORS for the static dashboard.

The existing benchmark core remains the source of truth:

- datasets are discovered through `DatasetLoader`
- estimators and featuresets use the existing registry
- benchmark output still flows through `write_artifacts`
- frontend JSON sync still uses `build_frontend_artifacts`

## Plugin Contracts

Estimator plugins must expose one of:

```python
class Estimator:
    name = "my_estimator"
    feature_mode = "unified"
    backend = "sklearn"
    def fit(self, X, y, meta=None): ...
    def predict(self, X, meta=None): ...
```

```python
def get_estimator():
    return MyEstimator
```

```python
PLUGIN = {"type": "estimator", "estimator": MyEstimator}
```

Feature-set plugins must expose a name and selected columns:

```python
FEATURE_COLUMNS = ["cycle_norm", "capacity_fade_rate"]
FEATURESET_NAME = "trend_features"

def transform(table, meta=None):
    out = table.copy()
    out["cycle_norm"] = out.groupby(["dataset", "cell_id"])["cycle_index"].transform(lambda s: s / max(float(s.max()), 1.0))
    out["capacity_fade_rate"] = 1.0 - out["soh"]
    return out
```

The loader smoke-tests uploaded plugins on a small synthetic table before registering them.

## Data Flow

1. The browser sends upload requests as JSON with `name`, `filename`, and base64 file content.
2. The backend validates and stores datasets under the configured `data_dir/processed`.
3. The backend stores plugin files under `work/trusted_plugins/{estimators,featuresets}` and records a catalog JSON.
4. The browser asks `/api/catalog` for datasets, estimators, featuresets, and previous jobs.
5. The browser posts `/api/jobs` with selected datasets, estimators, featuresets, protocols, and seeds.
6. A background thread loads datasets, applies selected feature transforms, runs the benchmark, writes artifacts, and syncs `frontend/app/data`.
7. The browser polls `/api/jobs/{id}` and reloads dashboard JSON after success.

## Error Handling

- Upload endpoints return `400` for invalid names, unsupported file types, missing columns, plugin contract failures, and failed smoke tests.
- Job failures are captured into the job record with a short error message.
- The frontend keeps the current dashboard data visible while showing upload or job errors in the workbench panel.
- Duplicate resource names are rejected unless `replace: true` is sent.

## Testing

Python tests cover:

- dataset upload validation and normalized `.pkl` persistence
- estimator plugin loading, smoke testing, and registry registration
- feature-set plugin loading, transform application, and registry registration
- inline job execution against a synthetic dataset
- API catalog/upload/job endpoints through raw HTTP requests

Frontend tests cover:

- catalog option normalization for datasets, estimators, and featuresets
- job payload construction from checked resource rows
- existing dataset-scoped dashboard behavior remains green

End-to-end validation covers:

- downloading a public dataset not already in the current six-dataset artifact set
- converting it into sohbench rows
- uploading/registering a new estimator plugin and feature-set plugin
- triggering a benchmark through the backend
- confirming the dashboard data artifacts include the new dataset, estimator, and featureset

## Security Note

Trusted local plugin mode executes uploaded Python code in-process. The UI must present this as a local research mode. A future hardened mode should run plugin code in a separate process or container with explicit mount, network, CPU, and memory limits.
