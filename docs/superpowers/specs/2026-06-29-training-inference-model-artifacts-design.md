# Training and Inference Evaluation Split Design

## Goal

Split the current local SOH evaluation workbench into two explicit modes:

- Training evaluation: run `fit + predict`, compare algorithms and feature sets,
  and persist trained model artifacts.
- Inference evaluation: load existing persisted models, run `predict` only on
  selected datasets, and compare transfer generalization plus inference speed.

This makes the benchmark answers clearer. Training evaluation answers which
algorithm trains best under a protocol. Inference evaluation answers whether an
existing trained model transfers to new data and whether it is fast enough for
deployment-like use.

## Current Context

The existing trusted local system supports:

- dataset upload as prepared SOH tables
- estimator upload as trusted `.py` source plugins
- feature-set upload as trusted `.py` source plugins
- background benchmark jobs through `EvaluationService`
- frontend JSON artifacts for leaderboard, dataset leaderboard, QC, and run config

The runner currently treats every estimator as trainable and executes `fit`
before `predict` in every fold. Algorithm upload also currently means Python
source upload, not persisted model upload.

## Recommended Product Structure

The frontend should expose three related areas.

### Training Evaluation

Training evaluation keeps the current benchmark semantics:

- inputs: datasets, source-code estimator plugins, feature-set plugins, protocols,
  and seeds
- execution: `fit + predict`
- outputs: accuracy, calibration, physical consistency, training runtime, and
  persisted model artifacts

Each trained model artifact should be saved after a successful fit. The minimum
model package is:

```text
model.pkl
model_manifest.json
```

The manifest records:

```json
{
  "model_name": "ridge_full_lodo_seed0_nasa",
  "model_kind": "pickle",
  "estimator": "ridge",
  "featureset": "full",
  "feature_columns": ["voltage_mean", "resistance_proxy"],
  "target": "soh",
  "trained_on": ["calce_cs2", "hust", "nasa"],
  "held_out_key": "oxford",
  "protocol": "lodo",
  "seed": 0,
  "created_by_run": "20260629T151929",
  "predict_contract": "predict(X, meta=None)"
}
```

Persisted models created by training jobs become entries in the model library.

### Inference Evaluation

Inference evaluation is a separate page/tab. It never calls `fit`.

It supports two model sources:

- models produced by training evaluation
- externally uploaded trusted pickle models

Inputs:

- one or more persisted models
- one or more datasets
- optional feature-set selection when the manifest does not fully define the
  feature columns

Execution:

- load model package
- validate feature compatibility
- prepare the selected feature table
- run `predict` only
- measure pure prediction time separately from model load time
- compute transfer metrics when `soh` labels are available

Outputs:

- MAE, RMSE, MBE, and other point metrics
- optional interval metrics only if the model exposes `predict_interval`
- total prediction milliseconds
- milliseconds per 1000 samples
- samples per second
- load milliseconds
- dataset coverage and feature compatibility status

### Model Library

The model library is a shared inventory used by both modes. It lists:

- model name
- model source: training artifact or external upload
- path to model package
- estimator name
- featureset
- feature columns
- training datasets
- creation run
- best known validation metrics
- compatibility status for selected inference datasets

## Backend Architecture

Add a dedicated model artifact layer rather than overloading trusted estimator
plugins.

Recommended module:

```text
src/sohbench/model_store.py
```

Responsibilities:

- save trained estimators as model packages
- load trusted pickle model packages
- validate manifests
- register model entries in catalog storage
- expose model inventory to `EvaluationService`

The trusted plugin store remains responsible for source-code estimator and
feature-set plugins. It should not become responsible for trained model packages.

## API Shape

Keep the existing source-code upload endpoint for training algorithms:

```text
POST /api/upload/algorithm
```

Add model-specific endpoints:

```text
POST /api/upload/model
GET  /api/models
POST /api/inference-jobs
GET  /api/inference-jobs
GET  /api/inference-jobs/{id}
```

The model upload endpoint accepts a trusted local pickle package. The first
implementation can accept JSON with:

- `name`
- `modelFilename`
- `modelContentBase64`
- `manifestFilename`
- `manifestContentBase64`
- `replace`

Later, this can be expanded to a zip package if the UI needs one-file upload.

## Model Loading Contract

Supported pickle model objects should expose one of:

```python
predict(X)
```

```python
predict(X, meta=None)
```

Optional:

```python
predict_interval(X, meta=None)
```

The loader should adapt either predict signature. It should reject models that:

- do not expose `predict`
- cannot run a smoke prediction
- return a wrong number of predictions
- return non-finite predictions
- have no manifest
- request feature columns that are absent from the selected data

## Security Boundary

Pickle loading is trusted local mode. A pickle file can execute Python code at
load time, so the UI and docs must describe pickle uploads as trusted local
research artifacts only.

This design does not claim sandboxing. A future hardened mode should load
pickle models in a separate process or container with explicit filesystem,
network, CPU, memory, and timeout limits.

## Artifact Additions

Training jobs should add:

```text
models/
  <model_name>/
    model.pkl
    model_manifest.json
model_catalog.json
```

Inference jobs should add:

```text
inference_results.parquet
inference_results.csv
inference_benchmark.json
inference_run_config.json
```

`inference_benchmark.json` should be synced to the frontend data directory.

## Frontend Behavior

Rename the current workbench area conceptually to training evaluation. Keep its
algorithm upload behavior as source-code upload.

Add a new inference tab with:

- model selector with uploaded and training-produced models
- dataset checklist
- compatibility preview
- start inference evaluation button
- result cards for best transfer accuracy, fastest model, and highest throughput
- table comparing model, dataset, featureset, MAE, RMSE, prediction milliseconds,
  milliseconds per 1000 samples, and samples per second

Old runs without inference artifacts should show a clear empty state.

## Data Flow

Training evaluation:

1. User selects datasets, estimator source plugins, feature sets, protocols, and
   seeds.
2. Backend runs the benchmark with `fit + predict`.
3. Backend writes normal benchmark artifacts.
4. Backend saves fitted model packages with manifests.
5. Backend updates the model catalog.
6. Frontend refreshes leaderboard and model library.

Inference evaluation:

1. User selects persisted models and datasets.
2. Backend validates model manifests against selected data.
3. Backend loads each model.
4. Backend times `predict` only, while separately recording load time.
5. Backend computes transfer metrics when labels are present.
6. Backend writes inference artifacts and syncs frontend data.
7. Frontend renders the inference tab results.

## Error Handling

Training model persistence failures should not hide the main benchmark result.
They should be recorded as warnings in the job status and model catalog.

Inference jobs should fail fast per model/dataset pair for:

- missing model package
- invalid manifest
- unsupported predict contract
- missing features
- non-finite predictions

The job should continue across independent model/dataset pairs when possible and
record failed pairs in the inference artifact.

## Testing Strategy

Backend tests:

- saving a fitted sklearn-like estimator writes `model.pkl` and manifest
- model catalog includes training-produced and uploaded models
- pickle model upload validates manifest and smoke prediction
- inference job never calls `fit`
- inference job measures predict time and writes `inference_benchmark.json`
- feature mismatch returns a useful validation error

Frontend tests:

- model catalog rows normalize correctly
- inference payload includes selected models and datasets
- empty inference artifacts render a no-data state
- inference rows sort by milliseconds per 1000 samples
- speed and transfer metric cards render from `inference_benchmark.json`

Integration test:

- train a small model on a synthetic dataset
- persist the model
- run inference-only evaluation on a different synthetic dataset
- verify metrics, timing fields, and frontend artifacts

## Implementation Boundary

This is one coherent feature, but it should be implemented in slices:

1. backend model package save/load/catalog
2. inference-only runner and artifacts
3. API endpoints
4. frontend inference tab and model library
5. end-to-end validation using a small persisted model

The deep long-running matrix can remain accuracy-focused until this feature is
stable. It does not need to be restarted just to add pickle inference support.
