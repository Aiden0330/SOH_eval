from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from sohbench.evaluation_service import EvaluationService


RWTH_REPO = "https://github.com/isea-rwth-aachen/battery-degradation-trajectory-prediction.git"
RWTH_MAT = Path("Pre-processing Codes") / "Degradation_Prediction_Dataset_ISEA.mat"


ESTIMATOR_PLUGIN = '''
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sohbench.contracts import feature_columns


class Estimator:
    name = "trusted_huber_decay"
    feature_mode = "unified"
    backend = "sklearn"

    def __init__(self):
        self._cols = None
        self._model = Pipeline([
            ("scale", StandardScaler()),
            ("huber", HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=500)),
        ])

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        self._model.fit(X[self._cols].to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X, meta=None):
        pred = self._model.predict(X[self._cols].to_numpy(dtype=float))
        return np.clip(pred, 0.0, 1.3)
'''


FEATURE_PLUGIN = '''
import numpy as np

FEATURESET_NAME = "rwth_cycle_shape"
FEATURE_COLUMNS = [
    "rwth_cycle_norm",
    "rwth_sqrt_cycle",
    "rwth_log_cycle",
    "rwth_cell_progress",
]


def transform(table, meta=None):
    out = table.copy()
    max_cycle = out.groupby(["dataset", "cell_id"])["cycle_index"].transform("max").replace(0, 1)
    out["rwth_cycle_norm"] = out["cycle_index"] / max_cycle
    out["rwth_sqrt_cycle"] = np.sqrt(out["cycle_index"].clip(lower=0))
    out["rwth_log_cycle"] = np.log1p(out["cycle_index"].clip(lower=0))
    out["rwth_cell_progress"] = out.groupby(["dataset", "cell_id"]).cumcount() / out.groupby(["dataset", "cell_id"])["cycle_index"].transform("count").clip(lower=1)
    return out
'''


def ensure_rwth_source(source_dir: Path) -> Path:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (source_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", RWTH_REPO, str(source_dir)],
            check=True,
        )
    mat_path = source_dir / RWTH_MAT
    if not mat_path.exists():
        raise FileNotFoundError(f"RWTH source file not found: {mat_path}")
    return mat_path


def build_rwth_sohbench_table(mat_path: Path, *, dataset_name: str, max_cells: int) -> pd.DataFrame:
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    records: dict[tuple[int, int], list[float]] = {}
    for item in mat["TDS"]:
        cell = int(item.Cell)
        for cycle, capacity in zip(np.asarray(item.History_Cycle).ravel(), np.asarray(item.History).ravel()):
            records.setdefault((cell, int(round(float(cycle)))), []).append(float(capacity))
        for cycle, capacity in zip(np.asarray(item.Target_Cycle).ravel(), np.asarray(item.Target).ravel()):
            records.setdefault((cell, int(round(float(cycle)))), []).append(float(capacity))

    rows = []
    selected_cells = sorted({cell for cell, _ in records})[:max_cells]
    for cell in selected_cells:
        points = sorted(
            (cycle, float(np.mean(values)))
            for (record_cell, cycle), values in records.items()
            if record_cell == cell
        )
        if len(points) < 8:
            continue
        initial_capacity = points[0][1]
        for cycle, capacity in points:
            rows.append(
                {
                    "dataset": dataset_name,
                    "cell_id": f"rwth_cell_{cell:03d}",
                    "cycle_index": cycle,
                    "soh": capacity / initial_capacity,
                    "capacity_ah": capacity,
                    "cycle_index_feature": float(cycle),
                }
            )
    table = pd.DataFrame(rows).sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    table = table[np.isfinite(table["soh"])]
    return table


def write_validation_inputs(work_dir: Path, table: pd.DataFrame) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = work_dir / "rwth_isea_public.csv"
    estimator_path = work_dir / "trusted_huber_decay.py"
    featureset_path = work_dir / "rwth_cycle_shape.py"
    table.to_csv(dataset_path, index=False)
    estimator_path.write_text(ESTIMATOR_PLUGIN, encoding="utf-8")
    featureset_path.write_text(FEATURE_PLUGIN, encoding="utf-8")
    return {
        "dataset": dataset_path,
        "estimator": estimator_path,
        "featureset": featureset_path,
    }


def run_validation(args: argparse.Namespace) -> dict:
    mat_path = ensure_rwth_source(Path(args.source_dir))
    table = build_rwth_sohbench_table(mat_path, dataset_name=args.dataset_name, max_cells=args.max_cells)
    paths = write_validation_inputs(Path(args.work_dir), table)

    service = EvaluationService(
        data_dir=Path(args.data_dir),
        work_dir=Path(args.plugin_work_dir),
        runs_dir=Path(args.runs_dir),
        frontend_data_dir=Path(args.frontend_data_dir),
    )
    service.store.upload_dataset(
        args.dataset_name,
        paths["dataset"].name,
        paths["dataset"].read_bytes(),
        replace=True,
    )
    service.store.upload_estimator_plugin(
        "trusted_huber_decay",
        paths["estimator"].name,
        paths["estimator"].read_bytes(),
        replace=True,
    )
    service.store.upload_featureset_plugin(
        "rwth_cycle_shape",
        paths["featureset"].name,
        paths["featureset"].read_bytes(),
        replace=True,
    )
    job = service.create_job(
        {
            "datasets": [args.dataset_name],
            "estimators": ["trusted_huber_decay"],
            "featuresets": ["rwth_cycle_shape"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    return service.run_job_inline(job["id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("validate trusted local plugin workflow")
    parser.add_argument("--source-dir", default="work/public_sources/rwth-battery-degradation")
    parser.add_argument("--work-dir", default="work/trusted_plugin_validation")
    parser.add_argument("--data-dir", default=r"D:\battery_SOH\data")
    parser.add_argument("--plugin-work-dir", default="work/trusted_plugins")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--dataset-name", default="rwth_isea_public")
    parser.add_argument("--max-cells", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    job = run_validation(parse_args())
    print(job["status"])
    print(job.get("run_dir", job.get("error")))


if __name__ == "__main__":
    main()
