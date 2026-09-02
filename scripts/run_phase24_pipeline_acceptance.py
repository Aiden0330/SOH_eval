from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_CURVE_SOURCE = DEFAULT_DATA_DIR / "curves" / "nasa_curves.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the measured phase 2.4 case pipeline and sign its EvidencePack"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--curve-source",
        default=str(DEFAULT_CURVE_SOURCE),
    )
    parser.add_argument("--profile", default="")
    parser.add_argument("--dataset", default="nasa")
    parser.add_argument("--asset", default="B0018")
    parser.add_argument("--estimator", default="ridge")
    parser.add_argument("--featureset", default="full")
    parser.add_argument("--protocol", default="loco")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="work/trusted_plugins/phase24_pipeline_summary.json",
    )
    args = parser.parse_args()

    root = ROOT
    import sohbench

    resolved = Path(sohbench.__file__).resolve()
    expected = root / "src" / "sohbench" / "__init__.py"
    if resolved != expected:
        raise SystemExit(
            f"Refusing to run with stale sohbench import: {resolved}; expected {expected}. "
            "Set PYTHONPATH to the active repository src directory."
        )

    from sohbench.evaluation_service import EvaluationService
    from sohbench.phase24_pipeline import Phase24PipelineConfig, run_phase24_pipeline

    service = EvaluationService(
        data_dir=Path(args.data_dir),
        work_dir=root / "work" / "trusted_plugins",
        runs_dir=root / "runs",
        frontend_data_dir=root / "frontend" / "app" / "data",
    )
    summary = run_phase24_pipeline(
        service,
        Phase24PipelineConfig(
            profile_name=args.profile,
            dataset=args.dataset,
            asset_id=args.asset,
            estimator=args.estimator,
            featureset=args.featureset,
            protocol=args.protocol,
            seed=args.seed,
            curve_source=args.curve_source,
        ),
    )
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
