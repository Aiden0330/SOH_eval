from __future__ import annotations

import argparse
import json
from pathlib import Path

from sohbench.evaluation_service import EvaluationService
from sohbench.paths import DEFAULT_DATA_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser("register-battery-soh-plugins")
    parser.add_argument(
        "--battery-soh-root",
        default=None,
        help="battery_SOH checkout; default BATTERY_SOH_ROOT or the sibling battery_SOH directory",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default="work/trusted_plugins")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    service = EvaluationService(
        data_dir=Path(args.data_dir),
        work_dir=Path(args.work_dir),
        runs_dir=Path(args.runs_dir),
        frontend_data_dir=Path(args.frontend_data_dir),
    )
    result = service.import_battery_soh_plugins(
        battery_soh_root=Path(args.battery_soh_root) if args.battery_soh_root else None,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
