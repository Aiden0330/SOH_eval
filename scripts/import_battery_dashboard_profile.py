from __future__ import annotations

import argparse
import json
from pathlib import Path

from sohbench.battery_dashboard_import import publish_battery_dashboard_profile
from sohbench.paths import DEFAULT_CURVE_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser("import-battery-dashboard-profile")
    parser.add_argument("--source-dir", default=str(DEFAULT_CURVE_DIR))
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--profile-name", default="battery-soh-diagnostics")
    parser.add_argument("--include-curves", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    result = publish_battery_dashboard_profile(
        Path(args.frontend_data_dir),
        Path(args.source_dir),
        include_curves=args.include_curves,
        profile_name=args.profile_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
