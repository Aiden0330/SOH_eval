"""Build per-dataset window banks for the tuned-window search dimension.

Usage:
    python scripts/build_window_bank.py --datasets hust snl_lfp uconn calce_cs2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sohbench.window_bank import DATASET_LATTICE, bank_path, build_bank  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("build-window-bank")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LATTICE))
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        print(f"building window bank for {dataset} ...", flush=True)
        bank = build_bank(args.data_dir, dataset)
        cycles = bank[["cell_id", "cycle_index"]].drop_duplicates()
        print(
            f"  {dataset}: {len(bank)} rows, {len(cycles)} cycles, "
            f"{bank[['lo_mv', 'hi_mv']].drop_duplicates().shape[0]} windows "
            f"-> {bank_path(args.data_dir, dataset)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
