"""Show live AR tide iteration training status.

Usage (repo root):
    venv\\Scripts\\python.exe scripts/ar_tide_iter/show_status.py --id iter30
    venv\\Scripts\\python.exe scripts/ar_tide_iter/show_status.py --id iter30 --watch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_log import format_status, refresh_status_from_log


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True, help="Experiment id, e.g. iter30")
    p.add_argument(
        "--watch",
        action="store_true",
        help="Re-read log every 10s until interrupted",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between refreshes with --watch (default: 10)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    while True:
        status = refresh_status_from_log(args.id)
        sys.stdout.write(format_status(status) + "\n")
        sys.stdout.flush()
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
