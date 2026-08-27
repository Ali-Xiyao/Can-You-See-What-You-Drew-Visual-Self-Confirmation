"""Aggregate exactly three registered A800 seeds and decide Gate 2/2b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.formal import aggregate_formal_e2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/a800_80g_showo2.yaml")
    )
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.metrics) != 3:
        raise SystemExit("Exactly three --metrics paths are required")
    report = aggregate_formal_e2(
        config_path=args.config,
        metric_paths=args.metrics,
        output_dir=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
