"""Materialize the isolated v2.2 A1/A2/A3 benchmark manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from selfsight.data.readiness import DATA_NAMESPACE, materialize_readiness_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    output = args.output
    if output is None:
        data_root = os.environ.get("SELFSIGHT_DATA_ROOT")
        if not data_root:
            raise SystemExit("Run scripts/set_h_env.ps1 or pass --output")
        output = Path(data_root) / DATA_NAMESPACE
    report = materialize_readiness_dataset(output, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

