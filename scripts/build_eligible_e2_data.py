"""Materialize Gate -2 family-restricted E2 data after one green decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from selfsight.data.eligible_e2 import materialize_eligible_e2_dataset
from selfsight.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    decision = args.decision.resolve()
    output = args.output
    if output is None:
        data_root = os.environ.get("SELFSIGHT_DATA_ROOT")
        if not data_root:
            raise SystemExit("Set SELFSIGHT_DATA_ROOT or pass --output")
        output = (
            Path(data_root)
            / "selfsight-v2.2"
            / f"e2-{sha256_file(decision)[:12]}"
        )
    report = materialize_eligible_e2_dataset(
        decision,
        output,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
