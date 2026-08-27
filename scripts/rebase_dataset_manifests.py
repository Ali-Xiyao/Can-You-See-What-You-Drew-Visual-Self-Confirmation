"""Create a platform-local, hash-verified view of SelfSight dataset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.data.portability import rebase_dataset_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rebase_dataset_manifests(args.data_root, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
