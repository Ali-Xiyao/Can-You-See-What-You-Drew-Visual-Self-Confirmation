"""Materialize version-isolated v2.3 manifests from existing reference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.v23.data import materialize_v23_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_v23_data(source_root=args.source_root, output_root=args.output_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
