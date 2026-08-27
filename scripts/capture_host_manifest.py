"""Capture immutable host/software facts for an experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

from selfsight.utils.evidence import write_host_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = write_host_manifest(args.output)
    print(path)


if __name__ == "__main__":
    main()
