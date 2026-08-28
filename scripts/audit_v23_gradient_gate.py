"""Verify the complete local evidence surface of the v2.3 gradient gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.v23.audit import audit_v23_gradient_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_v23_gradient_gate(args.run_dir, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
