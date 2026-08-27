"""Render the v2.2 Gate -2 candidate route and latest family-readiness matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.readiness_figure import render_readiness_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-status", default="local engineering evidence")
    args = parser.parse_args()
    outputs = render_readiness_matrix(
        args.decision,
        args.output,
        evidence_status=args.evidence_status,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
