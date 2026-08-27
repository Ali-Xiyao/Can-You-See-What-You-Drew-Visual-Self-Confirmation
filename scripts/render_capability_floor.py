"""Render the Gate -1 capability-floor fallback figure from completed audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.capability_figure import render_capability_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-status", default="local Gate -1 audit")
    args = parser.parse_args()
    outputs = render_capability_figure(
        args.report,
        args.output,
        evidence_status=args.evidence_status,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
