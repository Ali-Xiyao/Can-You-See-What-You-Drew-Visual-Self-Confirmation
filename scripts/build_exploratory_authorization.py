"""Build a hash-bound authorization for non-formal post-Gate diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.exploratory import build_exploratory_authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--readiness-config", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--generated-report", type=Path, required=True)
    parser.add_argument("--human-report", type=Path, required=True)
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_exploratory_authorization(
        backbone_config_path=args.backbone_config,
        readiness_config_path=args.readiness_config,
        decision_path=args.decision,
        canary_report_path=args.canary_report,
        reference_report_path=args.reference_report,
        generated_report_path=args.generated_report,
        human_report_path=args.human_report,
        families=args.family,
        output_root=args.output_root,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
