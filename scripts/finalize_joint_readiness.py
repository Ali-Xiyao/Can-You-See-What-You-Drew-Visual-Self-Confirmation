"""Finalize the v2.2 Gate -2 decision from immutable A1-A4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.readiness import finalize_joint_readiness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--generated-report", type=Path, required=True)
    parser.add_argument("--human-report", type=Path, required=True)
    parser.add_argument("--lora-report", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_joint_readiness(
        backbone_config_path=args.backbone_config,
        readiness_config_path=args.readiness_config,
        canary_report_path=args.canary_report,
        reference_report_path=args.reference_report,
        generated_report_path=args.generated_report,
        human_report_path=args.human_report,
        lora_report_path=args.lora_report,
        predecessor_path=args.predecessor,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

