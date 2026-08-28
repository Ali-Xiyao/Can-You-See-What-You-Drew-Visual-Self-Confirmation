"""Finalize the v2.2 Gate -2 decision from immutable A1-A4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.readiness import (
    finalize_joint_readiness,
    finalize_joint_readiness_stop,
    finalize_joint_readiness_stop_after_human,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--generated-report", type=Path, required=True)
    parser.add_argument("--human-report", type=Path)
    parser.add_argument("--lora-report", type=Path)
    parser.add_argument("--stop-before-human-a4", action="store_true")
    parser.add_argument("--stop-after-human-before-a4", action="store_true")
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "backbone_config_path": args.backbone_config,
        "readiness_config_path": args.readiness_config,
        "canary_report_path": args.canary_report,
        "reference_report_path": args.reference_report,
        "generated_report_path": args.generated_report,
        "predecessor_path": args.predecessor,
        "output_path": args.output,
    }
    if args.stop_before_human_a4 and args.stop_after_human_before_a4:
        parser.error("Choose only one registered stop mode")
    if args.stop_before_human_a4:
        if args.human_report is not None or args.lora_report is not None:
            parser.error("Stop mode must not provide human or A4 reports")
        report = finalize_joint_readiness_stop(**common)
    elif args.stop_after_human_before_a4:
        if args.human_report is None or args.lora_report is not None:
            parser.error("Human-stop mode requires --human-report and forbids --lora-report")
        report = finalize_joint_readiness_stop_after_human(
            **common,
            human_report_path=args.human_report,
        )
    else:
        if args.human_report is None or args.lora_report is None:
            parser.error("Full Gate -2 finalization requires --human-report and --lora-report")
        report = finalize_joint_readiness(
            **common,
            human_report_path=args.human_report,
            lora_report_path=args.lora_report,
        )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
