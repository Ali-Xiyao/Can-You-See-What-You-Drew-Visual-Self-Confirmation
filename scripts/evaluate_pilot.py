"""Evaluate every local checkpoint and render the exploratory Figure 1 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_selected_detector_audit,
)
from selfsight.pilot.evaluate import evaluate_paired_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path, required=True)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    gate = require_gate_minus_one(args.gate_minus_1_report)
    require_selected_detector_audit(
        gate,
        args.detector_audit_report,
        model_id=args.detector_model_id,
        revision=args.detector_revision,
    )
    command = [
        str(args.detector_python.resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        args.detector_backend,
        "--model-id",
        args.detector_model_id,
        "--revision",
        args.detector_revision,
        "--device",
        args.device,
    ]
    report = evaluate_paired_run(
        config_path=args.config,
        run_root=args.run_root,
        outcome_manifest=args.outcome_manifest,
        probe_manifest=args.probe_manifest,
        detector_audit_report=args.detector_audit_report,
        detector_command=command,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
