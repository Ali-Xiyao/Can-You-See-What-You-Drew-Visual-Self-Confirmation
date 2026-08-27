"""Run Gate -1b with the capability-matched heterogeneous detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.gradient_gate import run_gradient_gate
from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_generated_domain,
    require_selected_detector_audit,
)
from selfsight.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path, required=True)
    parser.add_argument("--generated-domain-report", type=Path, required=True)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    gate = require_gate_minus_one(args.gate_minus_1_report)
    require_selected_detector_audit(
        gate,
        args.detector_audit_report,
        model_id=args.detector_model_id,
        revision=args.detector_revision,
    )
    require_generated_domain(
        args.generated_domain_report,
        configured_coverage_min=float(config.values["gates"]["verifier_coverage_min"]),
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
    report = run_gradient_gate(
        config_path=args.config,
        probe_manifest=args.probe_manifest,
        detector_command=command,
        output_dir=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
