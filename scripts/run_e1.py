"""Run the real Show-o Tier-B context experiment after Gate -1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.e1 import run_e1_tier_b
from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_generated_domain,
    require_selected_detector_audit,
)
from selfsight.config import load_config, write_config_snapshot
from selfsight.observers.client import ObserverServiceClient
from selfsight.showo_adapter import ShowoAdapter
from selfsight.utils.evidence import write_host_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path, required=True)
    parser.add_argument("--generated-domain-report", type=Path, required=True)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--limit", type=int)
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, output / "resolved_config.json")
    write_host_manifest(output / "host_manifest.json")
    adapter = ShowoAdapter(device=str(config.values["hardware"]["generator_device"]), trainable=False)
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
        str(config.values["hardware"]["observer_device"]),
        "--ready-report",
        str(output / "detector_ready.json"),
    ]
    with ObserverServiceClient(command, output / "detector_wire.jsonl") as detector:
        report = run_e1_tier_b(
            adapter=adapter,
            detector=detector,
            manifest_path=args.manifest,
            output_dir=output,
            limit=args.limit,
        )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
