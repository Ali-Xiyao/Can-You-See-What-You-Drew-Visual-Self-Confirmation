"""Run family-restricted E1 after v2.2 Gate -2 or the frozen v2.1 Gate -1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.analysis.e1 import run_e1_tier_b
from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_generated_domain,
    require_public_observer_audit,
    require_selected_detector_audit,
)
from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.config import load_config, write_config_snapshot
from selfsight.observers.client import ObserverServiceClient
from selfsight.showo_adapter import ShowoAdapter
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path)
    parser.add_argument("--generated-domain-report", type=Path)
    parser.add_argument("--joint-readiness-decision", type=Path)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=Path("configs/backbones/showo2_1p5b.yaml"),
    )
    parser.add_argument(
        "--observer-config",
        type=Path,
        default=Path("configs/observers/qwen2vl_2b.yaml"),
    )
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    eligible_families = None
    evidence_bindings = {}
    if args.joint_readiness_decision is not None:
        if args.gate_minus_1_report is not None or args.generated_domain_report is not None:
            parser.error("Do not mix v2.2 Joint Readiness with legacy v2.1 Gate -1 inputs")
        decision_path = args.joint_readiness_decision.resolve()
        decision = require_joint_readiness(decision_path)
        eligible_families = tuple(str(item) for item in decision["selected_eligible_families"])
        backbone_path = args.backbone_config.resolve()
        backbone = yaml.safe_load(backbone_path.read_text(encoding="utf-8"))
        if (
            decision.get("model_id") != backbone.get("backbone_id")
            or decision.get("revision") != backbone.get("revision")
        ):
            raise RuntimeError("E1 backbone config does not match the Gate -2 decision")
        experiment_model = config.values["model"]
        official = backbone["official_profile"]
        if (
            experiment_model.get("trainable_id") != backbone.get("backbone_id")
            or int(experiment_model.get("image_resolution", -1))
            != int(official["resolution"])
            or int(experiment_model.get("generation_timesteps", -1))
            != int(official["generation_steps"])
        ):
            raise RuntimeError("E1 experiment config does not match the locked Show-o2 profile")
        observer_path = args.observer_config.resolve()
        observer = yaml.safe_load(observer_path.read_text(encoding="utf-8"))
        if (
            observer.get("observer_id") != args.detector_model_id
            or observer.get("revision") != args.detector_revision
        ):
            raise RuntimeError("E1 detector does not match the locked public-observer config")
        require_public_observer_audit(
            args.detector_audit_report,
            model_id=args.detector_model_id,
            revision=args.detector_revision,
            eligible_families=eligible_families,
            family_accuracy_min=float(
                config.values["gates"]["observer_family_accuracy_min"]
            ),
            yes_bias_max=float(config.values["gates"]["forced_choice_bias_max"]),
            abstain_rate_max=float(config.values["gates"]["observer_abstain_max"]),
        )
        adapter = Showo2Adapter(
            backbone_config=backbone_path,
            device=str(backbone["hardware"]["generator_device"]),
            lazy=False,
        )
        evidence_bindings = {
            "joint_readiness_decision": {
                "path": str(decision_path),
                "sha256": sha256_file(decision_path),
            },
            "backbone_config": {
                "path": str(backbone_path),
                "sha256": sha256_file(backbone_path),
            },
            "public_observer_audit": {
                "path": str(args.detector_audit_report.resolve()),
                "sha256": sha256_file(args.detector_audit_report.resolve()),
            },
        }
    else:
        if args.gate_minus_1_report is None or args.generated_domain_report is None:
            parser.error(
                "Legacy E1 requires --gate-minus-1-report and --generated-domain-report; "
                "v2.2 requires --joint-readiness-decision"
            )
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
        adapter = ShowoAdapter(
            device=str(config.values["hardware"]["generator_device"]), trainable=False
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, output / "resolved_config.json")
    write_host_manifest(output / "host_manifest.json")
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
            eligible_families=eligible_families,
            evidence_bindings=evidence_bindings,
        )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
