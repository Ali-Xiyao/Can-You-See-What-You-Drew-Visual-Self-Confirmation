"""Evaluate every paired checkpoint under v2.2 Gate -2 or frozen v2.1 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_public_observer_audit,
    require_selected_detector_audit,
)
from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.lora_selection import validate_lora_target_selection
from selfsight.config import load_config
from selfsight.pilot.evaluate import evaluate_paired_run
from selfsight.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path)
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
    parser.add_argument("--lora-target-config", type=Path)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-python", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    eligible_families = ()
    lora_targets = None
    backbone_path = None
    if args.joint_readiness_decision is not None:
        if args.gate_minus_1_report is not None or args.lora_target_config is None:
            parser.error(
                "v2.2 evaluation requires --lora-target-config and cannot mix Gate -1"
            )
        config = load_config(args.config)
        decision = require_joint_readiness(args.joint_readiness_decision)
        eligible_families = tuple(
            str(item) for item in decision["selected_eligible_families"]
        )
        backbone_path = args.backbone_config.resolve()
        backbone = yaml.safe_load(backbone_path.read_text(encoding="utf-8"))
        if (
            decision.get("model_id") != backbone.get("backbone_id")
            or decision.get("revision") != backbone.get("revision")
            or config.values["model"].get("trainable_id") != backbone.get("backbone_id")
        ):
            raise RuntimeError("Evaluation config/backbone does not match Gate -2")
        observer = yaml.safe_load(args.observer_config.read_text(encoding="utf-8"))
        if (
            observer.get("observer_id") != args.detector_model_id
            or observer.get("revision") != args.detector_revision
        ):
            raise RuntimeError("Evaluation detector does not match public-observer config")
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
        canary_path = Path(str(decision["evidence"]["canary"]["path"])).resolve()
        lora_report = json.loads(
            Path(str(decision["evidence"]["lora"]["path"])).read_text(encoding="utf-8")
        )
        target_path = args.lora_target_config.resolve()
        selection = validate_lora_target_selection(
            target_path, canary_report=canary_path
        )
        if (
            lora_report.get("target_config_sha256") != sha256_file(target_path)
            or lora_report.get("target_selection_digest") != selection.get("selection_digest")
        ):
            raise RuntimeError("Evaluation LoRA targets do not match Gate -2 A4")
        lora_targets = tuple(str(item) for item in selection["target_modules"])
    else:
        if args.gate_minus_1_report is None:
            parser.error("Legacy evaluation requires --gate-minus-1-report")
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
        backbone_config=backbone_path,
        lora_target_modules=lora_targets,
        eligible_families=eligible_families,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
