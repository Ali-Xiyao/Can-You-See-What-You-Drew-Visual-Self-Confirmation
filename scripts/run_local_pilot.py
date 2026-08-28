"""Launch/resume paired E2 after v2.2 Gate -2 or frozen v2.1 prerequisites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.lora_selection import validate_lora_target_selection
from selfsight.config import load_config
from selfsight.pilot.real_loop import run_real_paired_pilot
from selfsight.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--gradient-gate-report", type=Path, required=True)
    parser.add_argument("--generated-domain-report", type=Path)
    parser.add_argument("--joint-readiness-decision", type=Path)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=Path("configs/backbones/showo2_1p5b.yaml"),
    )
    parser.add_argument("--lora-target-config", type=Path)
    parser.add_argument(
        "--frozen-observer-python",
        type=Path,
        default=Path(os.environ.get("SELFSIGHT_ENV_ROOT", "envs")) / "core" / "python.exe",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lora_targets = None
    if args.joint_readiness_decision is not None:
        if args.gate_report is not None or args.generated_domain_report is not None:
            parser.error("Do not mix v2.2 Joint Readiness with legacy v2.1 inputs")
        if args.lora_target_config is None:
            parser.error("v2.2 E2 requires --lora-target-config")
        config = load_config(args.config)
        decision = require_joint_readiness(args.joint_readiness_decision)
        backbone = yaml.safe_load(args.backbone_config.read_text(encoding="utf-8"))
        if (
            decision.get("model_id") != backbone.get("backbone_id")
            or decision.get("revision") != backbone.get("revision")
            or config.values["model"].get("trainable_id") != backbone.get("backbone_id")
        ):
            raise RuntimeError("E2 config/backbone identity does not match Gate -2")
        canary_path = Path(str(decision["evidence"]["canary"]["path"])).resolve()
        lora_report_path = Path(str(decision["evidence"]["lora"]["path"])).resolve()
        lora_report = json.loads(lora_report_path.read_text(encoding="utf-8"))
        target_path = args.lora_target_config.resolve()
        target_selection = validate_lora_target_selection(
            target_path, canary_report=canary_path
        )
        if (
            lora_report.get("target_config_sha256") != sha256_file(target_path)
            or lora_report.get("target_selection_digest")
            != target_selection.get("selection_digest")
        ):
            raise RuntimeError("E2 LoRA target selection does not match Gate -2 A4")
        lora_targets = tuple(str(item) for item in target_selection["target_modules"])
    elif args.gate_report is None or args.generated_domain_report is None:
        parser.error(
            "Legacy E2 requires --gate-report and --generated-domain-report; "
            "v2.2 requires --joint-readiness-decision"
        )
    report = run_real_paired_pilot(
        config_path=args.config,
        train_manifest=args.train_manifest,
        gate_report=args.gate_report,
        gradient_gate_report=args.gradient_gate_report,
        generated_domain_report=args.generated_domain_report,
        frozen_observer_python=args.frozen_observer_python,
        output_dir=args.output,
        resume=args.resume,
        joint_readiness_decision=args.joint_readiness_decision,
        backbone_config=args.backbone_config if args.joint_readiness_decision else None,
        lora_target_modules=lora_targets,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
