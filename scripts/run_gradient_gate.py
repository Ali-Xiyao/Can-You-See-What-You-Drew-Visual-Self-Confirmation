"""Run Gate -1b after v2.2 Gate -2 or the frozen v2.1 prerequisites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.analysis.exploratory import (
    validate_bound_exploratory_authorization,
    validate_exploratory_observer_audit,
)
from selfsight.analysis.gradient_gate import run_gradient_gate
from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_generated_domain,
    require_public_observer_audit,
    require_selected_detector_audit,
)
from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.lora_selection import validate_lora_target_selection
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.config import load_config
from selfsight.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path)
    parser.add_argument("--generated-domain-report", type=Path)
    parser.add_argument("--joint-readiness-decision", type=Path)
    parser.add_argument("--exploratory-authorization", type=Path)
    parser.add_argument("--frozen-decision", type=Path)
    parser.add_argument("--exploratory-a4-report", type=Path)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = None
    lora_targets = None
    eligible_families = None
    evidence_bindings = {}
    exploratory_authorization = None
    if args.exploratory_authorization is not None:
        if any(
            item is not None
            for item in (
                args.joint_readiness_decision,
                args.gate_minus_1_report,
                args.generated_domain_report,
            )
        ):
            parser.error("Do not mix exploratory authorization with registered Gate inputs")
        if args.frozen_decision is None or args.exploratory_a4_report is None:
            parser.error(
                "Exploratory Gate -1b requires --frozen-decision and --exploratory-a4-report"
            )
        if args.lora_target_config is None:
            parser.error("Exploratory Gate -1b requires --lora-target-config")
        output = args.output.resolve()
        exploratory_authorization = validate_bound_exploratory_authorization(
            args.exploratory_authorization,
            stage="gradient_gate",
            backbone_config_path=args.backbone_config,
            decision_path=args.frozen_decision,
            output_path=output,
        )
        eligible_families = tuple(
            str(item) for item in exploratory_authorization["families"]
        )
        backbone_path = args.backbone_config.resolve()
        backbone = yaml.safe_load(backbone_path.read_text(encoding="utf-8"))
        experiment_model = config.values["model"]
        official = backbone["official_profile"]
        if (
            experiment_model.get("trainable_id") != backbone.get("backbone_id")
            or int(experiment_model.get("image_resolution", -1))
            != int(official["resolution"])
            or int(experiment_model.get("generation_timesteps", -1))
            != int(official["generation_steps"])
        ):
            raise RuntimeError("Exploratory Gate -1b config does not match the locked HQ profile")
        observer_path = args.observer_config.resolve()
        observer = yaml.safe_load(observer_path.read_text(encoding="utf-8"))
        if (
            observer.get("observer_id") != args.detector_model_id
            or observer.get("revision") != args.detector_revision
        ):
            raise RuntimeError("Exploratory Gate -1b detector identity mismatch")
        validate_exploratory_observer_audit(
            args.detector_audit_report,
            model_id=args.detector_model_id,
            revision=args.detector_revision,
            eligible_families=eligible_families,
            family_accuracy_min=float(config.values["gates"]["observer_family_accuracy_min"]),
            yes_bias_max=float(config.values["gates"]["forced_choice_bias_max"]),
            abstain_rate_max=float(config.values["gates"]["observer_abstain_max"]),
        )
        canary_path = Path(
            str(exploratory_authorization["evidence"]["canary"]["path"])
        ).resolve()
        target_path = args.lora_target_config.resolve()
        target_sha = sha256_file(target_path)
        target_selection = validate_lora_target_selection(
            target_path, canary_report=canary_path
        )
        a4_path = args.exploratory_a4_report.resolve()
        a4 = json.loads(a4_path.read_text(encoding="utf-8"))
        authorization_sha = sha256_file(args.exploratory_authorization.resolve())
        if (
            a4.get("passed") is not True
            or a4.get("non_formal") is not True
            or a4.get("exploratory_authorization_sha256") != authorization_sha
            or a4.get("target_config_sha256") != target_sha
            or a4.get("target_selection_digest") != target_selection.get("selection_digest")
        ):
            raise RuntimeError("Exploratory A4 evidence does not bind this gradient route")
        lora_targets = tuple(str(item) for item in target_selection["target_modules"])
        adapter = Showo2Adapter(
            backbone_config=backbone_path,
            device=str(backbone["hardware"]["generator_device"]),
            lazy=True,
        )
        evidence_bindings = {
            "exploratory_authorization": {
                "path": str(args.exploratory_authorization.resolve()),
                "sha256": authorization_sha,
            },
            "frozen_red_decision": {
                "path": str(args.frozen_decision.resolve()),
                "sha256": sha256_file(args.frozen_decision.resolve()),
            },
            "exploratory_a4": {"path": str(a4_path), "sha256": sha256_file(a4_path)},
            "backbone_config": {
                "path": str(backbone_path),
                "sha256": sha256_file(backbone_path),
            },
            "public_observer_audit": {
                "path": str(args.detector_audit_report.resolve()),
                "sha256": sha256_file(args.detector_audit_report.resolve()),
            },
            "lora_target_config": {"path": str(target_path), "sha256": target_sha},
        }
    elif args.joint_readiness_decision is not None:
        if args.gate_minus_1_report is not None or args.generated_domain_report is not None:
            parser.error("Do not mix v2.2 Joint Readiness with legacy v2.1 inputs")
        if args.lora_target_config is None:
            parser.error("v2.2 Gate -1b requires --lora-target-config")
        decision_path = args.joint_readiness_decision.resolve()
        decision = require_joint_readiness(decision_path)
        eligible_families = tuple(str(item) for item in decision["selected_eligible_families"])
        backbone_path = args.backbone_config.resolve()
        backbone = yaml.safe_load(backbone_path.read_text(encoding="utf-8"))
        if (
            decision.get("model_id") != backbone.get("backbone_id")
            or decision.get("revision") != backbone.get("revision")
        ):
            raise RuntimeError("Gate -1b backbone config does not match Gate -2")
        experiment_model = config.values["model"]
        official = backbone["official_profile"]
        if (
            experiment_model.get("trainable_id") != backbone.get("backbone_id")
            or int(experiment_model.get("image_resolution", -1))
            != int(official["resolution"])
            or int(experiment_model.get("generation_timesteps", -1))
            != int(official["generation_steps"])
        ):
            raise RuntimeError(
                "Gate -1b experiment config does not match the locked Show-o2 profile"
            )
        observer_path = args.observer_config.resolve()
        observer = yaml.safe_load(observer_path.read_text(encoding="utf-8"))
        if (
            observer.get("observer_id") != args.detector_model_id
            or observer.get("revision") != args.detector_revision
        ):
            raise RuntimeError("Gate -1b detector does not match the public-observer config")
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
        lora_report_path = Path(str(decision["evidence"]["lora"]["path"])).resolve()
        lora_report = json.loads(lora_report_path.read_text(encoding="utf-8"))
        target_path = args.lora_target_config.resolve()
        target_sha = sha256_file(target_path)
        if lora_report.get("target_config_sha256") != target_sha:
            raise RuntimeError("LoRA target config does not match Gate -2 A4 evidence")
        target_selection = validate_lora_target_selection(
            target_path, canary_report=canary_path
        )
        if lora_report.get("target_selection_digest") != target_selection.get(
            "selection_digest"
        ):
            raise RuntimeError("LoRA target selection digest does not match Gate -2 A4")
        lora_targets = tuple(str(item) for item in target_selection["target_modules"])
        adapter = Showo2Adapter(
            backbone_config=backbone_path,
            device=str(backbone["hardware"]["generator_device"]),
            lazy=True,
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
            "lora_target_config": {"path": str(target_path), "sha256": target_sha},
        }
    else:
        if args.gate_minus_1_report is None or args.generated_domain_report is None:
            parser.error(
                "Legacy Gate -1b requires --gate-minus-1-report and "
                "--generated-domain-report; v2.2 requires --joint-readiness-decision"
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
        adapter=adapter,
        lora_target_modules=lora_targets,
        eligible_families=eligible_families,
        evidence_bindings=evidence_bindings,
        non_formal=exploratory_authorization is not None,
        exploratory_authorization=(
            None
            if exploratory_authorization is None
            else {
                "path": str(args.exploratory_authorization.resolve()),
                "sha256": sha256_file(args.exploratory_authorization.resolve()),
                "digest": exploratory_authorization["authorization_digest"],
            }
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
