"""Resumable three-seed A800 E2 orchestrator; stops before E3 until Gate 2/2b is decided."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from selfsight.analysis.prerequisites import require_public_observer_audit
from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.lora_selection import validate_lora_target_selection
from selfsight.config import load_config
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import read_jsonl


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)


def _seed(config: Path) -> int:
    import yaml

    return int(yaml.safe_load(config.read_text(encoding="utf-8"))["seed"])


def _seed_invariant(values: dict) -> dict:
    normalized = deepcopy(values)
    normalized.pop("profile", None)
    normalized.pop("seed", None)
    normalized["training"].pop("seeds", None)
    return normalized


def _require_eligible_manifest_capacity(
    path: Path,
    *,
    families: tuple[str, ...],
    required: int,
    label: str,
) -> None:
    family_set = set(families)
    ids = {
        str(row["scene"]["scene_id"])
        for row in read_jsonl(path)
        if str(row["scene"]["family"]) in family_set
    }
    if len(ids) < required:
        raise RuntimeError(
            f"Formal {label} manifest has {len(ids)} unique eligible cases; {required} required"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/a800_80g_showo2.yaml")
    )
    parser.add_argument("--seed-config", type=Path, action="append", required=True)
    parser.add_argument("--core-python", type=Path, required=True)
    parser.add_argument("--showo2-python", type=Path, required=True)
    parser.add_argument("--observer-python", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--joint-readiness-decision", type=Path, required=True)
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
    parser.add_argument("--lora-target-config", type=Path, required=True)
    parser.add_argument("--migration-report", type=Path, required=True)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_config) != 3:
        raise SystemExit("Exactly three --seed-config paths are required")
    decision = require_joint_readiness(args.joint_readiness_decision)
    eligible_families = tuple(str(item) for item in decision["selected_eligible_families"])
    backbone = yaml.safe_load(args.backbone_config.read_text(encoding="utf-8"))
    observer = yaml.safe_load(args.observer_config.read_text(encoding="utf-8"))
    if (
        decision.get("model_id") != backbone.get("backbone_id")
        or decision.get("revision") != backbone.get("revision")
    ):
        raise RuntimeError("Formal E2 backbone does not match Gate -2")
    if (
        observer.get("observer_id") != args.detector_model_id
        or observer.get("revision") != args.detector_revision
    ):
        raise RuntimeError("Formal E2 detector does not match public-observer config")
    configs = [load_config(path) for path in [args.base_config, *args.seed_config]]
    official = backbone["official_profile"]
    for config in configs:
        model = config.values["model"]
        if (
            model.get("trainable_id") != backbone.get("backbone_id")
            or int(model.get("image_resolution", -1)) != int(official["resolution"])
            or int(model.get("generation_timesteps", -1))
            != int(official["generation_steps"])
        ):
            raise RuntimeError(f"Formal config has the wrong backbone: {config.source}")
    base_config = configs[0]
    seed_configs = configs[1:]
    registered_seeds = sorted(int(item) for item in base_config.values["training"]["seeds"])
    materialized_seeds = sorted(int(config.values["seed"]) for config in seed_configs)
    if len(set(materialized_seeds)) != 3 or materialized_seeds != registered_seeds:
        raise RuntimeError(
            f"Formal seed configs do not match registered seeds: {materialized_seeds} != "
            f"{registered_seeds}"
        )
    invariant = _seed_invariant(base_config.values)
    for config in seed_configs:
        if _seed_invariant(config.values) != invariant:
            raise RuntimeError(f"Formal seed config changes more than seed/profile: {config.source}")
    require_public_observer_audit(
        args.detector_audit_report,
        model_id=args.detector_model_id,
        revision=args.detector_revision,
        eligible_families=eligible_families,
        family_accuracy_min=max(
            float(config.values["gates"]["observer_family_accuracy_min"])
            for config in configs
        ),
        yes_bias_max=min(
            float(config.values["gates"]["forced_choice_bias_max"])
            for config in configs
        ),
        abstain_rate_max=min(
            float(config.values["gates"]["observer_abstain_max"])
            for config in configs
        ),
    )
    canary_path = Path(str(decision["evidence"]["canary"]["path"])).resolve()
    lora_report = json.loads(
        Path(str(decision["evidence"]["lora"]["path"])).read_text(encoding="utf-8")
    )
    target_selection = validate_lora_target_selection(
        args.lora_target_config, canary_report=canary_path
    )
    if (
        lora_report.get("target_config_sha256") != sha256_file(args.lora_target_config)
        or lora_report.get("target_selection_digest")
        != target_selection.get("selection_digest")
    ):
        raise RuntimeError("Formal E2 LoRA targets do not match Gate -2 A4")
    migration = json.loads(args.migration_report.read_text(encoding="utf-8"))
    conditions = migration.get("conditions")
    calculated_migration = bool(conditions) and all(
        bool(value) for value in conditions.values()
    )
    if (
        migration.get("gate") != "a800_migration"
        or bool(migration.get("passed")) != calculated_migration
        or not calculated_migration
    ):
        raise SystemExit("A800 migration Gate is not green; formal E2 is forbidden")
    local_summary = migration.get("local_summary", {})
    a800_summary = migration.get("a800_summary", {})
    if (
        local_summary.get("revision") != decision.get("revision")
        or a800_summary.get("revision") != decision.get("revision")
    ):
        raise RuntimeError("A800 migration Gate does not match the Gate -2 backbone revision")
    migration_evidence = migration.get("evidence")
    if not isinstance(migration_evidence, Mapping) or len(migration_evidence) != 4:
        raise RuntimeError("A800 migration Gate has no complete evidence index")
    for label, record in migration_evidence.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"A800 migration evidence is malformed: {label}")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"A800 migration evidence SHA-256 mismatch: {label}")
    _require_eligible_manifest_capacity(
        args.train_manifest,
        families=eligible_families,
        required=int(base_config.values["training"]["rounds"])
        * int(base_config.values["training"]["prompts_per_round"]),
        label="training",
    )
    _require_eligible_manifest_capacity(
        args.outcome_manifest,
        families=eligible_families,
        required=int(base_config.values["data"]["tier_a_outcome"]),
        label="outcome",
    )
    _require_eligible_manifest_capacity(
        args.probe_manifest,
        families=eligible_families,
        required=int(base_config.values["gradient_probe"]["size"]),
        label="gradient-probe",
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for config in sorted(args.seed_config, key=_seed):
        seed = _seed(config)
        seed_root = output / f"seed-{seed}"
        seed_root.mkdir(exist_ok=True)
        gradient_root = seed_root / "gate-minus-1b"
        gradient_report = gradient_root / "gate_minus_1b.json"
        if not gradient_report.is_file():
            if gradient_root.exists() and any(gradient_root.iterdir()):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                os.replace(gradient_root, seed_root / f"gate-minus-1b.abandoned-{stamp}")
            _run(
                [
                    str(args.showo2_python),
                    "scripts/run_gradient_gate.py",
                    "--config",
                    str(config),
                    "--probe-manifest",
                    str(args.probe_manifest),
                    "--joint-readiness-decision",
                    str(args.joint_readiness_decision),
                    "--backbone-config",
                    str(args.backbone_config),
                    "--observer-config",
                    str(args.observer_config),
                    "--lora-target-config",
                    str(args.lora_target_config),
                    "--detector-audit-report",
                    str(args.detector_audit_report),
                    "--detector-python",
                    str(args.observer_python),
                    "--detector-backend",
                    args.detector_backend,
                    "--detector-model-id",
                    args.detector_model_id,
                    "--detector-revision",
                    args.detector_revision,
                    "--device",
                    "cuda:0",
                    "--output",
                    str(gradient_root),
                ],
                seed_root / "logs" / "gate-minus-1b.log",
            )
        run_root = seed_root / "e2"
        training_report = run_root / "training_report.json"
        command = [
            str(args.showo2_python),
            "scripts/run_local_pilot.py",
            "--config",
            str(config),
            "--train-manifest",
            str(args.train_manifest),
            "--joint-readiness-decision",
            str(args.joint_readiness_decision),
            "--backbone-config",
            str(args.backbone_config),
            "--lora-target-config",
            str(args.lora_target_config),
            "--gradient-gate-report",
            str(gradient_report),
            "--frozen-observer-python",
            str(args.showo2_python),
            "--output",
            str(run_root),
        ]
        if run_root.exists() and any(run_root.iterdir()):
            command.append("--resume")
        if not training_report.is_file():
            _run(command, seed_root / "logs" / "training.log")
        evaluation_report = run_root / "evaluations" / "evaluation_report.json"
        if not evaluation_report.is_file():
            _run(
                [
                    str(args.showo2_python),
                    "scripts/evaluate_pilot.py",
                    "--config",
                    str(config),
                    "--run-root",
                    str(run_root),
                    "--outcome-manifest",
                    str(args.outcome_manifest),
                    "--probe-manifest",
                    str(args.probe_manifest),
                    "--joint-readiness-decision",
                    str(args.joint_readiness_decision),
                    "--backbone-config",
                    str(args.backbone_config),
                    "--observer-config",
                    str(args.observer_config),
                    "--lora-target-config",
                    str(args.lora_target_config),
                    "--detector-audit-report",
                    str(args.detector_audit_report),
                    "--detector-python",
                    str(args.observer_python),
                    "--detector-backend",
                    args.detector_backend,
                    "--detector-model-id",
                    args.detector_model_id,
                    "--detector-revision",
                    args.detector_revision,
                    "--device",
                    "cuda:0",
                ],
                seed_root / "logs" / "evaluation.log",
            )
        metrics.append(run_root / "evaluations" / "checkpoint_metrics.csv")
    aggregate = output / "formal-aggregate"
    if not (aggregate / "formal_gate_2_2b.json").is_file():
        command = [
            str(args.core_python),
            "scripts/aggregate_formal_e2.py",
            "--config",
            str(args.base_config),
            "--output",
            str(aggregate),
        ]
        for path in metrics:
            command.extend(["--metrics", str(path)])
        _run(command, output / "formal-aggregate.log")
    report = json.loads((aggregate / "formal_gate_2_2b.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
