"""Resumable three-seed A800 E2 orchestrator; stops before E3 until Gate 2/2b is decided."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from selfsight.analysis.prerequisites import (
    require_gate_minus_one,
    require_generated_domain,
    require_selected_detector_audit,
)
from selfsight.config import load_config


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)


def _seed(config: Path) -> int:
    import yaml

    return int(yaml.safe_load(config.read_text(encoding="utf-8"))["seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/a800_80g.yaml"))
    parser.add_argument("--seed-config", type=Path, action="append", required=True)
    parser.add_argument("--core-python", type=Path, required=True)
    parser.add_argument("--observer-python", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--gate-minus-1-report", type=Path, required=True)
    parser.add_argument("--generated-domain-report", type=Path, required=True)
    parser.add_argument("--migration-report", type=Path, required=True)
    parser.add_argument("--detector-audit-report", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("smolvlm", "internvl", "qwen2vl"), required=True)
    parser.add_argument("--detector-model-id", required=True)
    parser.add_argument("--detector-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_config) != 3:
        raise SystemExit("Exactly three --seed-config paths are required")
    gate = require_gate_minus_one(args.gate_minus_1_report)
    require_selected_detector_audit(
        gate,
        args.detector_audit_report,
        model_id=args.detector_model_id,
        revision=args.detector_revision,
    )
    coverage_mins = [
        float(load_config(path).values["gates"]["verifier_coverage_min"])
        for path in [args.base_config, *args.seed_config]
    ]
    require_generated_domain(
        args.generated_domain_report,
        configured_coverage_min=max(coverage_mins),
    )
    migration = json.loads(args.migration_report.read_text(encoding="utf-8"))
    if not bool(migration.get("passed")):
        raise SystemExit("A800 migration Gate is not green; formal E2 is forbidden")
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
                    str(args.core_python),
                    "scripts/run_gradient_gate.py",
                    "--config",
                    str(config),
                    "--probe-manifest",
                    str(args.probe_manifest),
                    "--gate-minus-1-report",
                    str(args.gate_minus_1_report),
                    "--generated-domain-report",
                    str(args.generated_domain_report),
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
            str(args.core_python),
            "scripts/run_local_pilot.py",
            "--config",
            str(config),
            "--train-manifest",
            str(args.train_manifest),
            "--gate-report",
            str(args.gate_minus_1_report),
            "--gradient-gate-report",
            str(gradient_report),
            "--generated-domain-report",
            str(args.generated_domain_report),
            "--frozen-observer-python",
            str(args.core_python),
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
                    str(args.core_python),
                    "scripts/evaluate_pilot.py",
                    "--config",
                    str(config),
                    "--run-root",
                    str(run_root),
                    "--outcome-manifest",
                    str(args.outcome_manifest),
                    "--probe-manifest",
                    str(args.probe_manifest),
                    "--gate-minus-1-report",
                    str(args.gate_minus_1_report),
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
