"""SelfSight command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _config(path: Path):
    from selfsight.config import load_config

    return load_config(path)


def _cmd_doctor(args: argparse.Namespace) -> int:
    from selfsight.config import write_config_snapshot
    from selfsight.utils.evidence import capture_host_manifest

    config = _config(args.config)
    run_root = Path(config.section("paths")["run_root"])
    output = args.output or (run_root / "host" / f"{config.values['profile']}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    from selfsight.utils.jsonl import atomic_write_json

    manifest = capture_host_manifest()
    manifest["config_digest"] = config.digest
    manifest["profile"] = config.values["profile"]
    atomic_write_json(output, manifest)
    write_config_snapshot(config, output.with_name(output.stem + "-config.json"))
    print(json.dumps({"ok": True, "config_digest": config.digest, "host_manifest": str(output)}, indent=2))
    return 0


def _cmd_build_data(args: argparse.Namespace) -> int:
    from selfsight.data.manifest import create_registered_dataset

    config = _config(args.config)
    root = args.output or (Path(config.section("paths")["data_root"]) / "selfsight-v1")
    outputs = create_registered_dataset(root, int(config.values["seed"]))
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_audit_data(args: argparse.Namespace) -> int:
    from selfsight.data.audit import audit_reference_manifests

    root = args.root.resolve()
    manifest_dir = args.manifest_dir.resolve() if args.manifest_dir else root / "manifests"
    manifests = {
        name: manifest_dir / f"{name}.jsonl"
        for name in ("train", "tier_a_probe", "tier_a_outcome")
    }
    output = args.output or (root / "audits" / "reference_verifier.json")
    report = audit_reference_manifests(manifests, output)
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    return 0 if report["gate_reference_pass"] else 2


def _cmd_mock_pilot(args: argparse.Namespace) -> int:
    from selfsight.pilot.mock_loop import run_mock_pilot

    outputs = run_mock_pilot(args.config, args.output)
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_audit_tier_b(args: argparse.Namespace) -> int:
    from selfsight.data.audit import audit_tier_b_manifest

    report = audit_tier_b_manifest(args.manifest, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    return 0 if report["gate_tier_b_reference_pass"] else 2


def _cmd_build_tier_d(args: argparse.Namespace) -> int:
    from selfsight.data.manifest import materialize_tier_d

    report = materialize_tier_d(args.root, args.seed)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_audit_tier_d(args: argparse.Namespace) -> int:
    from selfsight.data.audit import audit_tier_d_manifest

    report = audit_tier_d_manifest(args.manifest, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    return 0 if report["gate_tier_d_manifest_pass"] else 2


def _cmd_audit_observer(args: argparse.Namespace) -> int:
    from selfsight.analysis.observer_audit import audit_observer_manifest
    from selfsight.observers.client import ObserverServiceClient

    command = [
        str(args.python.resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]
    if args.backend != "mock":
        if not args.model_id or not args.revision:
            raise SystemExit("--model-id and --revision are required for a real observer")
        command.extend(["--model-id", args.model_id, "--revision", args.revision])
    with ObserverServiceClient(command, args.wire_log) as client:
        report = audit_observer_manifest(
            client,
            args.manifest,
            limit=args.limit,
            output_path=args.output,
        )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["gate_minus_1_pass"] else 2


def _cmd_finalize_gate_minus_1(args: argparse.Namespace) -> int:
    from selfsight.analysis.observer_audit import finalize_gate_minus_1

    report = finalize_gate_minus_1(
        reference_audit_path=args.reference_audit,
        showo_report_path=args.showo_report,
        candidate_report_paths=args.candidate_report,
        output_path=args.output,
        delta_max=args.delta_max,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def _cmd_figure1(args: argparse.Namespace) -> int:
    from selfsight.analysis.figure1 import render_figure1_from_csv

    outputs = render_figure1_from_csv(
        args.metrics,
        args.output,
        arm=args.arm,
        d_g=args.d_g,
        d_star=args.d_star,
        evidence_status=args.evidence_status,
    )
    print(json.dumps(outputs, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selfsight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate paths/config and capture host facts")
    doctor.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    doctor.add_argument("--output", type=Path)
    doctor.set_defaults(function=_cmd_doctor)

    data = subparsers.add_parser("build-data", help="render registered synthetic datasets")
    data.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    data.add_argument("--output", type=Path)
    data.set_defaults(function=_cmd_build_data)

    audit = subparsers.add_parser("audit-data", help="audit split leakage and reference verifier")
    audit.add_argument("root", type=Path)
    audit.add_argument("--manifest-dir", type=Path)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(function=_cmd_audit_data)

    tier_b = subparsers.add_parser("audit-tier-b", help="audit all registered pixel interventions")
    tier_b.add_argument("manifest", type=Path)
    tier_b.add_argument("--output", type=Path)
    tier_b.set_defaults(function=_cmd_audit_tier_b)

    tier_d_build = subparsers.add_parser(
        "build-tier-d", help="materialize the fixed Tier-A/B 600-image mechanism subset"
    )
    tier_d_build.add_argument("root", type=Path)
    tier_d_build.add_argument("--seed", type=int, default=20260827)
    tier_d_build.set_defaults(function=_cmd_build_tier_d)

    tier_d_audit = subparsers.add_parser(
        "audit-tier-d", help="audit Tier-D selection, RGB hashes, atoms, and pair completeness"
    )
    tier_d_audit.add_argument("manifest", type=Path)
    tier_d_audit.add_argument("--output", type=Path)
    tier_d_audit.set_defaults(function=_cmd_audit_tier_d)

    observer = subparsers.add_parser("audit-observer", help="run a blind Gate -1 observer audit")
    observer.add_argument("manifest", type=Path)
    observer.add_argument("--python", type=Path, required=True, help="Python executable for the isolated service")
    observer.add_argument(
        "--backend",
        choices=("mock", "showo", "showo_discrete", "janus", "smolvlm", "qwen2vl", "internvl"),
        required=True,
    )
    observer.add_argument("--model-id")
    observer.add_argument("--revision")
    observer.add_argument("--device", default="cuda:1")
    observer.add_argument("--limit", type=int)
    observer.add_argument("--output", type=Path, required=True)
    observer.add_argument("--wire-log", type=Path, required=True)
    observer.set_defaults(function=_cmd_audit_observer)

    finalize = subparsers.add_parser(
        "finalize-gate-minus-1", help="select the capability-matched detector and enforce stop rules"
    )
    finalize.add_argument("--reference-audit", type=Path, required=True)
    finalize.add_argument("--showo-report", type=Path, required=True)
    finalize.add_argument("--candidate-report", type=Path, action="append", required=True)
    finalize.add_argument("--delta-max", type=float, default=0.03)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(function=_cmd_finalize_gate_minus_1)

    mock = subparsers.add_parser("mock-pilot", help="run a non-scientific full pipeline smoke test")
    mock.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    mock.add_argument("--output", type=Path)
    mock.set_defaults(function=_cmd_mock_pilot)

    figure = subparsers.add_parser("figure1", help="render registered Figure 1")
    figure.add_argument("metrics", type=Path)
    figure.add_argument("output", type=Path)
    figure.add_argument("--arm", default="naive")
    figure.add_argument("--d-g", type=float)
    figure.add_argument("--d-star", type=float)
    figure.add_argument("--evidence-status", default="formal")
    figure.set_defaults(function=_cmd_figure1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
