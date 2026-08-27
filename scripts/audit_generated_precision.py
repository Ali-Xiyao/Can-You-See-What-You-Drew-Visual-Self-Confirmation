"""Generate, export, and score Gate -2C precision-first readiness evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml

from selfsight.analysis.readiness_audit import summarize_generated_rows
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.data.generated_verifier import verify_generated_image
from selfsight.data.readiness_precision import (
    export_generated_precision_audit,
    score_generated_precision_audit,
)
from selfsight.schemas import Atom, AtomicQuestion, as_serializable
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _generate(args: argparse.Namespace) -> dict:
    backbone = yaml.safe_load(args.backbone_config.read_text(encoding="utf-8"))
    readiness = yaml.safe_load(args.readiness_config.read_text(encoding="utf-8"))
    reference = _read_json(args.reference_report)
    model_id = str(backbone["backbone_id"])
    revision = str(backbone["revision"])
    if reference.get("model_id") != model_id or reference.get("revision") != revision:
        raise RuntimeError("A2 reference report does not match the selected backbone")
    retained = tuple(str(item) for item in reference.get("passing_families", ()))
    if len(retained) < int(readiness["thresholds"]["reference"]["families_passing_min"]):
        raise RuntimeError("A2 retained fewer than four families; A3 generation is forbidden")
    records = list(read_jsonl(args.manifest))
    expected = len(readiness["main_families"]) * int(
        readiness["sampling"]["generated_prompts_per_family"]
    )
    if len(records) != expected:
        raise RuntimeError(f"A3 requires {expected} generated prompts, got {len(records)}")

    adapter = Showo2Adapter(
        backbone_config=args.backbone_config,
        device=str(backbone["hardware"]["generator_device"]),
        lazy=False,
    )
    seeds = tuple(int(item) for item in readiness["sampling"]["fixed_candidate_seeds"])
    if len(seeds) != int(readiness["sampling"]["k_oracle"]):
        raise RuntimeError("Configured fixed seeds do not match K-oracle")
    rows = []
    started = perf_counter()
    for record in records:
        scene = record["scene"]
        family = str(scene["family"])
        atom = Atom.from_dict(record["atom"])
        AtomicQuestion.from_dict(record["questions"][0])
        candidate_seeds = seeds if family in retained else seeds[:1]
        for candidate_index, seed in enumerate(candidate_seeds):
            candidate = adapter.generate_images(
                [str(scene["prompt"])],
                [seed],
                args.output.parent / "a3-images",
                "gate-minus-2c",
            )[0]
            result = verify_generated_image(candidate.image_path, (atom,))
            answer = result.answers[atom.atom_id]
            rows.append(
                {
                    "schema_version": 2,
                    "scene_id": scene["scene_id"],
                    "family": family,
                    "prompt": scene["prompt"],
                    "primary_atom": record["atom"],
                    "primary_question": record["questions"][0],
                    "expected_answer": atom.answer,
                    "candidate_id": candidate.candidate_id,
                    "candidate_index": candidate_index,
                    "sampling_seed": seed,
                    "image_path": candidate.image_path,
                    "rgb_sha256": candidate.rgb_sha256,
                    "primary_answer": answer,
                    "primary_answer_covered": answer is not None,
                    "primary_correct": answer == atom.answer,
                    "detections": [as_serializable(item) for item in result.detections],
                    "parse_errors": list(result.parse_errors),
                }
            )
    rows_path = args.output.with_name(args.output.stem + "-rows.jsonl")
    atomic_write_jsonl(rows_path, rows)
    summary = summarize_generated_rows(
        rows,
        families=tuple(str(item) for item in readiness["main_families"]),
        oracle_families=retained,
    )
    thresholds = readiness["thresholds"]["generated"]
    checks = {
        "overall_primary_answer_coverage": summary["overall_coverage"]
        >= float(thresholds["overall_coverage_min"]),
        "retained_family_coverage": all(
            summary["family_coverage"][family] >= float(thresholds["family_coverage_min"])
            for family in retained
        ),
        "oracle_at_4": summary["overall_oracle_at_4"]
        >= float(thresholds["oracle_at_4_min"]),
        "fixed_seed_coverage_swing": summary["fixed_seed_coverage_swing_points"]
        <= float(thresholds["fixed_seed_coverage_swing_points_max"]),
    }
    return {
        "schema_version": 2,
        "stage": "minus_2c_generated_measurability",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "source_revision": adapter.identity.source_revision,
        "dependency_revisions": adapter.dependency_revisions(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "reference_report": str(args.reference_report),
        "reference_report_sha256": sha256_file(args.reference_report),
        "retained_families_from_a2": list(retained),
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "elapsed_seconds": perf_counter() - started,
        **summary,
        "checks": checks,
        "passed_without_human_precision": all(checks.values()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--backbone-config", type=Path, required=True)
    generate.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    generate.add_argument("--reference-report", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--generated-report", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--seed", type=int, default=20260828)
    score = subparsers.add_parser("score")
    score.add_argument("--review-csv", type=Path, required=True)
    score.add_argument("--answer-key", type=Path, required=True)
    score.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        for field in ("backbone_config", "readiness_config", "reference_report", "manifest"):
            setattr(args, field, getattr(args, field).resolve())
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite A3 report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = _generate(args)
        atomic_write_json(args.output, report)
    elif args.command == "export":
        report = export_generated_precision_audit(
            args.generated_report.resolve(), args.output.resolve(), seed=args.seed
        )
    else:
        readiness = yaml.safe_load(args.readiness_config.read_text(encoding="utf-8"))
        report = score_generated_precision_audit(
            args.review_csv,
            args.answer_key,
            families=[str(item) for item in readiness["main_families"]],
            threshold=float(readiness["thresholds"]["generated"]["verifier_precision_min"]),
            output=args.output,
        )
    print(json.dumps(report, indent=2))
    if args.command == "score":
        raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
