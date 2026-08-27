"""Run Show-o2 Gate -2 A1 engineering or A2 reference-observation audits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import yaml
from PIL import Image

from selfsight.analysis.readiness_audit import summarize_reference_rows
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.schemas import AtomicQuestion
from selfsight.utils.evidence import capture_host_manifest
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _configs(args: argparse.Namespace) -> tuple[dict, dict]:
    backbone = yaml.safe_load(args.backbone_config.read_text(encoding="utf-8"))
    readiness = yaml.safe_load(args.readiness_config.read_text(encoding="utf-8"))
    return backbone, readiness


def _adapter(args: argparse.Namespace) -> Showo2Adapter:
    import torch

    backbone, _ = _configs(args)
    device = str(backbone["hardware"]["generator_device"])
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    return Showo2Adapter(
        backbone_config=args.backbone_config,
        device=device,
        lazy=False,
    )


def _run_canary(args: argparse.Namespace) -> dict:
    import torch

    backbone, readiness = _configs(args)
    records = list(read_jsonl(args.manifest))
    if len(records) != len(readiness["main_families"]):
        raise RuntimeError("A1 requires exactly one canary record per main family")
    adapter = _adapter(args)
    module_tree_path = args.output.with_name(args.output.stem + "-lora-module-tree.json")
    module_tree = asdict(adapter.discover_lora_targets())
    atomic_write_json(module_tree_path, module_tree)
    rows = []
    started = perf_counter()
    base_seed = int(readiness["sampling"]["fixed_candidate_seeds"][0])
    for index, record in enumerate(records):
        question = AtomicQuestion.from_dict(record["questions"][0])
        seed = base_seed + index
        candidate = adapter.generate_images(
            [str(record["scene"]["prompt"])],
            [seed],
            args.output.parent / "a1-images",
            "gate-minus-2a",
        )[0]
        reference = adapter.observe_atoms(record["reference_image"], (question, question))
        generated = adapter.observe_atoms(candidate.image_path, (question, question))
        with Image.open(candidate.image_path) as opened:
            generated_size = list(opened.size)
        rows.append(
            {
                "scene_id": record["scene"]["scene_id"],
                "family": record["scene"]["family"],
                "sampling_seed": seed,
                "generated_image": candidate.image_path,
                "generated_rgb_sha256": candidate.rgb_sha256,
                "generated_size": generated_size,
                "expected": question.expected_answer,
                "reference_answers": [item.normalized_answer for item in reference.answers],
                "generated_answers": [item.normalized_answer for item in generated.answers],
            }
        )
    elapsed = perf_counter() - started
    rows_path = args.output.with_name(args.output.stem + "-rows.jsonl")
    atomic_write_jsonl(rows_path, rows)
    engineering_checks = {
        "all_images_written": all(Path(row["generated_image"]).is_file() for row in rows),
        "native_resolution": all(
            row["generated_size"] == [int(backbone["official_profile"]["resolution"])] * 2
            for row in rows
        ),
        "reference_observation_completed": all(
            len(row["reference_answers"]) == 2 for row in rows
        ),
        "generated_observation_completed": all(
            len(row["generated_answers"]) == 2 for row in rows
        ),
    }
    return {
        "schema_version": 2,
        "stage": "minus_2a_inference_canary",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "source_revision": adapter.identity.source_revision,
        "dependency_revisions": adapter.dependency_revisions(),
        "native_resolution": adapter.native_resolution,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "lora_module_tree": str(module_tree_path),
        "lora_module_tree_sha256": sha256_file(module_tree_path),
        "samples": len(rows),
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(adapter.device)),
        "resource_report": asdict(adapter.resource_report()),
        "checks": engineering_checks,
        "passed": all(engineering_checks.values()),
        "host": capture_host_manifest(),
    }


def _run_reference(args: argparse.Namespace) -> dict:
    import torch

    _, readiness = _configs(args)
    records = list(read_jsonl(args.manifest))
    expected_images = len(readiness["main_families"]) * int(
        readiness["sampling"]["reference_per_family"]
    )
    if len(records) != expected_images:
        raise RuntimeError(f"A2 requires {expected_images} reference images, got {len(records)}")
    adapter = _adapter(args)
    rows = []
    started = perf_counter()
    for record in records:
        questions = tuple(AtomicQuestion.from_dict(item) for item in record["questions"])
        request_questions = (questions[0], questions[0], questions[1], questions[2])
        result = adapter.observe_atoms(record["reference_image"], request_questions)
        predictions = [answer.normalized_answer for answer in result.answers]
        rows.append(
            {
                "scene_id": record["scene"]["scene_id"],
                "family": record["scene"]["family"],
                "expected": questions[0].expected_answer,
                "open_prediction": predictions[0],
                "open_repeat_prediction": predictions[1],
                "forced_predictions": predictions[2:],
                "forced_expected": [questions[1].expected_answer, questions[2].expected_answer],
            }
        )
    elapsed = perf_counter() - started
    rows_path = args.output.with_name(args.output.stem + "-rows.jsonl")
    atomic_write_jsonl(rows_path, rows)
    summary = summarize_reference_rows(rows, readiness["thresholds"]["reference"])
    return {
        "schema_version": 2,
        "stage": "minus_2b_reference_observation",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "source_revision": adapter.identity.source_revision,
        "dependency_revisions": adapter.dependency_revisions(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(adapter.device)),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("canary", "reference"))
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.backbone_config = args.backbone_config.resolve()
    args.readiness_config = args.readiness_config.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite readiness report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = _run_canary(args) if args.stage == "canary" else _run_reference(args)
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "host"}, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
