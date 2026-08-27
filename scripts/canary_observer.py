"""Pinned observer load, RGB-only accuracy, and repeatability canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from selfsight.data.subsets import stable_stratified_sample
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import hard_render, make_blind_request
from selfsight.schemas import AtomicQuestion
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("showo", "showo_discrete", "janus", "smolvlm", "qwen2vl", "internvl"),
        required=True,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready_report = output / "service_ready.json"
    command = [
        str(args.python.resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        args.backend,
        "--model-id",
        args.model_id,
        "--revision",
        args.revision,
        "--device",
        args.device,
        "--ready-report",
        str(ready_report),
    ]
    records = stable_stratified_sample(
        list(read_jsonl(args.manifest)),
        args.limit,
        stratum=lambda record: str(record["atom"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=20260827,
    )
    rows = []
    with ObserverServiceClient(command, output / "observer_wire.jsonl") as client:
        for index, record in enumerate(records):
            question = AtomicQuestion.from_dict(record["questions"][0])
            image_path = output / "hard_rgb" / f"{index:04d}.png"
            with Image.open(record["reference_image"]) as source:
                hard_render(source.convert("RGB"), image_path)
            answers = []
            for repeat in range(2):
                result = client.observe(
                    make_blind_request(image_path, (question,), f"canary-{index:04d}-r{repeat}")
                )
                answers.append(result.answers[0].normalized_answer)
            rows.append(
                {
                    "scene_id": record["scene"]["scene_id"],
                    "family": question.family.value,
                    "expected": question.expected_answer,
                    "answers": answers,
                    "repeat_agreement": answers[0] == answers[1],
                    "correct_first": answers[0] == question.expected_answer,
                }
            )
    if not rows:
        raise RuntimeError("Observer canary received no manifest rows")
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "model_id": args.model_id,
        "revision": args.revision,
        "device": args.device,
        "samples": len(rows),
        "repeat_consistency": sum(row["repeat_agreement"] for row in rows) / len(rows),
        "accuracy": sum(row["correct_first"] for row in rows) / len(rows),
        "repeatability_gate_pass": (
            sum(row["repeat_agreement"] for row in rows) / len(rows) >= 0.90
        ),
        "service_ready_report": str(ready_report),
        "rows": rows,
    }
    atomic_write_json(output / "canary_report.json", report)
    write_host_manifest(output / "host_manifest.json")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
