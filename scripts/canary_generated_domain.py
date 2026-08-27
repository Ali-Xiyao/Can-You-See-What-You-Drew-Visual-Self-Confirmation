"""Measure whether Show-o outputs remain parseable by the registered pixel verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from selfsight.analysis.generation_domain import (
    descriptor_quality,
    detection_descriptors,
    scene_descriptors,
    summarize_generation_rows,
)
from selfsight.config import load_config, write_config_snapshot
from selfsight.data.generated_verifier import verify_generated_image
from selfsight.data.subsets import stable_stratified_sample
from selfsight.data.verifier import verify_image
from selfsight.pilot.real_loop import _stable_seed
from selfsight.schemas import Atom, SceneSpec, as_serializable
from selfsight.showo_adapter import ShowoAdapter
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _contact_sheet(rows: list[dict[str, object]], output: Path, columns: int = 4) -> Path:
    tile = 256
    label_height = 44
    rows_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows_count * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        with Image.open(str(row["image_path"])) as opened:
            image = opened.convert("RGB").resize((tile, tile))
        canvas.paste(image, (x, y))
        label = (
            f"{row['family']} | correct={int(bool(row['primary_correct']))} | "
            f"obj={int(row['matched_objects'])}/{int(row['expected_objects'])}"
        )
        draw.text((x + 4, y + tile + 4), label, fill="black")
        draw.text((x + 4, y + tile + 22), str(row["scene_id"]), fill="black")
    path = output / "contact_sheet.png"
    canvas.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--verifier",
        choices=("strict_palette", "generated_cv_v2"),
        default="strict_palette",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_config_snapshot(config, output / "resolved_config.json")
    source = list(read_jsonl(args.manifest))
    records = stable_stratified_sample(
        source,
        args.limit,
        stratum=lambda record: str(record["atom"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=int(config.values["seed"]),
    )
    timesteps = (
        args.timesteps
        if args.timesteps is not None
        else int(config.values["model"]["generation_timesteps"])
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(config.values["model"]["guidance_scale"])
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(config.values["model"]["temperature"])
    )
    adapter = ShowoAdapter(
        device=str(config.values["hardware"]["generator_device"]),
        trainable=False,
        load_vision_tower=False,
        generation_timesteps=timesteps,
        guidance_scale=guidance_scale,
        temperature=temperature,
    )

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        scene = SceneSpec.from_dict(record["scene"])
        atom = Atom.from_dict(record["atom"])
        seed = _stable_seed(config.values["seed"], "generated-domain", scene.scene_id, index)
        candidate = adapter.generate_images(
            [scene.prompt], [seed], output / "images", "base-generated-domain"
        )[0]
        verifier = verify_image if args.verifier == "strict_palette" else verify_generated_image
        verification = verifier(candidate.image_path, [atom])
        primary_answer = verification.answers[atom.atom_id]
        quality = descriptor_quality(
            scene_descriptors(scene.objects), detection_descriptors(verification.detections)
        )
        rows.append(
            {
                "scene_id": scene.scene_id,
                "family": scene.family.value,
                "prompt": scene.prompt,
                "sampling_seed": seed,
                "image_path": candidate.image_path,
                "rgb_sha256": candidate.rgb_sha256,
                "primary_atom": as_serializable(atom),
                "primary_answer": primary_answer,
                "primary_answer_covered": primary_answer is not None,
                "primary_correct": primary_answer == atom.answer,
                "detections": [as_serializable(item) for item in verification.detections],
                **quality,
            }
        )

    threshold = float(config.values["gates"]["verifier_coverage_min"])
    summary = summarize_generation_rows(rows, parseability_min=threshold)
    contact_sheet = _contact_sheet(rows, output)
    report = {
        "schema_version": 1,
        "status": "engineering_generated_domain_canary_not_scientific_evidence",
        "source_split": "train",
        "held_out_prompts_used": False,
        "manifest": str(args.manifest.resolve()),
        "generator_id": adapter.model_id,
        "generator_revision": adapter.revision,
        "generation_timesteps": timesteps,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
        "verifier": args.verifier,
        "sample_count": len(rows),
        "contact_sheet": str(contact_sheet.resolve()),
        **summary,
    }
    atomic_write_jsonl(output / "generated_domain_trials.jsonl", rows)
    atomic_write_json(output / "generated_domain_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
