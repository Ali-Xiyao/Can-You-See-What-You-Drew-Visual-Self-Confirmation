"""Compare preregistration-safe prompt styles on a separate development scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.config import load_config, write_config_snapshot
from selfsight.data.verifier import verify_image
from selfsight.schemas import Atom, QuestionFamily, as_serializable
from selfsight.showo_adapter import ShowoAdapter
from selfsight.utils.jsonl import atomic_write_json

PROMPTS = {
    "current_like": (
        "On an empty white field, precisely place a large blue square in the middle-right area "
        "and a small green triangle in the lower-center area."
    ),
    "flat_vector": (
        "Create a flat 2D vector diagram on a pure white square canvas. Draw only a large solid "
        "blue square in the middle-right area and a small solid green triangle in the lower-center "
        "area. No shadows, texture, perspective, border, text, or decoration."
    ),
    "geometry_worksheet": (
        "Geometry worksheet image, front view, pure white background: one large blue square at "
        "middle right and one small green triangle at lower center. Use uniform solid fills and "
        "sharp simple outlines. Nothing else; no floor, sky, lighting, shadows, or 3D rendering."
    ),
    "symbol_layout": (
        "Minimal two-symbol layout on #FFFFFF: [large BLUE SQUARE] at right-center; [small GREEN "
        "TRIANGLE] at bottom-center. Flat colored shapes only. Orthographic 2D, no text, no extra "
        "objects, no gradients, no photographic elements."
    ),
}

ATOMS = (
    Atom("dev:exists-square", QuestionFamily.EXISTENCE, "shape=square;color=blue", "exists", "yes"),
    Atom("dev:exists-triangle", QuestionFamily.EXISTENCE, "shape=triangle;color=green", "exists", "yes"),
    Atom("dev:count-square", QuestionFamily.COUNT, "shape=square", "count", "1"),
    Atom("dev:count-triangle", QuestionFamily.COUNT, "shape=triangle", "count", "1"),
    Atom("dev:color-square", QuestionFamily.COLOR, "shape=square", "color", "blue"),
    Atom("dev:color-triangle", QuestionFamily.COLOR, "shape=triangle", "color", "green"),
    Atom("dev:size-square", QuestionFamily.SIZE, "shape=square", "size", "large"),
    Atom("dev:size-triangle", QuestionFamily.SIZE, "shape=triangle", "size", "small"),
    Atom(
        "dev:above",
        QuestionFamily.SPATIAL,
        "shape=square|shape=triangle",
        "above",
        "yes",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_config_snapshot(config, output / "resolved_config.json")
    adapter = ShowoAdapter(
        device=str(config.values["hardware"]["generator_device"]),
        trainable=False,
        load_vision_tower=False,
        generation_timesteps=int(config.values["model"]["generation_timesteps"]),
        guidance_scale=float(config.values["model"]["guidance_scale"]),
        temperature=float(config.values["model"]["temperature"]),
    )
    rows = []
    for index, (style, prompt) in enumerate(PROMPTS.items()):
        seed = int(config.values["seed"]) + index
        candidate = adapter.generate_images([prompt], [seed], output / "images", style)[0]
        verification = verify_image(candidate.image_path, ATOMS)
        correct = sum(verification.answers[atom.atom_id] == atom.answer for atom in ATOMS)
        rows.append(
            {
                "style": style,
                "prompt": prompt,
                "seed": seed,
                "image_path": candidate.image_path,
                "rgb_sha256": candidate.rgb_sha256,
                "verifier_accuracy": correct / len(ATOMS),
                "verifier_coverage": verification.coverage,
                "answers": verification.answers,
                "detections": [as_serializable(item) for item in verification.detections],
            }
        )
    report = {
        "schema_version": 1,
        "status": "engineering_prompt_style_canary_not_scientific_evidence",
        "dev_scene_is_outside_registered_splits": True,
        "rows": rows,
    }
    atomic_write_json(output / "prompt_style_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
