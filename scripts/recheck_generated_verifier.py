"""Re-score fixed generated RGBs while developing a deterministic verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.analysis.generation_domain import (
    descriptor_quality,
    detection_descriptors,
    scene_descriptors,
    summarize_generation_rows,
)
from selfsight.data.generated_verifier import verify_generated_image
from selfsight.data.verifier import verify_image
from selfsight.schemas import Atom, SceneSpec, as_serializable
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        choices=("strict_palette", "generated_cv_v2"),
        required=True,
    )
    parser.add_argument("--parseability-min", type=float, default=0.95)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = {str(row["scene"]["scene_id"]): row for row in read_jsonl(args.manifest)}
    verifier = verify_image if args.verifier == "strict_palette" else verify_generated_image
    rows = []
    for original in read_jsonl(args.trials):
        scene_id = str(original["scene_id"])
        record = source[scene_id]
        scene = SceneSpec.from_dict(record["scene"])
        atom = Atom.from_dict(record["atom"])
        verification = verifier(str(original["image_path"]), [atom])
        answer = verification.answers[atom.atom_id]
        quality = descriptor_quality(
            scene_descriptors(scene.objects), detection_descriptors(verification.detections)
        )
        rows.append(
            {
                "scene_id": scene_id,
                "family": scene.family.value,
                "prompt": scene.prompt,
                "sampling_seed": int(original["sampling_seed"]),
                "image_path": str(Path(str(original["image_path"])).resolve()),
                "rgb_sha256": str(original["rgb_sha256"]),
                "primary_atom": as_serializable(atom),
                "primary_answer": answer,
                "primary_answer_covered": answer is not None,
                "primary_correct": answer == atom.answer,
                "detections": [as_serializable(item) for item in verification.detections],
                **quality,
            }
        )
    summary = summarize_generation_rows(rows, parseability_min=args.parseability_min)
    report = {
        "schema_version": 1,
        "status": "engineering_fixed_rgb_verifier_recheck_not_scientific_evidence",
        "source_split": "train",
        "held_out_prompts_used": False,
        "manifest": str(args.manifest.resolve()),
        "source_trials": str(args.trials.resolve()),
        "verifier": args.verifier,
        "sample_count": len(rows),
        **summary,
    }
    atomic_write_jsonl(output / "verifier_trials.jsonl", rows)
    atomic_write_json(output / "verifier_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
