"""Materialize v3.0 Tier-A scene records at a chosen difficulty.

Gate A measured Show-o2-1.5B-HQ at per-candidate verifier accuracy 0.949 on the
two-object color scenes: almost never wrong, so a bounded bank cannot supply two
verifier-incorrect candidates. This regenerates the primary-family scenes with a
larger object population to move accuracy into a measurable regime.

The difficulty setting is calibrated on a dedicated `calibration` split and only
then frozen for the held-out probe/train/outcome splits, so the Gate A
measurement is never taken on the split used to choose the difficulty.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from selfsight.data.generator import generate_split, with_scene_id
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.schemas import QuestionFamily, QuestionFormat, as_serializable
from selfsight.utils.hashing import sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl
from selfsight.v3.vocabulary import (
    DISPLAY_NAME,
    INTERNAL_NAME,
    assert_display_vocabulary,
    display_text,
    scene_vocabulary_metadata,
)

PRIMARY = (QuestionFamily.EXISTENCE, QuestionFamily.COLOR, QuestionFamily.SPATIAL)
SPLIT_SEEDS = {"calibration": 0, "tier_a_probe": 1, "train": 2, "tier_a_outcome": 3}


def build(split, total, seed, objects_per_scene, forbidden, families=PRIMARY):
    # Calibration borrows the probe templates so the difficulty it measures
    # transfers, but keeps its own scene ids and is excluded from every later split.
    template_split = "tier_a_probe" if split == "calibration" else split
    scenes = generate_split(
        split=template_split,
        total=total,
        seed=seed + SPLIT_SEEDS[split] * 1_000_003,
        families=families,
        forbidden_signatures=forbidden,
        objects_per_scene=objects_per_scene,
    )
    if split != template_split:
        scenes = [
            with_scene_id(scene, scene.scene_id.replace(template_split, split, 1), split=split)
            for scene in scenes
        ]
    rows = []
    for scene in scenes:
        # The primary atom is built from the untransformed scene so that the
        # verifier keeps looking up the internal geometry code; only text a
        # model or a reviewer can read is rewritten.
        atom = build_primary_atom(scene)
        question = build_question(atom, question_format=QuestionFormat.OPEN)
        displayed = replace(
            scene,
            prompt=display_text(scene.prompt),
            metadata={**scene.metadata, **scene_vocabulary_metadata()},
        )
        rows.append(
            {
                "benchmark_version": "3.0",
                "schema_version": 3,
                "scene": as_serializable(displayed),
                "atom": as_serializable(atom),
                "questions": [
                    as_serializable(replace(question, text=display_text(question.text))),
                ],
                "objects_per_scene": objects_per_scene,
                "vocabulary": {f"{INTERNAL_NAME}_display_name": DISPLAY_NAME},
            }
        )
    # Fail closed: a row that claims the display vocabulary while its visible
    # text still says "square" is the defect that shipped in the first probe
    # manifest, and it is invisible to every downstream consumer.
    assert_display_vocabulary(rows)
    return scenes, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/selfsight-v3")
    parser.add_argument("--objects-per-scene", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--family", action="append", default=None,
        help="Restrict to these families (default: all primary families)",
    )
    parser.add_argument(
        "--split", action="append", required=True, metavar="NAME=TOTAL",
        help="e.g. --split calibration=24 --split tier_a_probe=96",
    )
    args = parser.parse_args()

    families = PRIMARY
    if args.family:
        wanted = {value.lower() for value in args.family}
        families = tuple(f for f in PRIMARY if f.value in wanted)
        if len(families) != len(wanted):
            raise SystemExit(f"Unknown family in {sorted(wanted)}")

    root = Path(args.output_root).resolve()
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    forbidden: set[str] = set()
    registry: dict[str, object] = {
        "benchmark_version": "3.0",
        "objects_per_scene": args.objects_per_scene,
        "seed": args.seed,
        "primary_families": [family.value for family in families],
        "splits": {},
    }
    for entry in args.split:
        name, _, total = entry.partition("=")
        if name not in SPLIT_SEEDS:
            raise SystemExit(f"Unknown split {name}; expected one of {sorted(SPLIT_SEEDS)}")
        scenes, rows = build(
            name, int(total), args.seed, args.objects_per_scene, forbidden, families
        )
        overlap = forbidden & {scene.signature for scene in scenes}
        if overlap:
            raise SystemExit(f"Split {name} reuses {len(overlap)} scene signatures")
        forbidden.update(scene.signature for scene in scenes)
        path = manifests / f"{name}.jsonl"
        atomic_write_jsonl(path, rows)
        registry["splits"][name] = {
            "path": str(path),
            "rows": len(rows),
            "signature_digest": sha256_json(sorted(scene.signature for scene in scenes)),
        }
        print(f"{name:<16} {len(rows):>4} rows -> {path}")

    registry["zero_overlap_verified"] = True
    atomic_write_json(manifests / "registry.json", registry)
    print(json.dumps(registry["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
