"""Manifest creation for registered synthetic splits and rendered references."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from selfsight.data.counterfactuals import CounterfactualPair, build_tier_b
from selfsight.data.generator import build_splits
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.renderer import render_scene
from selfsight.data.subsets import stable_stratified_sample
from selfsight.schemas import QuestionFormat, SceneSpec, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _scene_record(scene: SceneSpec, image_path: Path) -> dict[str, object]:
    atom = build_primary_atom(scene)
    open_question = build_question(atom, QuestionFormat.OPEN)
    forced_question_a = build_question(atom, QuestionFormat.FORCED_CHOICE, 0)
    forced_question_b = build_question(atom, QuestionFormat.FORCED_CHOICE, 1)
    return {
        "schema_version": 1,
        "scene": as_serializable(scene),
        "atom": as_serializable(atom),
        "questions": [
            as_serializable(open_question),
            as_serializable(forced_question_a),
            as_serializable(forced_question_b),
        ],
        "reference_image": str(image_path.resolve()),
        "reference_file_sha256": sha256_file(image_path),
        "reference_rgb_sha256": rgb_sha256(image_path),
    }


def render_split_manifest(scenes: Iterable[SceneSpec], root: str | Path, split: str) -> Path:
    root = Path(root)
    image_root = root / "reference_images" / split
    records = []
    for scene in scenes:
        image_path = image_root / f"{scene.scene_id}.png"
        render_scene(scene, image_path)
        records.append(_scene_record(scene, image_path))
    manifest_path = root / "manifests" / f"{split}.jsonl"
    atomic_write_jsonl(manifest_path, records)
    return manifest_path


def render_tier_b_manifest(pairs: Iterable[CounterfactualPair], root: str | Path) -> Path:
    root = Path(root)
    image_root = root / "reference_images" / "tier_b"
    records = []
    for pair in pairs:
        source_path = image_root / f"{pair.source.scene_id}.png"
        changed_path = image_root / f"{pair.counterfactual.scene_id}.png"
        render_scene(pair.source, source_path)
        render_scene(pair.counterfactual, changed_path)
        records.append(
            {
                "schema_version": 1,
                "pair": as_serializable(pair),
                "source_image": str(source_path.resolve()),
                "counterfactual_image": str(changed_path.resolve()),
                "source_rgb_sha256": rgb_sha256(source_path),
                "counterfactual_rgb_sha256": rgb_sha256(changed_path),
            }
        )
    path = root / "manifests" / "tier_b.jsonl"
    atomic_write_jsonl(path, records)
    return path


def select_tier_d_sources(
    tier_a_records: list[dict[str, Any]],
    tier_b_records: list[dict[str, Any]],
    *,
    seed: int = 20260827,
    tier_a_images: int = 300,
    tier_b_pairs: int = 150,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the registered 300 Tier-A images and 150 complete Tier-B pairs."""

    selected_a = stable_stratified_sample(
        tier_a_records,
        tier_a_images,
        stratum=lambda record: str(record["scene"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=seed + 40_000_121,
    )
    selected_b = stable_stratified_sample(
        tier_b_records,
        tier_b_pairs,
        stratum=lambda record: str(record["pair"]["category"]),
        item_id=lambda record: str(record["pair"]["pair_id"]),
        seed=seed + 50_000_123,
    )
    return selected_a, selected_b


def _tier_d_records(
    tier_a_records: list[dict[str, Any]],
    tier_b_records: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    selected_a, selected_b = select_tier_d_sources(
        tier_a_records,
        tier_b_records,
        seed=seed,
    )
    records: list[dict[str, Any]] = []
    for source in selected_a:
        scene_id = str(source["scene"]["scene_id"])
        records.append(
            {
                **source,
                "tier_d_id": f"tier_d-a-{scene_id}",
                "source_tier": "tier_a",
                "source_record_id": scene_id,
                "image_role": "reference",
                "tier_d_stratum": str(source["scene"]["family"]),
            }
        )
    for source in selected_b:
        pair = source["pair"]
        pair_id = str(pair["pair_id"])
        category = str(pair["category"])
        for role, scene_key, image_key in (
            ("source", "source", "source_image"),
            ("counterfactual", "counterfactual", "counterfactual_image"),
        ):
            scene = SceneSpec.from_dict(pair[scene_key])
            record = _scene_record(scene, Path(source[image_key]))
            record.update(
                {
                    "tier_d_id": f"tier_d-b-{pair_id}-{role}",
                    "source_tier": "tier_b",
                    "source_record_id": pair_id,
                    "image_role": role,
                    "tier_d_stratum": category,
                    "tier_b_category": category,
                    "paired_tier_d_id": f"tier_d-b-{pair_id}-"
                    + ("counterfactual" if role == "source" else "source"),
                }
            )
            records.append(record)
    if len(records) != 600:
        raise AssertionError(f"Tier D must contain 600 images, got {len(records)}")
    return records


def materialize_tier_d(root: str | Path, seed: int = 20260827) -> dict[str, Any]:
    """Create or verify the immutable Tier-D subset and register its selection digest."""

    root = Path(root)
    tier_a_path = root / "manifests" / "tier_a_outcome.jsonl"
    tier_b_path = root / "manifests" / "tier_b.jsonl"
    records = _tier_d_records(
        list(read_jsonl(tier_a_path)),
        list(read_jsonl(tier_b_path)),
        seed=seed,
    )
    path = root / "manifests" / "tier_d.jsonl"
    if path.exists():
        existing = list(read_jsonl(path))
        if sha256_json(existing) != sha256_json(records):
            raise RuntimeError("Existing Tier-D manifest does not match the registered selection")
    else:
        atomic_write_jsonl(path, records)
    composition: dict[str, int] = {}
    for record in records:
        key = f"{record['source_tier']}:{record['tier_d_stratum']}:{record['image_role']}"
        composition[key] = composition.get(key, 0) + 1
    selection = {
        "schema_version": 1,
        "seed": seed,
        "images": len(records),
        "tier_a_images": sum(record["source_tier"] == "tier_a" for record in records),
        "tier_b_images": sum(record["source_tier"] == "tier_b" for record in records),
        "selection_digest": sha256_json([record["tier_d_id"] for record in records]),
        "composition": composition,
        "manifest": str(path.resolve()),
    }
    atomic_write_json(root / "manifests" / "tier_d_selection.json", selection)
    registry_path = root / "manifests" / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["counts"]["tier_d"] = len(records)
        registry["manifests"]["tier_d"] = str(path.resolve())
        registry["tier_d_selection_digest"] = selection["selection_digest"]
        atomic_write_json(registry_path, registry)
    return selection


def create_registered_dataset(root: str | Path, seed: int = 20260827) -> dict[str, str]:
    root = Path(root)
    splits = build_splits(seed)
    outputs = {
        split: str(render_split_manifest(scenes, root, split))
        for split, scenes in splits.items()
    }
    tier_b = build_tier_b(splits["tier_a_outcome"])
    outputs["tier_b"] = str(render_tier_b_manifest(tier_b, root))
    registry = {
        "schema_version": 1,
        "seed": seed,
        "counts": {name: len(scenes) for name, scenes in splits.items()} | {"tier_b": len(tier_b)},
        "manifests": outputs,
        "split_signature_digest": sha256_json(
            {name: [scene.signature for scene in scenes] for name, scenes in splits.items()}
        ),
    }
    registry_path = root / "manifests" / "registry.json"
    atomic_write_json(registry_path, registry)
    outputs["registry"] = str(registry_path)
    tier_d = materialize_tier_d(root, seed)
    outputs["tier_d"] = str(tier_d["manifest"])
    return outputs
