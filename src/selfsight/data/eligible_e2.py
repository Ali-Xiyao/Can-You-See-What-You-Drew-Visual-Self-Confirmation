"""Decision-bound, family-restricted train/probe/outcome data for v2.2 E2."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from selfsight.analysis.readiness import require_joint_readiness
from selfsight.data.generator import generate_split
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.renderer import render_scene
from selfsight.schemas import QuestionFamily, QuestionFormat, SceneSpec, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl

E2_SPLIT_COUNTS = {"train": 2400, "tier_a_probe": 200, "tier_a_outcome": 600}


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {resolved}")
    return value


def readiness_signatures(decision: Mapping[str, Any]) -> set[str]:
    """Collect all A1/A2/A3 scene signatures excluded from phenomenon data."""

    output: set[str] = set()
    evidence = decision["evidence"]
    for label in ("canary", "reference", "generated"):
        report = _read_object(evidence[label]["path"], f"Gate -2 {label} evidence")
        manifest_path = Path(str(report["manifest"])).resolve()
        if sha256_file(manifest_path) != str(report["manifest_sha256"]):
            raise RuntimeError(f"Gate -2 {label} manifest SHA-256 mismatch")
        for row in read_jsonl(manifest_path):
            signature = str(row["scene"].get("signature", ""))
            if not signature:
                raise RuntimeError(f"Gate -2 {label} scene has no signature")
            output.add(signature)
    return output


def build_eligible_scenes(
    *,
    families: tuple[QuestionFamily, ...],
    forbidden_signatures: set[str],
    seed: int = 20260827,
    split_counts: Mapping[str, int] = E2_SPLIT_COUNTS,
) -> dict[str, list[SceneSpec]]:
    """Build isolated E2 splits with counts redistributed only across eligible families."""

    if len(set(families)) < 4:
        raise ValueError("E2 data requires at least four Gate -2 eligible families")
    used = set(forbidden_signatures)
    output: dict[str, list[SceneSpec]] = {}
    for offset, (split, total) in enumerate(split_counts.items()):
        scenes = generate_split(
            split=split,
            total=int(total),
            seed=seed + offset * 1_000_003,
            forbidden_signatures=used,
            families=families,
        )
        output[split] = scenes
        used.update(scene.signature for scene in scenes)
    return output


def _record(scene: SceneSpec, image_path: Path) -> dict[str, Any]:
    atom = build_primary_atom(scene)
    questions = (
        build_question(atom, QuestionFormat.OPEN),
        build_question(atom, QuestionFormat.FORCED_CHOICE, 0),
        build_question(atom, QuestionFormat.FORCED_CHOICE, 1),
    )
    return {
        "schema_version": 2,
        "data_namespace": "selfsight-v2.2-e2-eligible",
        "scene": as_serializable(scene),
        "atom": as_serializable(atom),
        "questions": [as_serializable(question) for question in questions],
        "reference_image": str(image_path.resolve()),
        "reference_file_sha256": sha256_file(image_path),
        "reference_rgb_sha256": rgb_sha256(image_path),
    }


def materialize_eligible_e2_dataset(
    decision_path: str | Path,
    output_root: str | Path,
    *,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Materialize the exact decision-conditioned 2400/200/600 E2 dataset."""

    decision_path = Path(decision_path).resolve()
    decision = require_joint_readiness(decision_path)
    family_values = tuple(
        dict.fromkeys(str(item) for item in decision["selected_eligible_families"])
    )
    families = tuple(QuestionFamily(value) for value in family_values)
    readiness = readiness_signatures(decision)
    splits = build_eligible_scenes(
        families=families,
        forbidden_signatures=readiness,
        seed=seed,
    )
    root = Path(output_root).resolve()
    manifests: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    split_signatures: dict[str, set[str]] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for split, scenes in splits.items():
        split_signatures[split] = {scene.signature for scene in scenes}
        family_counts[split] = dict(
            sorted(Counter(scene.family.value for scene in scenes).items())
        )
        records = []
        for scene in scenes:
            image_path = root / "reference_images" / split / f"{scene.scene_id}.png"
            rendered = render_scene(scene)
            expected_hash = rgb_sha256(rendered)
            if image_path.is_file():
                if rgb_sha256(image_path) != expected_hash:
                    raise RuntimeError(f"Existing eligible E2 RGB mismatch: {image_path}")
            else:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(image_path, format="PNG", optimize=False)
            records.append(_record(scene, image_path))
        manifest = root / "manifests" / f"{split}.jsonl"
        if manifest.is_file():
            if sha256_json(list(read_jsonl(manifest))) != sha256_json(records):
                raise RuntimeError(f"Existing eligible E2 manifest is not reproducible: {manifest}")
        else:
            atomic_write_jsonl(manifest, records)
        manifests[split] = str(manifest)
        manifest_hashes[split] = sha256_file(manifest)
    names = tuple(splits)
    overlaps = {
        f"{left}__{right}": len(split_signatures[left].intersection(split_signatures[right]))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    readiness_overlap = {
        split: len(signatures.intersection(readiness))
        for split, signatures in split_signatures.items()
    }
    if any(overlaps.values()) or any(readiness_overlap.values()):
        raise RuntimeError(
            f"Eligible E2 split isolation failed: internal={overlaps}, readiness={readiness_overlap}"
        )
    registry = {
        "schema_version": 2,
        "data_namespace": "selfsight-v2.2-e2-eligible",
        "seed": seed,
        "joint_readiness_decision": str(decision_path),
        "joint_readiness_sha256": sha256_file(decision_path),
        "model_id": decision["model_id"],
        "revision": decision["revision"],
        "eligible_families": list(family_values),
        "readiness_signature_count": len(readiness),
        "counts": {split: len(scenes) for split, scenes in splits.items()},
        "family_counts": family_counts,
        "manifests": manifests,
        "manifest_sha256": manifest_hashes,
        "split_signature_overlap": overlaps,
        "readiness_signature_overlap": readiness_overlap,
    }
    registry_path = root / "manifests" / "registry.json"
    if registry_path.is_file():
        if _read_object(registry_path, "eligible E2 registry") != registry:
            raise RuntimeError(f"Existing eligible E2 registry is not reproducible: {registry_path}")
    else:
        atomic_write_json(registry_path, registry)
    return {**registry, "registry": str(registry_path)}


__all__ = [
    "E2_SPLIT_COUNTS",
    "build_eligible_scenes",
    "materialize_eligible_e2_dataset",
    "readiness_signatures",
]
