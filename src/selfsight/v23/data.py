"""Versioned v2.3 manifests with an appearance-tolerant box vocabulary."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from selfsight.data.audit import assert_zero_split_overlap, audit_reference_manifests
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.schemas import QuestionFormat, SceneSpec, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl

PRIMARY_FAMILIES = ("existence", "color", "spatial")
HARD_FAMILIES = ("count", "binding")
DISPLAY_NAME = "box"
INTERNAL_NAME = "square"
ASPECT_RATIO_RANGE = (0.5, 2.0)


def _project_root() -> Path:
    configured = os.environ.get("SELFSIGHT_PROJECT_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[3]


def display_text(text: str) -> str:
    """Use box/boxes in visible v2.3 language without changing internal geometry codes."""

    return text.replace("squares", "boxes").replace("square", "box")


def _transform_record(source: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    source_scene = SceneSpec.from_dict(dict(source["scene"]))
    scene_id = f"v23-{source_scene.scene_id}"
    scene = replace(
        source_scene,
        scene_id=scene_id,
        split=split,
        prompt=display_text(source_scene.prompt),
        template_id=f"v23-{split}-{source_scene.template_id}",
        signature=sha256_json(["v2.3", source_scene.signature, DISPLAY_NAME]),
        metadata={
            **source_scene.metadata,
            "v23_source_scene_id": source_scene.scene_id,
            "shape_display_alias": {INTERNAL_NAME: DISPLAY_NAME},
            "quadrilateral_aspect_ratio_range": list(ASPECT_RATIO_RANGE),
        },
    )
    atom = build_primary_atom(scene)
    questions = [
        replace(build_question(atom, QuestionFormat.OPEN), text=display_text(build_question(atom).text)),
        replace(
            build_question(atom, QuestionFormat.FORCED_CHOICE, 0),
            text=display_text(build_question(atom, QuestionFormat.FORCED_CHOICE, 0).text),
        ),
        replace(
            build_question(atom, QuestionFormat.FORCED_CHOICE, 1),
            text=display_text(build_question(atom, QuestionFormat.FORCED_CHOICE, 1).text),
        ),
    ]
    image = Path(str(source["reference_image"]))
    if not image.is_file():
        original_split = source_scene.split
        image = (
            _project_root()
            / "data"
            / "selfsight-v1"
            / "reference_images"
            / original_split
            / f"{source_scene.scene_id}.png"
        ).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"v2.3 source reference is missing: {image}")
    if rgb_sha256(image) != str(source["reference_rgb_sha256"]):
        raise RuntimeError(f"v2.3 source RGB changed: {source_scene.scene_id}")
    return {
        "schema_version": 3,
        "benchmark_version": "2.3",
        "source_manifest_sha256": None,
        "source_scene_id": source_scene.scene_id,
        "scene": as_serializable(scene),
        "atom": as_serializable(atom),
        "questions": [as_serializable(question) for question in questions],
        "reference_image": str(image),
        "reference_file_sha256": sha256_file(image),
        "reference_rgb_sha256": rgb_sha256(image),
        "vocabulary": {
            "internal_shape": INTERNAL_NAME,
            "display_shape": DISPLAY_NAME,
            "quadrilateral_aspect_ratio_min": ASPECT_RATIO_RANGE[0],
            "quadrilateral_aspect_ratio_max": ASPECT_RATIO_RANGE[1],
        },
    }


def _select(
    source_path: Path,
    *,
    families: Sequence[str],
    size: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    source_hash = sha256_file(source_path)
    candidates = [
        record for record in read_jsonl(source_path) if record["scene"]["family"] in set(families)
    ]
    selected = stable_stratified_sample(
        candidates,
        size,
        stratum=lambda record: str(record["scene"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=seed,
    )
    output = []
    for record in selected:
        transformed = _transform_record(record, split=split)
        transformed["source_manifest_sha256"] = source_hash
        output.append(transformed)
    return output


def materialize_v23_data(
    *,
    source_root: str | Path,
    output_root: str | Path,
    seed: int = 20260829,
    train_size: int = 432,
    probe_size: int = 96,
    outcome_size: int = 90,
    hard_size: int = 60,
) -> dict[str, Any]:
    """Derive new manifests without modifying source rows or RGBs."""

    project_root = _project_root()
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    expected = (project_root / "data" / "selfsight-v2.3").resolve()
    if output != expected:
        raise RuntimeError(f"v2.3 data must be materialized at {expected}, got {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v2.3 data: {output}")
    manifests = output / "manifests"
    transformed = {
        "train": _select(
            source / "manifests" / "train.jsonl",
            families=PRIMARY_FAMILIES,
            size=train_size,
            seed=seed,
            split="v23_train",
        ),
        "gradient_probe": _select(
            source / "manifests" / "tier_a_probe.jsonl",
            families=PRIMARY_FAMILIES,
            size=probe_size,
            seed=seed,
            split="v23_gradient_probe",
        ),
        "outcome": _select(
            source / "manifests" / "tier_a_outcome.jsonl",
            families=PRIMARY_FAMILIES,
            size=outcome_size,
            seed=seed,
            split="v23_outcome",
        ),
        "hard_outcome": _select(
            source / "manifests" / "tier_a_outcome.jsonl",
            families=HARD_FAMILIES,
            size=hard_size,
            seed=seed,
            split="v23_hard_outcome",
        ),
    }
    paths: dict[str, Path] = {}
    for name, rows in transformed.items():
        paths[name] = atomic_write_jsonl(manifests / f"{name}.jsonl", rows)
    primary_overlap = assert_zero_split_overlap(
        {name: paths[name] for name in ("train", "gradient_probe", "outcome")}
    )
    reference_audit = audit_reference_manifests(
        {name: path for name, path in paths.items()}, output / "reference_audit.json"
    )
    counts = {name: len(rows) for name, rows in transformed.items()}
    expected_counts = {
        "train": train_size,
        "gradient_probe": probe_size,
        "outcome": outcome_size,
        "hard_outcome": hard_size,
    }
    if counts != expected_counts or not reference_audit["gate_reference_pass"]:
        raise RuntimeError("v2.3 data materialization failed count/reference checks")
    registry = {
        "schema_version": 3,
        "benchmark_version": "2.3",
        "data_namespace": "selfsight-v2.3",
        "seed": seed,
        "primary_families": list(PRIMARY_FAMILIES),
        "hard_families": list(HARD_FAMILIES),
        "vocabulary": {
            "internal_shape": INTERNAL_NAME,
            "display_shape": DISPLAY_NAME,
            "quadrilateral_aspect_ratio_range": list(ASPECT_RATIO_RANGE),
        },
        "counts": counts,
        "manifests": {name: str(path.resolve()) for name, path in paths.items()},
        "manifest_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "source_root": str(source),
        "source_registry_sha256": sha256_file(source / "manifests" / "registry.json"),
        "split_audit": primary_overlap,
        "reference_audit": {
            "path": str((output / "reference_audit.json").resolve()),
            "sha256": sha256_file(output / "reference_audit.json"),
        },
    }
    registry["registry_digest"] = sha256_json(registry)
    atomic_write_json(manifests / "registry.json", registry)
    return registry
