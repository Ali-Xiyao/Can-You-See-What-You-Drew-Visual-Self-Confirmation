"""Versioned family-minimal benchmark for the v2.2 Joint Readiness gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.renderer import render_scene
from selfsight.schemas import (
    Color,
    QuestionFamily,
    QuestionFormat,
    SceneObject,
    SceneSpec,
    Shape,
    Size,
    as_serializable,
)
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl

BENCHMARK_VERSION = "2.2"
DATA_NAMESPACE = "selfsight-v2.2"
MAIN_FAMILIES = tuple(QuestionFamily)
SPLIT_COUNTS = {"canary": 1, "reference": 20, "generated": 10}
_SPLIT_OFFSETS = {"reference": 0, "generated": 20, "canary": 30}
_SHAPES = tuple(Shape)
_COLORS = tuple(Color)
_POSITIONS = (
    (128, 128),
    (256, 128),
    (384, 128),
    (128, 256),
    (256, 256),
    (384, 256),
    (128, 384),
    (256, 384),
    (384, 384),
)


def _object(
    object_id: str,
    shape: Shape,
    color: Color,
    *,
    size: Size = Size.LARGE,
    center: tuple[int, int] = (256, 256),
) -> SceneObject:
    return SceneObject(object_id, shape, color, size, center)


def _split_prompt(split: str, generation_prompt: str, reference_description: str) -> str:
    if split == "generated":
        return generation_prompt
    if split == "canary":
        return generation_prompt.replace("Draw", "Create an image showing", 1)
    return f"Program reference only: {reference_description}"


def _existence_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    positive = case % 2 == 0
    shape = _SHAPES[(case // 2) % len(_SHAPES)]
    color = _COLORS[(case // 6) % len(_COLORS)]
    center = _POSITIONS[(case // 24) % len(_POSITIONS)]
    shown = _object("obj0", shape, color, center=center)
    if positive:
        target_shape, target_color = shape, color
    else:
        target_shape = _SHAPES[(_SHAPES.index(shape) + 1) % len(_SHAPES)]
        target_color = _COLORS[(_COLORS.index(color) + 1) % len(_COLORS)]
    prompt = _split_prompt(
        split,
        f"Draw one {color.value} {shape.value} on a plain white background.",
        f"one {color.value} {shape.value} on white",
    )
    metadata = {
        "target_shape": target_shape.value,
        "target_color": target_color.value,
        "positive": positive,
        "minimal_constraints": ("shape", "color"),
    }
    return prompt, (shown,), metadata


def _color_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    shape = _SHAPES[case % len(_SHAPES)]
    color = _COLORS[(case // 3) % len(_COLORS)]
    center = _POSITIONS[(case // 12) % len(_POSITIONS)]
    shown = _object("obj0", shape, color, center=center)
    prompt = _split_prompt(
        split,
        f"Draw one {shape.value} in {color.value} on a plain white background.",
        f"one {color.value} {shape.value} on white",
    )
    return prompt, (shown,), {
        "target_shape": shape.value,
        "target_object_id": shown.object_id,
        "minimal_constraints": ("shape", "color"),
    }


def _size_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    shape = _SHAPES[case % len(_SHAPES)]
    size = (Size.SMALL, Size.LARGE)[(case // 3) % 2]
    center = _POSITIONS[(case // 6) % len(_POSITIONS)]
    shown = _object("obj0", shape, Color.BLUE, size=size, center=center)
    prompt = _split_prompt(
        split,
        f"Draw one {size.value} {shape.value} on a plain white background.",
        f"one {size.value} {shape.value} on white",
    )
    return prompt, (shown,), {
        "target_shape": shape.value,
        "target_object_id": shown.object_id,
        "minimal_constraints": ("shape", "absolute_size"),
    }


def _spatial_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    first_shape = _SHAPES[case % len(_SHAPES)]
    second_shape = _SHAPES[(_SHAPES.index(first_shape) + 1 + (case // 3) % 2) % len(_SHAPES)]
    first_color = _COLORS[(case // 6) % len(_COLORS)]
    second_color = _COLORS[(_COLORS.index(first_color) + 1) % len(_COLORS)]
    relation = ("left_of", "above")[(case // 24) % 2]
    if relation == "left_of":
        first_center, second_center = (150, 256), (362, 256)
        phrase = "to the left of"
    else:
        first_center, second_center = (256, 150), (256, 362)
        phrase = "above"
    first = _object("obj0", first_shape, first_color, center=first_center)
    second = _object("obj1", second_shape, second_color, center=second_center)
    truth = case % 2 == 0
    subject, target = (first, second) if truth else (second, first)
    prompt = _split_prompt(
        split,
        (
            f"Draw a {first_color.value} {first_shape.value} {phrase} a "
            f"{second_color.value} {second_shape.value} on a plain white background."
        ),
        (
            f"a {first_color.value} {first_shape.value} {phrase} a "
            f"{second_color.value} {second_shape.value} on white"
        ),
    )
    return prompt, (first, second), {
        "subject_shape": subject.shape.value,
        "object_shape": target.shape.value,
        "relation": relation,
        "truth": truth,
        "minimal_constraints": ("shape", "color", "spatial_relation"),
    }


def _count_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    count = case % 4 + 1
    shape = _SHAPES[(case // 4) % len(_SHAPES)]
    color = _COLORS[(case // 12) % len(_COLORS)]
    centers = ((128, 256), (256, 256), (384, 256), (256, 384))
    objects = tuple(
        _object(f"obj{index}", shape, color, size=Size.SMALL, center=centers[index])
        for index in range(count)
    )
    plural = f"{shape.value}s"
    prompt = _split_prompt(
        split,
        (
            f"Draw exactly {count} {color.value} {plural}, separated from each other, "
            "on a plain white background."
        ),
        f"exactly {count} separated {color.value} {plural} on white",
    )
    return prompt, objects, {
        "target_shape": shape.value,
        "count": count,
        "minimal_constraints": ("shape", "color", "count", "separated"),
    }


def _binding_case(split: str, case: int) -> tuple[str, tuple[SceneObject, ...], dict[str, Any]]:
    first_shape = _SHAPES[case % len(_SHAPES)]
    second_shape = _SHAPES[(_SHAPES.index(first_shape) + 1 + (case // 3) % 2) % len(_SHAPES)]
    first_color = _COLORS[(case // 6) % len(_COLORS)]
    second_color = _COLORS[(_COLORS.index(first_color) + 1 + (case // 24) % 2) % len(_COLORS)]
    first = _object("obj0", first_shape, first_color, center=(160, 256))
    second = _object("obj1", second_shape, second_color, center=(352, 256))
    target = (first, second)[case % 2]
    prompt = _split_prompt(
        split,
        (
            f"Draw a {first_color.value} {first_shape.value} and a {second_color.value} "
            f"{second_shape.value}, separated from each other, on a plain white background."
        ),
        (
            f"a separated {first_color.value} {first_shape.value} and "
            f"{second_color.value} {second_shape.value} on white"
        ),
    )
    return prompt, (first, second), {
        "target_shape": target.shape.value,
        "target_object_id": target.object_id,
        "minimal_constraints": ("shape", "color_binding", "separated"),
    }


_BUILDERS: dict[
    QuestionFamily,
    Callable[[str, int], tuple[str, tuple[SceneObject, ...], dict[str, Any]]],
] = {
    QuestionFamily.EXISTENCE: _existence_case,
    QuestionFamily.COUNT: _count_case,
    QuestionFamily.COLOR: _color_case,
    QuestionFamily.SIZE: _size_case,
    QuestionFamily.SPATIAL: _spatial_case,
    QuestionFamily.BINDING: _binding_case,
}


def build_minimal_scenes(split: str, *, per_family: int, seed: int = 20260828) -> list[SceneSpec]:
    """Build deterministic minimal scenes without calling the compound v1 generator."""

    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"Unknown readiness split: {split}")
    if per_family < 1 or per_family > SPLIT_COUNTS[split]:
        raise ValueError(f"{split} supports 1..{SPLIT_COUNTS[split]} scenes per family")
    scenes = []
    for family in MAIN_FAMILIES:
        for index in range(per_family):
            case = _SPLIT_OFFSETS[split] + index
            prompt, objects, metadata = _BUILDERS[family](split, case)
            scene_id = f"v2p2-{split}-{family.value}-{index:03d}"
            signature_payload = {
                "family": family.value,
                "objects": [as_serializable(item) for item in objects],
                "metadata": metadata,
            }
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    split=f"readiness_{split}",
                    family=family,
                    prompt=prompt,
                    objects=objects,
                    canvas_size=512,
                    generator_seed=seed,
                    template_id=f"v2p2-{split}-{family.value}-minimal-v1",
                    signature=sha256_json(signature_payload),
                    metadata={
                        **metadata,
                        "benchmark_version": BENCHMARK_VERSION,
                        "data_namespace": DATA_NAMESPACE,
                        "case_number": case,
                    },
                )
            )
    return scenes


def _readiness_record(scene: SceneSpec, image_path: Path) -> dict[str, Any]:
    atom = build_primary_atom(scene)
    questions = (
        build_question(atom, QuestionFormat.OPEN),
        build_question(atom, QuestionFormat.FORCED_CHOICE, 0),
        build_question(atom, QuestionFormat.FORCED_CHOICE, 1),
    )
    return {
        "schema_version": 2,
        "benchmark_version": BENCHMARK_VERSION,
        "data_namespace": DATA_NAMESPACE,
        "scene": as_serializable(scene),
        "atom": as_serializable(atom),
        "questions": [as_serializable(question) for question in questions],
        "reference_image": str(image_path.resolve()),
        "reference_file_sha256": sha256_file(image_path),
        "reference_rgb_sha256": rgb_sha256(image_path),
    }


def _materialize_scene(scene: SceneSpec, image_path: Path) -> None:
    expected = render_scene(scene)
    expected_rgb = rgb_sha256(expected)
    if image_path.exists():
        if rgb_sha256(image_path) != expected_rgb:
            raise RuntimeError(f"Existing v2.2 RGB disagrees with registered scene: {image_path}")
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    expected.save(image_path, format="PNG", optimize=False)


def materialize_readiness_dataset(root: str | Path, *, seed: int = 20260828) -> dict[str, Any]:
    """Create the isolated A1/A2/A3 manifests and program references."""

    root = Path(root).resolve()
    if root.name.lower() != DATA_NAMESPACE:
        raise ValueError(f"Readiness data root must end in {DATA_NAMESPACE!r}: {root}")
    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    signature_sets: dict[str, set[str]] = {}
    for split, per_family in SPLIT_COUNTS.items():
        scenes = build_minimal_scenes(split, per_family=per_family, seed=seed)
        signature_sets[split] = {scene.signature for scene in scenes}
        records = []
        for scene in scenes:
            image_path = root / "reference_images" / split / f"{scene.scene_id}.png"
            _materialize_scene(scene, image_path)
            records.append(_readiness_record(scene, image_path))
        manifest_path = root / "manifests" / f"{split}.jsonl"
        if manifest_path.exists():
            existing = list(read_jsonl(manifest_path))
            if sha256_json(existing) != sha256_json(records):
                raise RuntimeError(f"Existing v2.2 manifest is not reproducible: {manifest_path}")
        else:
            atomic_write_jsonl(manifest_path, records)
        manifests[split] = str(manifest_path)
        counts[split] = len(records)

    split_names = tuple(signature_sets)
    overlaps = {
        f"{left}__{right}": len(signature_sets[left].intersection(signature_sets[right]))
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"v2.2 readiness splits overlap: {overlaps}")
    registry = {
        "schema_version": 2,
        "benchmark_version": BENCHMARK_VERSION,
        "data_namespace": DATA_NAMESPACE,
        "seed": seed,
        "main_families": [family.value for family in MAIN_FAMILIES],
        "appendix_families": ["relative_size"],
        "counts": counts,
        "manifests": manifests,
        "split_signature_overlap": overlaps,
        "manifest_sha256": {name: sha256_file(path) for name, path in manifests.items()},
    }
    registry_path = root / "manifests" / "registry.json"
    if registry_path.exists():
        current = json.loads(registry_path.read_text(encoding="utf-8"))
        if current != registry:
            raise RuntimeError(f"Existing v2.2 registry is not reproducible: {registry_path}")
    else:
        atomic_write_json(registry_path, registry)
    return {**registry, "registry": str(registry_path)}
