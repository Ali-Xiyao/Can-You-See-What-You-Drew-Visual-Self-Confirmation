"""Deterministic scene-graph and English-prompt generation with split isolation."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace

from selfsight.schemas import Color, QuestionFamily, SceneObject, SceneSpec, Shape, Size
from selfsight.utils.hashing import sha256_json

FAMILIES = tuple(QuestionFamily)
COLORS = tuple(Color)
SHAPES = tuple(Shape)
SIZES = tuple(Size)
ANCHORS = (
    (112, 128),
    (256, 112),
    (400, 128),
    (112, 256),
    (400, 256),
    (112, 392),
    (256, 400),
    (400, 392),
)

TEMPLATES: dict[str, tuple[str, ...]] = {
    "train": (
        "Create a clean image containing {objects} on a white background.",
        "Draw {objects} against a plain white canvas.",
        "Render a simple scene with {objects}; use no text or decoration.",
        "Make a minimal white-background picture of {objects}.",
    ),
    "tier_a_probe": (
        "On an empty white field, precisely place {objects}.",
        "Produce a sparse geometric composition showing {objects}.",
    ),
    "tier_a_outcome": (
        "Illustrate only the following colored forms: {objects}.",
        "Generate a white-canvas arrangement consisting of {objects}.",
    ),
}


def _article(phrase: str) -> str:
    return f"{'an' if phrase[0].lower() in 'aeiou' else 'a'} {phrase}"


def _position_phrase(center: tuple[int, int]) -> str:
    x, y = center
    horizontal = "left" if x < 200 else "right" if x > 312 else "center"
    vertical = "upper" if y < 200 else "lower" if y > 312 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "in the center"
    return f"in the {vertical}-{horizontal} area"


def _describe(obj: SceneObject) -> str:
    phrase = f"{obj.size.value} {obj.color.value} {obj.shape.value} {_position_phrase(obj.center)}"
    return _article(phrase)


def _join_descriptions(objects: Iterable[SceneObject]) -> str:
    parts = [_describe(obj) for obj in objects]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _scene_signature(objects: Iterable[SceneObject]) -> str:
    canonical = sorted(
        (obj.shape.value, obj.color.value, obj.size.value, obj.center[0], obj.center[1])
        for obj in objects
    )
    return sha256_json(canonical)


def _make_objects(family: QuestionFamily, rng: random.Random) -> tuple[SceneObject, ...]:
    count = 3 if family in {QuestionFamily.COUNT, QuestionFamily.BINDING} else 2
    positions = rng.sample(ANCHORS, count)
    shapes = list(rng.sample(SHAPES, count)) if count <= len(SHAPES) else [rng.choice(SHAPES) for _ in range(count)]

    if family == QuestionFamily.COUNT:
        repeated_shape = rng.choice(SHAPES)
        repeated_count = rng.choice((1, 2, 3))
        shapes = [repeated_shape] * repeated_count
        while len(shapes) < count:
            other = rng.choice([shape for shape in SHAPES if shape != repeated_shape])
            shapes.append(other)
        rng.shuffle(shapes)

    colors = list(rng.sample(COLORS, count))
    sizes = [rng.choice(SIZES) for _ in range(count)]
    if family == QuestionFamily.SPATIAL and sizes[0] == sizes[1]:
        sizes[1] = Size.LARGE if sizes[0] == Size.SMALL else Size.SMALL
    return tuple(
        SceneObject(
            object_id=f"obj{index}",
            shape=shapes[index],
            color=colors[index],
            size=sizes[index],
            center=positions[index],
        )
        for index in range(count)
    )


def _primary_metadata(family: QuestionFamily, objects: tuple[SceneObject, ...], rng: random.Random) -> dict[str, object]:
    if family == QuestionFamily.EXISTENCE:
        positive = bool(rng.getrandbits(1))
        if positive:
            target_shape, target_color = objects[0].shape, objects[0].color
        else:
            occupied = {(item.shape, item.color) for item in objects}
            alternatives = [(shape, color) for shape in SHAPES for color in COLORS if (shape, color) not in occupied]
            target_shape, target_color = rng.choice(alternatives)
        return {"target_shape": target_shape.value, "target_color": target_color.value, "positive": positive}
    if family == QuestionFamily.COUNT:
        counts = Counter(item.shape for item in objects)
        target_shape = rng.choice(tuple(counts))
        return {"target_shape": target_shape.value, "count": counts[target_shape]}
    if family in {QuestionFamily.COLOR, QuestionFamily.SIZE, QuestionFamily.BINDING}:
        target = rng.choice(objects)
        return {"target_shape": target.shape.value, "target_object_id": target.object_id}
    first, second = objects[:2]
    relation = rng.choice(("left_of", "above", "larger_than"))
    truth = {
        "left_of": first.center[0] < second.center[0],
        "above": first.center[1] < second.center[1],
        "larger_than": first.size == Size.LARGE and second.size == Size.SMALL,
    }[relation]
    if bool(rng.getrandbits(1)) != truth:
        first, second = second, first
        truth = {
            "left_of": first.center[0] < second.center[0],
            "above": first.center[1] < second.center[1],
            "larger_than": first.size == Size.LARGE and second.size == Size.SMALL,
        }[relation]
    return {
        "subject_shape": first.shape.value,
        "object_shape": second.shape.value,
        "relation": relation,
        "truth": truth,
    }


def generate_split(
    *,
    split: str,
    total: int,
    seed: int,
    forbidden_signatures: set[str] | None = None,
    families: Sequence[QuestionFamily] | None = None,
) -> list[SceneSpec]:
    if split not in TEMPLATES:
        raise ValueError(f"Unknown split: {split}")
    selected_families = tuple(dict.fromkeys(families or FAMILIES))
    if not selected_families or any(family not in FAMILIES for family in selected_families):
        raise ValueError("Each split requires registered, unique question families")
    if total < len(selected_families):
        raise ValueError("Split total must cover every selected question family")
    forbidden = set(forbidden_signatures or ())
    local_signatures: set[str] = set()
    rng = random.Random(seed)
    base, remainder = divmod(total, len(selected_families))
    family_counts = {
        family: base + (index < remainder)
        for index, family in enumerate(selected_families)
    }
    scenes: list[SceneSpec] = []
    for family in selected_families:
        for family_index in range(family_counts[family]):
            for attempt in range(10_000):
                objects = _make_objects(family, rng)
                signature = _scene_signature(objects)
                if signature not in forbidden and signature not in local_signatures:
                    break
            else:
                raise RuntimeError(f"Exhausted unique scene signatures for {split}/{family.value}")
            template_index = (family_index + rng.randrange(len(TEMPLATES[split]))) % len(TEMPLATES[split])
            prompt = TEMPLATES[split][template_index].format(objects=_join_descriptions(objects))
            scene_id = f"{split}-{family.value}-{family_index:04d}"
            metadata = _primary_metadata(family, objects, rng)
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    split=split,
                    family=family,
                    prompt=prompt,
                    objects=objects,
                    generator_seed=seed,
                    template_id=f"{split}-t{template_index}",
                    signature=signature,
                    metadata=metadata,
                )
            )
            local_signatures.add(signature)
    return scenes


def build_splits(seed: int = 20260827) -> dict[str, list[SceneSpec]]:
    """Build the registered 2400/200/600 split with global signature exclusion."""

    used: set[str] = set()
    output: dict[str, list[SceneSpec]] = {}
    for offset, (name, total) in enumerate(
        (("train", 2400), ("tier_a_probe", 200), ("tier_a_outcome", 600))
    ):
        scenes = generate_split(
            split=name,
            total=total,
            seed=seed + offset * 1_000_003,
            forbidden_signatures=used,
        )
        output[name] = scenes
        used.update(scene.signature for scene in scenes)
    return output


def with_scene_id(scene: SceneSpec, scene_id: str, split: str | None = None) -> SceneSpec:
    return replace(scene, scene_id=scene_id, split=split or scene.split)
