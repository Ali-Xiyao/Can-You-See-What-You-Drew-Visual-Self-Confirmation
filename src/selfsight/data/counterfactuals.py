"""Registered Tier-B pixel counterfactual construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from selfsight.data.questions import build_primary_atom
from selfsight.schemas import Color, QuestionFamily, SceneSpec, Size
from selfsight.utils.hashing import sha256_json


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    category: str
    intent_prompt: str
    source: SceneSpec
    counterfactual: SceneSpec
    source_answer: str
    counterfactual_answer: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finalize(source: SceneSpec, changed: SceneSpec, category: str, index: int) -> CounterfactualPair:
    intent_prompt = source.prompt
    changed = replace(
        changed,
        scene_id=f"tier_b-{category}-{index:03d}-cf",
        split="tier_b",
        prompt="[WITHHELD: pixel counterfactual]",
        signature=sha256_json([asdict(item) for item in changed.objects]),
    )
    source = replace(
        source,
        scene_id=f"tier_b-{category}-{index:03d}-source",
        split="tier_b",
        prompt="[WITHHELD: pixel counterfactual]",
    )
    source_answer = build_primary_atom(source).answer
    changed_answer = build_primary_atom(changed).answer
    if source_answer == changed_answer:
        raise AssertionError(f"Counterfactual failed to change answer: {category}/{index}")
    return CounterfactualPair(
        pair_id=f"tier_b-{category}-{index:03d}",
        category=category,
        intent_prompt=intent_prompt,
        source=source,
        counterfactual=changed,
        source_answer=source_answer,
        counterfactual_answer=changed_answer,
    )


def _replace_object(scene: SceneSpec, object_id: str, **changes: object) -> SceneSpec:
    objects = tuple(
        replace(item, **changes) if item.object_id == object_id else item for item in scene.objects
    )
    return replace(scene, objects=objects)


def build_tier_b(outcome_scenes: list[SceneSpec]) -> list[CounterfactualPair]:
    by_family = {
        family: [scene for scene in outcome_scenes if scene.family == family]
        for family in QuestionFamily
    }
    pairs: list[CounterfactualPair] = []

    for index, source in enumerate(by_family[QuestionFamily.COUNT][:100]):
        target_shape = str(source.metadata["target_shape"])
        target = next(item for item in source.objects if item.shape.value == target_shape)
        objects = tuple(item for item in source.objects if item.object_id != target.object_id)
        metadata = dict(source.metadata)
        metadata["count"] = int(metadata["count"]) - 1
        changed = replace(source, objects=objects, metadata=metadata)
        pairs.append(_finalize(source, changed, "count_delete", index))

    for index, source in enumerate(by_family[QuestionFamily.COLOR][:100]):
        target_id = str(source.metadata["target_object_id"])
        target = next(item for item in source.objects if item.object_id == target_id)
        new_color = next(color for color in Color if color != target.color)
        changed = _replace_object(source, target_id, color=new_color)
        pairs.append(_finalize(source, changed, "color_change", index))

    spatial_sources = by_family[QuestionFamily.SPATIAL]
    position_sources = [
        scene for scene in spatial_sources if scene.objects[0].center[0] != scene.objects[1].center[0]
    ][:50]
    if len(position_sources) != 50:
        raise AssertionError("Not enough Tier-B sources with a strict horizontal relation")
    for index, source in enumerate(position_sources):
        first, second = source.objects[:2]
        objects = tuple(
            replace(item, center=second.center) if item.object_id == first.object_id else
            replace(item, center=first.center) if item.object_id == second.object_id else item
            for item in source.objects
        )
        metadata = {
            "subject_shape": first.shape.value,
            "object_shape": second.shape.value,
            "relation": "left_of",
            "truth": first.center[0] < second.center[0],
        }
        source = replace(source, metadata=metadata)
        changed_metadata = dict(metadata)
        changed_metadata["truth"] = not bool(metadata["truth"])
        changed = replace(source, objects=objects, metadata=changed_metadata)
        pairs.append(_finalize(source, changed, "relation_left_right", index))

    for index, source in enumerate(spatial_sources[50:100]):
        first, second = source.objects[:2]
        source_objects = tuple(
            replace(item, size=Size.LARGE) if item.object_id == first.object_id else
            replace(item, size=Size.SMALL) if item.object_id == second.object_id else item
            for item in source.objects
        )
        source = replace(
            source,
            objects=source_objects,
            metadata={
                "subject_shape": first.shape.value,
                "object_shape": second.shape.value,
                "relation": "larger_than",
                "truth": True,
            },
        )
        changed_objects = tuple(
            replace(item, size=Size.SMALL) if item.object_id == first.object_id else
            replace(item, size=Size.LARGE) if item.object_id == second.object_id else item
            for item in source.objects
        )
        changed = replace(source, objects=changed_objects, metadata={**source.metadata, "truth": False})
        pairs.append(_finalize(source, changed, "relation_size", index))

    for index, source in enumerate(by_family[QuestionFamily.BINDING][:100]):
        first, second = source.objects[:2]
        objects = tuple(
            replace(item, color=second.color) if item.object_id == first.object_id else
            replace(item, color=first.color) if item.object_id == second.object_id else item
            for item in source.objects
        )
        target_id = first.object_id
        source = replace(source, metadata={"target_shape": first.shape.value, "target_object_id": target_id})
        changed = replace(source, objects=objects)
        pairs.append(_finalize(source, changed, "binding_swap", index))

    expected = {
        "count_delete": 100,
        "color_change": 100,
        "relation_left_right": 50,
        "relation_size": 50,
        "binding_swap": 100,
    }
    actual = {category: sum(pair.category == category for pair in pairs) for category in expected}
    if actual != expected:
        raise AssertionError(f"Tier B composition mismatch: {actual}")
    return pairs
