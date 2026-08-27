from __future__ import annotations

from collections import Counter

from selfsight.data.eligible_e2 import build_eligible_scenes
from selfsight.data.generator import generate_split
from selfsight.schemas import QuestionFamily


def test_eligible_e2_splits_are_balanced_and_exclude_readiness_signatures() -> None:
    readiness = generate_split(split="train", total=6, seed=11)
    forbidden = {scene.signature for scene in readiness}
    families = (
        QuestionFamily.EXISTENCE,
        QuestionFamily.COUNT,
        QuestionFamily.COLOR,
        QuestionFamily.BINDING,
    )
    splits = build_eligible_scenes(
        families=families,
        forbidden_signatures=forbidden,
        seed=17,
        split_counts={"train": 8, "tier_a_probe": 4, "tier_a_outcome": 4},
    )
    seen = set(forbidden)
    for split, scenes in splits.items():
        assert len(scenes) == (8 if split == "train" else 4)
        assert not {scene.signature for scene in scenes}.intersection(seen)
        assert set(Counter(scene.family for scene in scenes)) == set(families)
        seen.update(scene.signature for scene in scenes)
