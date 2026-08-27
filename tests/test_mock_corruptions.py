from __future__ import annotations

from selfsight.data.generator import build_splits
from selfsight.pilot.mock_loop import _corrupt_scene
from selfsight.schemas import QuestionFamily


def test_mock_corruption_is_defined_for_every_question_family() -> None:
    scenes = build_splits(20260827)["train"]
    for family in QuestionFamily:
        scene = next(item for item in scenes if item.family == family)
        corrupted = _corrupt_scene(scene)
        assert corrupted.scene_id == scene.scene_id
        assert corrupted.objects != scene.objects
