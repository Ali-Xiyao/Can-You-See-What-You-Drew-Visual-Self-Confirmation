from __future__ import annotations

from selfsight.analysis.gradient_gate import _batches
from selfsight.backbones.showo2 import Showo2Adapter, Showo2GenerationBatch
from selfsight.schemas import CandidateRecord, QuestionFamily, SceneSpec
from selfsight.showo_adapter import ShowoSFTBatch


def _inputs():
    scene = SceneSpec(
        scene_id="scene-1",
        split="probe",
        family=QuestionFamily.EXISTENCE,
        prompt="A red circle.",
        objects=(),
    )
    candidate = CandidateRecord(
        candidate_id="candidate-1",
        prompt_id=scene.scene_id,
        scene_id=scene.scene_id,
        sampling_seed=1,
        image_path="image.png",
        rgb_sha256="0" * 64,
        generator_id="model",
        generator_revision="revision",
        checkpoint_id="step-0",
    )
    return [scene.scene_id], {scene.scene_id: candidate}, {scene.scene_id: scene}


def test_gradient_batches_follow_showo2_public_training_contract() -> None:
    prompt_ids, selected, scenes = _inputs()
    adapter = Showo2Adapter(lazy=True)
    batches = _batches(
        prompt_ids,
        selected,
        scenes,
        adapter=adapter,
        micro_size=1,
        seed=17,
    )
    assert len(batches) == 1
    assert isinstance(batches[0], Showo2GenerationBatch)
    assert batches[0].images == ("image.png",)


def test_gradient_batches_keep_legacy_showo_contract() -> None:
    prompt_ids, selected, scenes = _inputs()
    batches = _batches(
        prompt_ids,
        selected,
        scenes,
        adapter=object(),
        micro_size=1,
        seed=17,
    )
    assert isinstance(batches[0], ShowoSFTBatch)
