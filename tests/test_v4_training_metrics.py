"""Dynamic pilot: score coverage, holdout isolation, resume and real updates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from selfsight.schemas import AtomicObservation, CandidateRecord, ObservationResult, SelectionDecision
from selfsight.utils.hashing import rgb_sha256
from selfsight.v4.evaluate import fixed_atomic_score, self_selection_scores, summarize_selection
from selfsight.v4.probe import spec_questions
from selfsight.v4.spec import SceneSpec, SpecObject
from selfsight.v4.train import (
    ARMS, ReplayExample, TrainingCorpus, parameter_digest, parameter_update_stats,
    pending_rounds, prepare_round, previous_checkpoint, restrict_replay,
    seed_training, select_by_observation, train_arm, trainable_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def _spec(name="p"):
    return SceneSpec(spec_id=name, prompt="one red book and two blue notebooks",
                     objects=(SpecObject("book", "red", 1), SpecObject("book", "blue", 2)))


def _observation(answers, image_hash="hash"):
    return ObservationResult("request", "self", "revision", image_hash, tuple(answers))


def _correct(question):
    return AtomicObservation(question.question_id, question.expected_answer,
                             question.expected_answer, False)


def test_missing_error_and_abstention_cannot_inflate_atomic_score():
    questions = spec_questions(_spec())
    observation = _observation([
        _correct(questions[0]),
        replace(_correct(questions[1]), abstain=True),
        replace(_correct(questions[2]), error="inference failed"),
    ])
    score = fixed_atomic_score(observation, questions)
    assert score == {"s_select": 0.25, "correct": 1, "available": 1, "total": 4,
                     "errors": 1, "abstained": 1, "missing": 1}
    assert fixed_atomic_score(_observation([]), questions)["s_select"] == 0


def test_atomic_score_rejects_duplicate_and_unrequested_answers():
    questions = spec_questions(_spec())
    with pytest.raises(ValueError, match="Duplicate"):
        fixed_atomic_score(_observation([_correct(questions[0])] * 2), questions)
    with pytest.raises(ValueError, match="Unexpected"):
        fixed_atomic_score(_observation([replace(_correct(questions[0]), question_id="other")]), questions)


def test_same_outcome_images_have_durable_scores_and_resume_identity_guards(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    spec = _spec()
    calls = []

    class Backbone:
        def observe_atoms(self, path, questions):
            calls.append(path)
            assert all(spec.prompt in question.text for question in questions)
            return _observation([_correct(question) for question in questions], rgb_sha256(path))

    kwargs = dict(specs={"p": spec}, images={"p": str(image)},
                  output_path=tmp_path / "s_select.jsonl", metadata={"step": 0})
    first = self_selection_scores(Backbone(), **kwargs)
    second = self_selection_scores(Backbone(), **kwargs)
    assert first == second and len(calls) == 1
    assert first[0]["s_select"] == 1
    assert len(first[0]["questions"]) == len(first[0]["observation"]["answers"]) == 4
    summary = summarize_selection(first)
    assert summary["n"] == 1 and summary["coverage"] == 1 and summary["mean"] == 1
    with pytest.raises(ValueError, match="different checkpoint"):
        self_selection_scores(Backbone(), **{**kwargs, "metadata": {"step": 1}})
    changed_spec = replace(spec, prompt="two red books")
    with pytest.raises(ValueError, match="questions changed"):
        self_selection_scores(Backbone(), **{**kwargs, "specs": {"p": changed_spec}})
    Image.new("RGB", (4, 4), "blue").save(image)
    with pytest.raises(ValueError, match="image/questions changed"):
        self_selection_scores(Backbone(), **kwargs)


def test_replay_is_restricted_by_spec_not_just_by_selected_training_images():
    corpus = TrainingCorpus(specs={}, replay=tuple(
        ReplayExample(f"{name}.png", "question", "answer", f"sample-{name}", prompt_id=name)
        for name in ("train", "outcome", "probe")))
    assert [item.prompt_id for item in restrict_replay(corpus, ["train"]).replay] == ["train"]
    with pytest.raises(ValueError, match="no prompt ID"):
        restrict_replay(replace(corpus, replay=(replace(corpus.replay[0], prompt_id=""),)), ["train"])


def test_max_rounds_limits_invocation_without_shrinking_frozen_total():
    assert pending_rounds(10, [], 1) == [0]
    assert pending_rounds(10, [0], 2) == [1, 2]
    assert pending_rounds(10, list(range(9)), 3) == [9]
    assert pending_rounds(10, list(range(10)), 1) == []
    assert pending_rounds(10, [0], None) == list(range(1, 10))
    with pytest.raises(ValueError, match="contiguous"):
        pending_rounds(10, [0, 2], 1)
    with pytest.raises(ValueError, match="positive"):
        pending_rounds(10, [], 0)


def test_missing_previous_checkpoint_is_only_allowed_for_round_zero(tmp_path):
    assert not previous_checkpoint(tmp_path, "naive", 0).exists()
    with pytest.raises(FileNotFoundError, match="refusing to reset to base"):
        previous_checkpoint(tmp_path, "naive", 1)
    checkpoint = tmp_path / "checkpoints" / "naive" / "round-000"
    checkpoint.mkdir(parents=True)
    for filename in ("manifest.json", "adapter.pt", "training_state.pt"):
        (checkpoint / filename).write_text("saved", encoding="utf-8")
    assert previous_checkpoint(tmp_path, "naive", 1) == checkpoint


def test_partial_round_checkpoints_are_archived_and_done_rounds_are_untouched(tmp_path):
    partial = tmp_path / "rounds" / "round-000"
    partial.mkdir(parents=True)
    (partial / "failure.log").write_text("failed in second arm", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / ARMS[0] / "round-000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.pt").write_text("first arm result", encoding="utf-8")
    restarted = prepare_round(tmp_path, 0, arms=ARMS)
    abandoned = list(restarted.parent.glob("round-000.abandoned-*"))
    assert len(abandoned) == 1 and not checkpoint.exists()
    assert (abandoned[0] / "failure.log").is_file()
    assert (abandoned[0] / "checkpoints" / ARMS[0] / "adapter.pt").read_text() == "first arm result"
    (restarted / "DONE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="completed round"):
        prepare_round(tmp_path, 0, arms=ARMS)
    assert (restarted / "DONE.json").is_file()


def test_an_arm_outside_the_registered_pairing_is_archived_too(tmp_path):
    """A blind-self run trains naive and blind_self, and `ARMS` is neither.

    Left behind, `checkpoints/blind_self/round-NNN` is what the retry resumes
    from: weights written by the attempt that crashed, carried forward with
    nothing in the output saying the round was ever restarted. The trajectory
    the paper reads for arm B would be wrong from that round on.
    """

    (tmp_path / "rounds" / "round-000").mkdir(parents=True)
    saved = {}
    for arm in ("naive", "blind_self"):
        checkpoint = tmp_path / "checkpoints" / arm / "round-000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.pt").write_text(f"{arm} result", encoding="utf-8")
        saved[arm] = checkpoint

    restarted = prepare_round(tmp_path, 0, arms=("naive", "blind_self"))
    abandoned = list(restarted.parent.glob("round-000.abandoned-*"))
    assert len(abandoned) == 1
    for arm, checkpoint in saved.items():
        assert not checkpoint.exists(), f"{arm} was left where the retry will find it"
        archived = abandoned[0] / "checkpoints" / arm / "adapter.pt"
        assert archived.read_text() == f"{arm} result"


def test_stage_train_archives_the_arms_it_resolved(tmp_path):
    """The argument has to reach the call, and the call is not unit-testable.

    `stage_train` loads a backbone before it gets here, so this reads the source
    the way tests/test_round_zero_weights.py does. What it pins is that
    `prepare_round` is called with `arms=arms` -- the set `resolve_arms`
    returned -- and not with a literal or nothing at all.
    """

    import ast

    tree = ast.parse((ROOT / "scripts" / "v4_train.py").read_text(encoding="utf-8"))
    stage = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "stage_train")
    calls = [node for node in ast.walk(stage)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "prepare_round"]
    assert len(calls) == 1, "one round loop, one archive point"
    passed = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert "arms" in passed, "prepare_round would archive the registered pairing instead"
    assert isinstance(passed["arms"], ast.Name) and passed["arms"].id == "arms", (
        "the arm set has to be the one resolve_arms returned, not a literal")


def _candidate(index):
    return CandidateRecord(candidate_id=f"p:{index}", prompt_id="p", scene_id="p",
                           sampling_seed=index, image_path=f"image-{index}.png", rgb_sha256="hash",
                           generator_id="test", generator_revision="test", checkpoint_id="test")


def test_training_selector_does_not_prefer_abstention_to_wrong_answer(tmp_path):
    spec = _spec()
    questions = spec_questions(spec)

    class Backbone:
        def observe_atoms(self, path, wrapped_questions):
            if path == "image-0.png":
                return _observation([_correct(questions[0])])
            return _observation([_correct(questions[0]), _correct(questions[1]),
                                 replace(_correct(questions[2]), normalized_answer="wrong"),
                                 replace(_correct(questions[3]), normalized_answer="wrong")])

    decisions = select_by_observation(
        arm="naive", backbone=Backbone(), observer=None,
        corpus=TrainingCorpus({"p": spec}, ()), pools={"p": [_candidate(0), _candidate(1)]},
        observation_path=tmp_path / "observations.jsonl")
    assert decisions[0].selected_candidate_id == "p:1"
    assert decisions[0].scores == {"p:0": 0.25, "p:1": 0.5}
    saved = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]
    assert saved[0]["missing"] == 3


def test_seeded_base_and_parameter_delta_measure_actual_update():
    torch = pytest.importorskip("torch")
    seed_training(91)
    model = torch.nn.Linear(2, 1)
    before = trainable_snapshot(model)
    seed_training(91)
    same = trainable_snapshot(torch.nn.Linear(2, 1))
    assert parameter_digest(before) == parameter_digest(same)
    with torch.no_grad():
        model.weight.add_(0.25)
    after = trainable_snapshot(model)
    stats = parameter_update_stats(before, after)
    assert stats["parameter_delta_l2"] == pytest.approx(2 ** 0.5 * 0.25)
    assert stats["parameter_changed_elements"] == 2
    assert stats["parameter_digest_before"] != stats["parameter_digest_after"]
    assert parameter_update_stats(before, same)["parameter_changed_elements"] == 0


def test_train_arm_records_gradient_and_real_parameter_movement():
    torch = pytest.importorskip("torch")

    class Backbone:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)

        def generation_loss(self, batch):
            return (self.model.weight - 1).square().mean()

    backbone = Backbone()
    optimizer = torch.optim.SGD(backbone.model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    decision = SelectionDecision("p", "naive", ("p:0",), "p:0", {}, "test", "test")
    training = {"micro_batch_size": 1, "gradient_accumulation_steps": 2,
                "optimizer_steps_per_round": 2, "understanding_replay_ratio": 0,
                "max_grad_norm": 1}
    report = train_arm(arm="naive", backbone=backbone, optimizer=optimizer, scheduler=scheduler,
                       decisions=[decision], candidates=[_candidate(0)],
                       corpus=TrainingCorpus({"p": _spec()}, ()), training=training,
                       seed=91, round_index=0)
    assert report["parameter_delta_l2"] > 0
    assert report["parameter_changed_elements"] == 1
    assert len(report["gradient_norms_before_clip"]) == 2
    assert report["parameter_digest_before"] != report["parameter_digest_after"]

