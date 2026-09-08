"""The blind-self arm: naive with the description withheld, and nothing else.

Naive and RFO-Gold differ in two ways at once -- who is looking (the backbone
itself against a frozen external observer) and what they are told (the
description against nothing) -- so the registered pairing cannot attribute its
gap to either factor. Blind-Self is the missing cell of that 2x2, and the whole
value of it rests on differing from naive in exactly one thing.

The measurement that motivates the arm: on trials where the picture and the
description disagree, the same backbone answers 0.654 blind and 0.336 told
(review-packets/context-ablation-20260908, McNemar 221:9, p=5.1e-54). A
blind-self arm that leaked the prompt would quietly be a second naive arm, the
comparison would read as "no effect", and nothing else in the run would notice.
That is what these tests exist to prevent.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfsight.schemas import AtomicObservation, CandidateRecord, ObservationResult
from selfsight.v4.observe import PROMPTED_PREAMBLE, blind, observe_blind_self, observe_naive
from selfsight.v4.probe import spec_questions
from selfsight.v4.spec import SceneSpec, SpecObject
from selfsight.v4.train import ARMS, BLIND_SELF, RFO_SELF, SELECTORS, select_by_observation

PROMPT = "a red cube and two blue spheres"


def _spec(spec_id: str = "v4a-0001") -> SceneSpec:
    return SceneSpec(
        spec_id=spec_id,
        prompt=PROMPT,
        objects=(
            SpecObject(object="cube", color="red", count=1),
            SpecObject(object="sphere", color="blue", count=2),
        ),
    )


def _backbone(seen: list) -> object:
    """Records what it was asked, and answers every question as requested.

    Answering rather than abstaining matters: `select_by_observation` drops
    candidates whose answers are all unavailable, so a backbone that abstained
    would make the routing test below pass for the wrong reason.
    """

    class Backbone:
        def observe_atoms(self, image_path, questions):
            questions = tuple(questions)
            seen.append((image_path, questions))
            return ObservationResult(
                request_id="x", observer_id="showo2", observer_revision="r",
                rgb_sha256="",
                answers=tuple(
                    AtomicObservation(question_id=question.question_id,
                                      raw_answer=question.expected_answer,
                                      normalized_answer=question.expected_answer,
                                      abstain=False)
                    for question in questions),
            )

    return Backbone()


def _candidate(spec: SceneSpec, index: int) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"{spec.spec_id}-{index}", prompt_id=spec.spec_id,
        scene_id=spec.spec_id, sampling_seed=700 + index,
        image_path=f"C:/img/{spec.spec_id}-{index}.png", rgb_sha256="",
        generator_id="showo2", generator_revision="r", checkpoint_id="ck")


# -------------------------------------------------------- the one difference


def test_blind_self_asks_the_question_with_no_description_attached():
    """The arm's entire content. If this passed trivially the arm is pointless."""

    spec = _spec()
    seen: list = []
    observe_blind_self(_backbone(seen), prompt=spec.prompt,
                       questions=spec_questions(spec), image_path="C:/img/a.png")
    (_, asked), = seen
    plain = spec_questions(spec)
    assert [question.text for question in asked] == [question.text for question in plain]
    for question in asked:
        assert PROMPT not in question.text
        assert "You were asked to draw" not in question.text


def test_naive_still_attaches_it_so_the_contrast_stays_a_contrast():
    """A refactor that blinded naive would zero the effect and pass every other test."""

    spec = _spec()
    seen: list = []
    observe_naive(_backbone(seen), prompt=spec.prompt,
                  questions=spec_questions(spec), image_path="C:/img/a.png")
    (_, asked), = seen
    for question, plain in zip(asked, spec_questions(spec)):
        assert question.text == PROMPTED_PREAMBLE.format(prompt=PROMPT, question=plain.text)


def test_the_two_arms_differ_in_the_text_and_in_nothing_else():
    """Same questions, same ids, same expected answers, same image, same order."""

    spec = _spec()
    told: list = []
    withheld: list = []
    observe_naive(_backbone(told), prompt=spec.prompt,
                  questions=spec_questions(spec), image_path="C:/img/a.png")
    observe_blind_self(_backbone(withheld), prompt=spec.prompt,
                       questions=spec_questions(spec), image_path="C:/img/a.png")
    (told_image, told_questions), = told
    (blind_image, blind_questions), = withheld
    assert told_image == blind_image
    assert [q.question_id for q in told_questions] == [q.question_id for q in blind_questions]
    assert ([q.expected_answer for q in told_questions]
            == [q.expected_answer for q in blind_questions])
    assert [q.text for q in told_questions] != [q.text for q in blind_questions]


# ------------------------------------------------------------------ the guard


def test_a_question_that_already_carries_the_prompt_is_refused():
    """`observe_rfo` gets this from the wire. Called directly, it has to be asserted."""

    spec = _spec()
    leaked = tuple(replace(question, text=f"Draw: {PROMPT}. {question.text}")
                   for question in spec_questions(spec))
    with pytest.raises(ValueError, match="carries the generating description"):
        blind(spec.prompt, leaked)


def test_the_guard_sees_through_the_line_wrapping_the_preamble_adds():
    """The realistic leak arrives reflowed across lines, so a substring test on
    the raw text would miss it. Normalising whitespace both sides is what catches it."""

    spec = _spec()
    wrapped = tuple(
        replace(question, text=PROMPTED_PREAMBLE.format(prompt=PROMPT, question=question.text))
        for question in spec_questions(spec))
    with pytest.raises(ValueError, match="carries the generating description"):
        blind(spec.prompt, wrapped)


def test_an_empty_prompt_is_refused_rather_than_vacuously_passing():
    """Searching for the empty string finds it everywhere, so the check would have
    to be skipped -- and a skipped check reads exactly like a passing one."""

    with pytest.raises(ValueError, match="in order to check it is absent"):
        blind("", spec_questions(_spec()))


# ---------------------------------------------------------------- the routing


def test_selecting_blind_self_never_reaches_the_backbone_with_the_prompt():
    """End to end through the selector, which is what the run actually calls."""

    spec = _spec()
    seen: list = []
    corpus = SimpleNamespace(specs={spec.spec_id: spec})
    pools = {spec.spec_id: [_candidate(spec, 0), _candidate(spec, 1)]}
    decisions = select_by_observation(arm=BLIND_SELF, backbone=_backbone(seen),
                                      observer=None, corpus=corpus, pools=pools)
    assert len(seen) == 2, "each candidate in the pool is looked at once"
    for _, questions in seen:
        for question in questions:
            assert PROMPT not in question.text
    (decision,) = decisions
    assert decision.arm == BLIND_SELF
    assert decision.selected_candidate_id is not None


def test_blind_self_is_not_handed_an_observer_it_could_start_using():
    with pytest.raises(ValueError, match="must not have one"):
        select_by_observation(arm=BLIND_SELF, backbone=object(), observer=object(),
                              corpus=None, pools={})


def test_blind_self_is_selectable_but_not_part_of_the_registered_pairing():
    """Adding it to ARMS would change what the frozen main run means."""

    assert BLIND_SELF in SELECTORS
    assert BLIND_SELF not in ARMS
    assert ARMS == ("naive", "rfo_gold")
    assert RFO_SELF not in ARMS


# ------------------------------------------------------------- the arm set


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_train.py"
    spec = importlib.util.spec_from_file_location("v4_train_arm_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_invocation_that_does_not_mention_arms_still_runs_the_registered_pair():
    """Every existing command has to keep reproducing the run it already produced."""

    runner = _runner()
    assert runner.resolve_arms(argparse.Namespace()) == ARMS
    assert runner.resolve_arms(argparse.Namespace(arms=None)) == ARMS


def test_a_repeated_arm_is_refused_instead_of_being_silently_deduplicated():
    """Two columns of one arm would train it twice and pair it against itself."""

    runner = _runner()
    with pytest.raises(SystemExit, match="repeats an arm"):
        runner.resolve_arms(argparse.Namespace(arms=["naive", "naive"]))


def test_an_empty_arm_set_is_refused():
    runner = _runner()
    with pytest.raises(SystemExit, match="at least one arm"):
        runner.resolve_arms(argparse.Namespace(arms=[]))


def test_the_arm_set_keeps_the_order_it_was_given():
    """`arms[0]` seeds the base checkpoint and names the pairing key, so the
    order is not cosmetic: a set here would make the run depend on hash order."""

    runner = _runner()
    assert runner.resolve_arms(argparse.Namespace(arms=[BLIND_SELF, "naive"])) == \
        (BLIND_SELF, "naive")


def test_blind_self_can_actually_be_asked_for_on_the_command_line(monkeypatch, tmp_path):
    """`choices` is a second, separate list of what an arm is allowed to be, and
    pinning it to the registered pairing would leave the arm implemented,
    tested, and unreachable from any command that could run it."""

    runner = _runner()
    reached: dict = {}
    monkeypatch.setattr(runner, "stage_train",
                        lambda args: reached.update(arms=runner.resolve_arms(args)))
    monkeypatch.setattr(sys, "argv", ["v4_train.py", "train", "--outdir", str(tmp_path),
                                      "--arms", BLIND_SELF])
    runner.main()
    assert reached["arms"] == (BLIND_SELF,)


def test_an_arm_nobody_implemented_is_refused_before_the_models_load(monkeypatch):
    """A typo in an arm name is a two-second error, not an hour-long one."""

    runner = _runner()
    monkeypatch.setattr(sys, "argv", ["v4_train.py", "train", "--outdir", ".",
                                      "--arms", "blind-self"])
    with pytest.raises(SystemExit) as refusal:
        runner.main()
    assert refusal.value.code == 2
