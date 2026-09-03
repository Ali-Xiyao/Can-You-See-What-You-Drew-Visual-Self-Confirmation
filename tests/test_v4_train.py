"""The paired loop's four ways of breaking the pairing without raising.

The loop itself needs two GPUs' worth of model and is not tested here. What is
tested is everything that would let the two arms stop being comparable while the
run carries on looking healthy: a schedule that hands the arms different
prompts, an abstention that removes a prompt from one arm only, a replay corpus
built from images the adjudicator rejected, and a resume that throws away the
round you most wanted to look at.

There is also one test that is not about correctness at all -- the bank-size
guard -- because the v4 corpus holds 228 specs and the config asks for 640
training prompts. That mismatch should stop a run at second zero with a sentence
naming the number, not thirty hours in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfsight.schemas import SelectionDecision
from selfsight.training.paired import PromptScheduleEntry
from selfsight.v4.observe import PROMPTED_PREAMBLE, prompted
from selfsight.v4.spec import SceneSpec, SpecObject
from selfsight.v4.train import (
    ARMS,
    abandon_incomplete,
    build_schedule,
    completed_rounds,
    generate_and_select,
    load_training_corpus,
    pair_decisions,
    replay_indices,
    round_entries,
    write_done,
)

REPO = Path(__file__).resolve().parents[1]


def _spec(spec_id: str) -> SceneSpec:
    return SceneSpec(
        spec_id=spec_id,
        prompt=f"a red cube and two blue spheres ({spec_id})",
        objects=(
            SpecObject(object="cube", color="red", count=1),
            SpecObject(object="sphere", color="blue", count=2),
        ),
    )


def _write_run(root: Path, name: str, *, specs: list[str], correct: dict[tuple[str, int], bool],
               resolution: str = "agreed") -> str:
    run = root / name
    (run / "images").mkdir(parents=True)
    rows, verdicts = [], []
    for spec_id in specs:
        spec = _spec(spec_id)
        for index in (0, 1):
            image = str(run / "images" / f"{spec_id}-{index}.png")
            rows.append({"spec": spec.to_dict(), "candidate_index": index,
                         "image_path": image, "seed": 700_000_000 + index})
            verdicts.append({"image_path": image, "spec_id": spec_id,
                             "image_correct": correct[(spec_id, index)],
                             "resolution": resolution})
    (run / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "verified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verdicts), encoding="utf-8")
    return str(run)


# ------------------------------------------------------------------- corpus


def test_replay_never_uses_an_image_the_adjudicator_rejected(tmp_path):
    """Replay teaches the backbone what is in a picture.

    The answer it is taught is the spec's intended answer, which is only the
    true answer when the picture actually matches the spec. On a rejected image
    the two come apart, and replaying it would train the backbone to misread --
    degrading the naive arm's selections and the internal-consistency curve
    together, which is the one confound this experiment cannot absorb.
    """

    run = _write_run(tmp_path, "r", specs=["a", "b"],
                     correct={("a", 0): True, ("a", 1): False,
                              ("b", 0): False, ("b", 1): False})
    corpus = load_training_corpus([run])
    used = {Path(item.image_path).name for item in corpus.replay}
    assert used == {"a-0.png"}


def test_a_prompt_drawn_in_two_batches_is_still_one_prompt(tmp_path):
    """The same 228 specs appear in every batch under fresh seeds.

    Keying the training bank by the pools' `run:spec_id` would turn one prompt
    into three, and the schedule would train on it three times while reporting
    three distinct prompts.
    """

    first = _write_run(tmp_path, "batch1", specs=["a", "b"],
                       correct={(s, i): True for s in "ab" for i in (0, 1)})
    second = _write_run(tmp_path, "batch2", specs=["a", "b"],
                        correct={(s, i): True for s in "ab" for i in (0, 1)})
    corpus = load_training_corpus([first, second])
    assert corpus.prompt_ids == ("a", "b")
    assert len({item.image_path for item in corpus.replay}) == 8


def test_a_pool_with_no_verdict_supplies_neither_prompt_nor_replay(tmp_path):
    run = _write_run(tmp_path, "r", specs=["a"],
                     correct={("a", 0): True, ("a", 1): True},
                     resolution="pending_human")
    corpus = load_training_corpus([run])
    assert corpus.prompt_ids == ()
    assert corpus.replay == ()


# ------------------------------------------------------------------ schedule


def test_both_arms_get_the_same_schedule_because_there_is_only_one():
    """Determinism is the whole guarantee: same seed, same prompts, same latents."""

    ids = [f"s{index}" for index in range(20)]
    left = build_schedule(ids, rounds=2, prompts_per_round=5, candidate_k=3, seed=99)
    right = build_schedule(list(reversed(ids)), rounds=2, prompts_per_round=5,
                           candidate_k=3, seed=99)
    assert left != right, "the bank order does feed the shuffle"
    again = build_schedule(ids, rounds=2, prompts_per_round=5, candidate_k=3, seed=99)
    assert left == again


def test_the_candidate_seeds_within_an_entry_are_distinct():
    """K identical seeds would make a pool of K copies and every selection a tie."""

    entries = build_schedule([f"s{i}" for i in range(10)], rounds=1, prompts_per_round=4,
                             candidate_k=4, seed=3)
    for entry in entries:
        assert len(set(entry.candidate_seeds)) == 4


def test_a_bank_too_small_for_the_schedule_stops_before_the_first_image():
    """228 specs, and the inherited config asks for 10 x 64."""

    with pytest.raises(ValueError, match="640 prompt slots"):
        build_schedule([f"s{i}" for i in range(228)], rounds=10, prompts_per_round=64,
                       candidate_k=2, seed=1)


def test_reuse_across_rounds_is_possible_but_has_to_be_asked_for():
    ids = [f"s{index}" for index in range(10)]
    with pytest.raises(ValueError):
        build_schedule(ids, rounds=3, prompts_per_round=5, candidate_k=2, seed=1)
    entries = build_schedule(ids, rounds=3, prompts_per_round=5, candidate_k=2,
                             seed=1, max_epochs=2)
    assert len(entries) == 15
    for index in range(3):
        names = [entry.prompt_id for entry in round_entries(entries, index)]
        assert len(set(names)) == len(names), "a prompt appears twice in one round"


def test_a_round_can_never_want_more_prompts_than_the_bank_holds():
    with pytest.raises(ValueError, match="bank holds"):
        build_schedule(["a", "b"], rounds=1, prompts_per_round=3, candidate_k=2, seed=1)


# ------------------------------------------------------------------- pairing


def _decision(arm: str, prompt_id: str, selected: str | None) -> SelectionDecision:
    return SelectionDecision(
        prompt_id=prompt_id,
        arm=arm,
        candidate_pool_ids=(f"{prompt_id}:0", f"{prompt_id}:1"),
        selected_candidate_id=selected,
        scores={},
        selector_id="test",
        observer_revision="test",
        abstain=selected is None,
    )


def test_an_abstention_in_one_arm_removes_the_prompt_from_the_other():
    entries = [PromptScheduleEntry(0, name, (1, 2)) for name in ("p0", "p1", "p2")]
    paired = pair_decisions(entries, {
        "naive": [_decision("naive", name, f"{name}:0") for name in ("p0", "p1", "p2")],
        "rfo_self": [_decision("rfo_self", "p0", "p0:1"),
                     _decision("rfo_self", "p1", None),
                     _decision("rfo_self", "p2", "p2:0")],
    })
    assert [d.prompt_id for d in paired["naive"]] == ["p0", "p2"]
    assert [d.prompt_id for d in paired["rfo_self"]] == ["p0", "p2"]


def test_pairing_keeps_the_schedule_order_rather_than_the_decision_order():
    """Both arms must walk the prompts in the same order for the seeds to line up."""

    entries = [PromptScheduleEntry(0, name, (1,)) for name in ("p0", "p1", "p2")]
    paired = pair_decisions(entries, {
        "naive": [_decision("naive", name, f"{name}:0") for name in ("p2", "p0", "p1")],
        "rfo_self": [_decision("rfo_self", name, f"{name}:1") for name in ("p1", "p2", "p0")],
    })
    assert [d.prompt_id for d in paired["naive"]] == ["p0", "p1", "p2"]
    assert [d.prompt_id for d in paired["rfo_self"]] == ["p0", "p1", "p2"]


def test_both_arms_abstaining_everywhere_leaves_an_empty_round_not_a_wrong_one():
    entries = [PromptScheduleEntry(0, "p0", (1,))]
    paired = pair_decisions(entries, {
        "naive": [_decision("naive", "p0", None)],
        "rfo_self": [_decision("rfo_self", "p0", None)],
    })
    assert paired == {"naive": [], "rfo_self": []}


# -------------------------------------------------------------- rounds on disk


def test_resume_moves_an_unfinished_round_aside_and_does_not_delete_it(tmp_path):
    """The round that died is the evidence about why it died."""

    round_dir = tmp_path / "rounds" / "round-003"
    round_dir.mkdir(parents=True)
    (round_dir / "partial.log").write_text("crashed here", encoding="utf-8")
    moved = abandon_incomplete(round_dir)
    assert moved is not None and moved.exists()
    assert (moved / "partial.log").read_text(encoding="utf-8") == "crashed here"
    assert not round_dir.exists()


def test_only_rounds_with_the_sentinel_count_as_done(tmp_path):
    for index, done in ((0, True), (1, True), (2, False)):
        path = tmp_path / "rounds" / f"round-{index:03d}"
        path.mkdir(parents=True)
        if done:
            write_done(path, {"round": index})
    assert completed_rounds(tmp_path) == [0, 1]


def test_abandoning_a_round_that_was_never_started_is_not_an_error(tmp_path):
    assert abandon_incomplete(tmp_path / "rounds" / "round-000") is None


# ---------------------------------------------------------------- micro-batches


def test_replay_wraps_rather_than_running_off_the_end():
    assert replay_indices(count=3, cursor=4, total=5) == [4, 0, 1]


def test_replay_with_an_empty_corpus_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="replay corpus is empty"):
        replay_indices(count=1, cursor=0, total=0)


# -------------------------------------------------------------------- the arms


def test_the_naive_arm_is_not_handed_an_observer_it_could_start_using():
    with pytest.raises(ValueError, match="must not have one"):
        generate_and_select(arm="naive", backbone=object(), observer=object(),
                            corpus=None, entries=(), output_dir=".", checkpoint_id="c")


def test_the_rfo_arm_refuses_to_run_without_one():
    with pytest.raises(ValueError, match="needs a frozen observer"):
        generate_and_select(arm="rfo_self", backbone=object(), observer=None,
                            corpus=None, entries=(), output_dir=".", checkpoint_id="c")


def test_the_arm_names_match_the_config():
    assert ARMS == ("naive", "rfo_self")


# ---------------------------------------------------------------- the leak itself


def test_the_preamble_this_loop_trains_under_is_the_one_everything_else_measures():
    """Three files carry this string. Two of them predate the library copy.

    If this fails, the fix is to make the script copies import
    `selfsight.v4.observe`, never to edit one side until it passes -- a drift
    here means the training curve and the Gate B probe stop describing the same
    leak.
    """

    for name in ("v4_run_pipeline.py", "v4_gate_b_probe.py"):
        source = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert PROMPTED_PREAMBLE in source, f"{name} carries a different preamble"


def test_wrapping_a_question_keeps_everything_except_the_text():
    from selfsight.v4.probe import spec_questions

    original = spec_questions(_spec("s"))
    wrapped = prompted("a red cube and two blue spheres", original)
    assert len(wrapped) == len(original)
    for before, after in zip(original, wrapped):
        assert after.question_id == before.question_id
        assert after.expected_answer == before.expected_answer
        assert after.choices == before.choices
        assert before.text in after.text
        assert "You were asked to draw" in after.text
