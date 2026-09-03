"""The probe set's three decisions, and the two ways the runner can silently lie.

The gradient stage is a GPU job and is not tested here. What is tested is
everything that can go wrong *without* raising: a pool built from an image that
never got a verdict, a question set that differs between candidates, a selection
that is really a tie broken two different ways, and -- the one that would waste
the whole run -- an rfo request that carries the prompt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.schemas import (
    AtomicObservation,
    AtomicQuestion,
    ObservationResult,
    QuestionFamily,
    QuestionFormat,
)
from selfsight.v4.probe import build_pools, gold_selection, spec_questions
from selfsight.v4.spec import SceneSpec, SpecObject

REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "v4_gate_b_probe", REPO / "scripts" / "v4_gate_b_probe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _spec(spec_id: str = "s1", *, count: int = 1) -> SceneSpec:
    return SceneSpec(
        spec_id=spec_id,
        prompt="a red cube and two blue spheres",
        objects=(
            SpecObject(object="cube", color="red", count=1),
            SpecObject(object="sphere", color="blue", count=count),
        ),
    )


def _write_run(root: Path, name: str, rows: list[dict], verdicts: list[dict]) -> str:
    run = root / name
    run.mkdir(parents=True)
    (run / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "verified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verdicts), encoding="utf-8")
    return str(run)


def _manifest_row(spec: SceneSpec, index: int, image: str, seed: int) -> dict:
    return {"spec": spec.to_dict(), "candidate_index": index, "image_path": image, "seed": seed}


# ------------------------------------------------------------------ questions


def test_the_questions_come_from_the_spec_and_not_from_any_one_image():
    """Every candidate in a pool must be asked the identical thing.

    The v4 corpus questions are generated per image from that image's own
    detections. Scoring candidates against their own detections would score each
    render against itself, and the selection would be meaningless.
    """

    questions = spec_questions(_spec())
    texts = [question.text for question in questions]
    assert all("red cube" in text or "blue sphere" in text for text in texts)
    assert all(question.expected_answer in ("yes", "one", "two") for question in questions)


def test_a_count_atom_appears_only_when_more_than_one_was_requested():
    assert len(spec_questions(_spec(count=1))) == 2
    counted = spec_questions(_spec(count=2))
    assert len(counted) == 3
    count_atom = next(q for q in counted if ":count:" in q.question_id)
    assert count_atom.expected_answer == "two"
    assert set(count_atom.choices) == {"two", "one"}


def test_every_atom_is_forced_choice():
    """Open answers would let the two arms differ by verbosity rather than belief."""

    for question in spec_questions(_spec(count=3)):
        assert question.question_format is QuestionFormat.FORCED_CHOICE
        assert len(question.choices) == 2


def test_the_correct_option_is_not_always_the_same_letter():
    """A model with a letter preference must not be able to score above chance.

    With the correct answer pinned to A, every candidate in a pool scores alike,
    all three criteria fall through to the same tie-break, and the Gate B cosine
    reads 1.000 while measuring nothing.
    """

    first_letters = []
    for index in range(40):
        for question in spec_questions(_spec(f"s{index}", count=2)):
            first_letters.append(question.choices[0] == question.expected_answer)
    share = sum(first_letters) / len(first_letters)
    assert 0.3 < share < 0.7, f"correct answer sat in A {share:.0%} of the time"


def test_the_placement_is_identical_for_every_candidate_in_a_pool():
    """The candidates are only comparable if they were asked the same thing."""

    assert spec_questions(_spec("s1", count=2)) == spec_questions(_spec("s1", count=2))


def test_counting_atoms_do_not_use_the_counting_fallback_vocabulary():
    """`normalize_answer`'s counting branch rewrites "two" to "2".

    An expected answer of "two" would then never match a reply that names the
    word but not the letter, and the atom would score as wrong rather than
    abstain.
    """

    from selfsight.data.questions import normalize_answer

    count_atom = next(q for q in spec_questions(_spec(count=2)) if ":count:" in q.question_id)
    assert count_atom.family is QuestionFamily.EXISTENCE
    assert normalize_answer("two", count_atom) is None
    letter = "A" if count_atom.choices[0] == "two" else "B"
    assert normalize_answer(f"{letter}. two", count_atom) == "two"


# ---------------------------------------------------------------------- pools


def test_a_pool_is_dropped_whole_when_one_candidate_lacks_a_verdict(tmp_path):
    """Not partly. Dropping one candidate changes the pool the criteria choose from."""

    spec = _spec()
    rows = [_manifest_row(spec, i, f"img{i}.png", 100 + i) for i in range(3)]
    verdicts = [
        {"image_path": "img0.png", "resolution": "agreed", "image_correct": True},
        {"image_path": "img1.png", "resolution": "agreed", "image_correct": False},
        {"image_path": "img2.png", "resolution": "pending_human", "image_correct": False},
    ]
    run = _write_run(tmp_path, "r", rows, verdicts)
    assert build_pools((run,)) == []


def test_a_missing_verdict_row_also_drops_the_pool(tmp_path):
    spec = _spec()
    rows = [_manifest_row(spec, i, f"img{i}.png", 100 + i) for i in range(2)]
    verdicts = [{"image_path": "img0.png", "resolution": "agreed", "image_correct": True}]
    run = _write_run(tmp_path, "r", rows, verdicts)
    assert build_pools((run,)) == []


def test_only_pools_where_the_criteria_can_disagree_are_balanced(tmp_path):
    """An all-correct pool narrows the CI without carrying any information."""

    spec_mixed, spec_uniform = _spec("mixed"), _spec("uniform")
    rows = ([_manifest_row(spec_mixed, i, f"m{i}.png", 10 + i) for i in range(2)]
            + [_manifest_row(spec_uniform, i, f"u{i}.png", 20 + i) for i in range(2)])
    verdicts = [
        {"image_path": "m0.png", "resolution": "agreed", "image_correct": True},
        {"image_path": "m1.png", "resolution": "agreed", "image_correct": False},
        {"image_path": "u0.png", "resolution": "agreed", "image_correct": True},
        {"image_path": "u1.png", "resolution": "agreed", "image_correct": True},
    ]
    run = _write_run(tmp_path, "r", rows, verdicts)
    pools = {pool.spec.spec_id: pool for pool in build_pools((run,))}
    assert len(pools) == 2
    assert pools["mixed"].balanced
    assert not pools["uniform"].balanced


def test_gold_prefers_a_correct_candidate_then_breaks_ties_by_seed(tmp_path):
    """The same tie-break every other selector uses.

    If gold broke ties differently, "the criteria agreed" would sometimes mean
    "they happened to break a tie the same way", and the cosine would be
    measuring the tie-break.
    """

    spec = _spec()
    rows = [_manifest_row(spec, i, f"g{i}.png", seed)
            for i, seed in enumerate((300, 100, 200))]
    verdicts = [
        {"image_path": "g0.png", "resolution": "agreed", "image_correct": True},
        {"image_path": "g1.png", "resolution": "agreed", "image_correct": True},
        {"image_path": "g2.png", "resolution": "agreed", "image_correct": False},
    ]
    run = _write_run(tmp_path, "r", rows, verdicts)
    pool = build_pools((run,))[0]
    assert gold_selection(pool) == "s1:1"  # correct, and the lowest seed of the two


# --------------------------------------------------------------------- arms


def test_the_naive_arm_puts_the_description_in_front_of_every_question():
    """This is the leak `g_naive` is defined by; it must be present and verbatim."""

    spec = _spec()
    pool = runner.Pool(prompt_id="p", run="r", spec=spec,
                       candidates=(runner.PoolCandidate("s1:0", "a.png", 1, True),))
    seen: list[AtomicQuestion] = []

    class Backbone:
        def observe_atoms(self, image_path, questions):
            seen.extend(questions)
            return ObservationResult(request_id="x", observer_id="showo2",
                                     observer_revision="r", rgb_sha256="",
                                     answers=())

    runner._observe_naive(Backbone(), pool, pool.candidates[0])
    assert len(seen) == len(spec_questions(spec))
    for question, plain in zip(seen, spec_questions(spec)):
        assert question.text == runner.PROMPTED_PREAMBLE.format(prompt=spec.prompt,
                                                                question=plain.text)
        assert question.question_id == plain.question_id


def test_the_naive_preamble_is_byte_identical_to_the_main_pipeline():
    """Two measurements of the same leak are only comparable at the same wording."""

    source = (REPO / "scripts" / "v4_run_pipeline.py").read_text(encoding="utf-8")
    body = runner.PROMPTED_PREAMBLE
    assert f'PROMPTED_PREAMBLE = """{body}"""' in source


def test_the_rfo_arm_cannot_carry_the_prompt_even_if_someone_adds_it():
    """The blindness of `g_rfo` is enforced by the wire, not by convention."""

    from selfsight.observers.protocol import assert_blind_wire_payload

    spec = _spec()
    payload = {
        "schema_version": 1, "request_id": "r", "image_path": "C:/x/a.png",
        "rgb_sha256": "0" * 64,
        "questions": [{"question_id": "q", "family": "existence", "text": "t",
                       "question_format": "forced_choice", "choices": ["yes", "no"],
                       "choice_order_seed": 0, "prompt": spec.prompt}],
    }
    with pytest.raises(ValueError, match="Forbidden context key"):
        assert_blind_wire_payload(payload)


def test_the_rfo_wire_drops_the_expected_answer():
    """An observer that is told the expected answer is not observing."""

    from selfsight.schemas import BlindObservationRequest

    request = BlindObservationRequest(request_id="r", image_path="C:/x/a.png",
                                      rgb_sha256="0" * 64,
                                      questions=spec_questions(_spec()))
    wire = request.to_wire()
    assert all("expected_answer" not in item for item in wire["questions"])


# ----------------------------------------------------------------- pairing


def _observation(candidate_id: str, questions, answers: list[str]) -> ObservationResult:
    return ObservationResult(
        request_id=candidate_id, observer_id="o", observer_revision="r", rgb_sha256="",
        answers=tuple(
            AtomicObservation(question_id=question.question_id, raw_answer=answer,
                              normalized_answer=answer or None, abstain=not answer,
                              latency_ms=0.0)
            for question, answer in zip(questions, answers)),
    )


def test_a_prompt_where_one_arm_abstains_is_dropped_from_all_of_them(tmp_path, monkeypatch):
    """The bootstrap is paired: an unpaired prompt would corrupt the Gram matrices."""

    spec = _spec()
    questions = spec_questions(spec)
    pool = runner.Pool(prompt_id="r:s1", run="r", spec=spec,
                       candidates=(runner.PoolCandidate("s1:0", "a.png", 1, True),
                                   runner.PoolCandidate("s1:1", "b.png", 2, False)))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for arm, answers in (("naive", ["yes", "yes"]), ("rfo", ["", ""])):
        runner.write_jsonl(out_dir / f"observations.{arm}.jsonl", [
            {"prompt_id": pool.prompt_id, "candidate_id": candidate.candidate_id,
             "observation": json.loads(json.dumps(
                 runner.as_serializable(_observation(candidate.candidate_id, questions, answers))))}
            for candidate in pool.candidates
        ])
    runner.write_jsonl(out_dir / "pools.jsonl", [{"prompt_id": pool.prompt_id}])
    monkeypatch.setattr(runner, "load_pools", lambda _: [pool])

    runner.stage_select(type("A", (), {"outdir": str(out_dir)})())
    rows = runner.read_jsonl(out_dir / "selection.jsonl")
    assert rows[0]["dropped"] == "selector_abstained"
    assert "selected" not in rows[0]


def test_a_reopened_store_pairs_with_the_one_that_wrote_it(tmp_path):
    """`gram_matrices` only checks the ids it is given; reopening must preserve them."""

    import numpy as np

    from selfsight.v3.paired import PerPromptGradientStore, gram_matrices

    ids = ["p0", "p1", "p2"]
    written = PerPromptGradientStore(tmp_path / "g.f32", criterion="naive",
                                     dimension=4, capacity=3)
    for index, prompt_id in enumerate(ids):
        written.add(prompt_id, np.arange(4, dtype=np.float64) + index)
    written.finalize().close()

    reopened = PerPromptGradientStore.open_existing(
        tmp_path / "g.f32", criterion="naive", dimension=4, prompt_ids=ids)
    assert reopened.prompt_ids == tuple(ids)
    gram = gram_matrices(reopened, reopened)
    assert np.allclose(np.diag(gram.g_ll), [14.0, 30.0, 54.0])


def test_reopening_with_the_wrong_order_is_caught_by_the_pairing_check(tmp_path):
    import numpy as np

    from selfsight.v3.paired import PerPromptGradientStore, gram_matrices

    for name in ("a", "b"):
        store = PerPromptGradientStore(tmp_path / f"{name}.f32", criterion=name,
                                       dimension=2, capacity=2)
        store.add("p0", np.array([1.0, 0.0]))
        store.add("p1", np.array([0.0, 1.0]))
        store.finalize().close()
    left = PerPromptGradientStore.open_existing(tmp_path / "a.f32", criterion="a",
                                                dimension=2, prompt_ids=["p0", "p1"])
    right = PerPromptGradientStore.open_existing(tmp_path / "b.f32", criterion="b",
                                                 dimension=2, prompt_ids=["p1", "p0"])
    with pytest.raises(ValueError, match="same prompts in the same order"):
        gram_matrices(left, right)
