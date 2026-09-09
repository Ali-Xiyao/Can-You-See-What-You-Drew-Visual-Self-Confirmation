"""Tests for the deviation 11.3 blind re-observation pass.

The script's GPU half cannot be tested without a GPU and a checkpoint. Its
other half decides which checkpoint each step maps to and which questions get
asked, and both of those are silent when wrong: a step pointed at the previous
round's adapter still produces 64 plausible scores, and a question that still
carries the preamble still produces a number called blind.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.analysis.endpoint2 import BLIND_CONDITION
from selfsight.schemas import AtomicObservation, ObservationResult
from selfsight.v4.checkpoint_reload import checkpoint_for
from selfsight.v4.observe import PROMPTED_PREAMBLE

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_blind_observe.py"


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_blind_observe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blind = _module()


def _question(index: int, text: str) -> dict:
    return {"question_id": f"p1:exists:{index}", "atom_id": f"p1:exists:a{index}",
            "family": "existence", "text": text, "expected_answer": "yes",
            "question_format": "forced_choice", "choices": ["yes", "no"],
            "choice_order_seed": 65}


# --- which checkpoint a step means ------------------------------------------


def test_step_zero_is_the_untrained_base_for_both_arms():
    run = Path("run")
    for arm in ("naive", "rfo_gold"):
        assert checkpoint_for(run, arm, 0, 8) == run / "checkpoints/base/round--01"


def test_the_first_trained_checkpoint_is_round_zero():
    # step 8 is round-000, not round-001. Off by one here would score every
    # checkpoint against the adapter from one round earlier, and the resulting
    # curve would still look like a curve.
    run = Path("run")
    assert checkpoint_for(run, "naive", 8, 8) == run / "checkpoints/naive/round-000"
    assert checkpoint_for(run, "naive", 24, 8) == run / "checkpoints/naive/round-002"
    assert checkpoint_for(run, "rfo_gold", 88, 8) == run / "checkpoints/rfo_gold/round-010"


def test_a_step_between_checkpoints_is_refused():
    with pytest.raises(ValueError, match="multiple of 8"):
        checkpoint_for(Path("run"), "naive", 12, 8)


def test_the_round_size_comes_from_the_config_not_a_constant():
    run = Path("run")
    assert checkpoint_for(run, "naive", 8, 4) == run / "checkpoints/naive/round-001"


# --- what gets asked --------------------------------------------------------


def test_the_saved_questions_are_used_verbatim():
    row = {"prompt_id": "p1", "questions": [_question(0, "Is there a red apple?"),
                                            _question(1, "Is there a blue mug?")]}
    questions = blind.bare_questions(row)
    assert [question.text for question in questions] == ["Is there a red apple?",
                                                         "Is there a blue mug?"]
    assert questions[0].choice_order_seed == 65


def test_a_question_still_carrying_the_preamble_is_refused():
    """Deviation 11.3's failure mode, and the only one that is silent.

    A wrapped question produces a perfectly ordinary score. Nothing downstream
    could tell it apart from a blind one, so the refusal has to happen here.
    """

    wrapped = PROMPTED_PREAMBLE.format(prompt="two blue pens", question="Is there a pen?")
    row = {"prompt_id": "p1", "questions": [_question(0, wrapped)]}
    with pytest.raises(SystemExit, match="would not be blind"):
        blind.bare_questions(row)


def test_the_preamble_marker_still_matches_the_preamble():
    # The script asserts this at import. Pinning it here as well means the
    # failure arrives from the test suite rather than from a GPU job that has
    # already loaded a checkpoint.
    assert blind.PREAMBLE_MARKER in PROMPTED_PREAMBLE


def test_the_script_never_applies_the_preamble():
    """Parsed, not grepped: the docstring names PROMPTED_PREAMBLE on purpose.

    The import exists so the marker cannot drift. What must not exist is a
    call that formats it into a question, or a call to `observe.prompted`.
    """

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "format":
            value = function.value
            if isinstance(value, ast.Name) and value.id == "PROMPTED_PREAMBLE":
                offenders.append(node.lineno)
        if isinstance(function, ast.Name) and function.id == "prompted":
            offenders.append(node.lineno)
    assert offenders == [], f"the preamble is applied at line(s) {offenders}"


# --- the condition label ----------------------------------------------------


def test_the_condition_label_is_the_one_the_analysis_demands():
    # The writer and the reader are in different files and the reader refuses
    # anything else, so a rename on one side would turn every checkpoint into
    # "deviation 11.3 forbids substituting the prompted score" at analysis
    # time -- after the GPU hours had been spent.
    assert blind.CONDITION == BLIND_CONDITION == "image_only"


# --- reading the prompted pass ----------------------------------------------


def _write_s_select(run: Path, arm: str, step: int, rows: list[dict]) -> None:
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s_select.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_steps_present_needs_the_prompted_pass_not_just_the_directory(tmp_path: Path):
    run = tmp_path / "run"
    (run / "evaluations" / "naive" / "step-00008").mkdir(parents=True)
    _write_s_select(run, "naive", 0, [{"prompt_id": "p1"}])
    assert blind.steps_present(run, "naive") == [0]


def test_steps_present_is_empty_for_an_arm_that_never_ran(tmp_path: Path):
    assert blind.steps_present(tmp_path, "blind_self") == []


def test_a_missing_prompted_pass_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit, match="No prompted pass"):
        blind.prompted_rows(tmp_path, "naive", 0)


def test_an_empty_prompted_pass_is_refused(tmp_path: Path):
    run = tmp_path / "run"
    _write_s_select(run, "naive", 0, [])
    with pytest.raises(SystemExit, match="is empty"):
        blind.prompted_rows(run, "naive", 0)


# --- the writer against the reader ------------------------------------------
#
# endpoint3 has this test and endpoint 2 did not. `observe_step` builds the
# record and `selfsight.analysis.endpoint2.load_series` reads it, and the two
# construct the same path and agree on the same key names in four places
# without anything checking that they still do. `s_select` is the sharpest
# one: the writer never names it, it arrives by `**fixed_atomic_score(...)`,
# so a rename there would write a valid file the reader cannot use, and the
# first thing to notice would be the analysis after 21 GPU-hours of passes.


class _Backbone:
    """Answers every question correctly, and returns the image it was given."""

    def __init__(self, rgb: str):
        self.rgb = rgb
        self.seen: list[str] = []

    def observe_atoms(self, image_path: str, questions):
        self.seen.append(image_path)
        answers = tuple(
            AtomicObservation(question.question_id, "A", question.expected_answer, False)
            for question in questions)
        return ObservationResult("request", "showo2", "revision", self.rgb, answers)


def _prompted_run(tmp_path: Path, arm: str, steps: tuple[int, ...]) -> Path:
    run = tmp_path / "run"
    for step in steps:
        _write_s_select(run, arm, step, [{
            "prompt_id": "p1", "image_path": str(tmp_path / "p1.png"),
            "rgb_sha256": "f" * 64, "question_digest": "q" * 64,
            "s_select": 0.5, "questions": [_question(0, "is there a cube?")]}])
        directory = run / "evaluations" / arm / f"step-{step:05d}"
        (directory / "verified.jsonl").write_text(json.dumps({
            "spec_id": "p1", "candidate_index": 0, "resolution": "agreed",
            "image_correct": True}) + "\n", encoding="utf-8")
        (directory / "manifest.jsonl").write_text(json.dumps({
            "spec_id": "p1", "candidate_index": 0}) + "\n", encoding="utf-8")
    return run


def test_the_blind_pass_this_script_writes_is_the_one_endpoint_2_reads(tmp_path: Path):
    from selfsight.analysis.endpoint2 import blind_path, load_series

    run = _prompted_run(tmp_path, "naive", (0, 8))
    for step in (0, 8):
        blind.observe_step(_Backbone("f" * 64), run, "naive", step,
                           run / "analysis" / "blind_observe" / "naive",
                           model_id="showo2-1p5b", adapter_digest="a" * 64)
        # The path the writer chose is the path the reader looks in. Both build
        # it from parts; neither imports it from the other.
        assert blind_path(run, "naive", step).exists()

    series = load_series(run, "naive", [0, 8])
    assert series.prompts == ("p1",) and series.steps == (0, 8)
    # Every question answered correctly, so the blind score is 1.0 -- and it
    # arrives under the key `s_select`, which the writer never spells: it comes
    # in through **fixed_atomic_score, so a rename there writes a valid file
    # this reader cannot use.
    assert series.scores[0, 0] == pytest.approx(1.0)
    assert series.scores[0, 1] == pytest.approx(1.0)
    # verified.jsonl said the image was right, so the label rides along.
    assert series.labels[0, 0] == 1 and series.labels[0, 1] == 1


def test_the_reader_rejects_what_the_writer_would_never_produce(tmp_path: Path):
    """The two checks load_series makes, against the writer's actual output.

    `condition` and `candidate_index` are constants in the record builder, so
    the writer cannot violate deviation 11.1 or 11.3. Reading them back off a
    written file is what makes that a fact rather than a reading of the source.
    """

    from selfsight.analysis.endpoint2 import BLIND_CONDITION, blind_path

    run = _prompted_run(tmp_path, "naive", (0,))
    blind.observe_step(_Backbone("f" * 64), run, "naive", 0,
                       run / "analysis" / "blind_observe" / "naive",
                       model_id="showo2-1p5b", adapter_digest="a" * 64)
    row = json.loads(blind_path(run, "naive", 0).read_text(encoding="utf-8").splitlines()[0])
    assert row["condition"] == BLIND_CONDITION
    assert row["candidate_index"] == 0
    # Deviation 11.3: the prompted score is carried alongside, never as the
    # score itself. Two different keys, and they differ here.
    assert row["prompted_s_select"] == 0.5
    assert row["s_select"] == pytest.approx(1.0)


def test_a_different_image_on_disk_stops_the_pass(tmp_path: Path):
    run = _prompted_run(tmp_path, "naive", (0,))
    with pytest.raises(SystemExit, match="not the one the prompted pass scored"):
        blind.observe_step(_Backbone("e" * 64), run, "naive", 0,
                           run / "analysis" / "blind_observe" / "naive",
                           model_id="showo2-1p5b", adapter_digest="a" * 64)


def test_the_default_outdir_and_the_reader_agree_on_the_directory(tmp_path: Path):
    """The round trip above passes the directory in; `main` has a default.

    So what is pinned above is `blind_path` against a path this test chose,
    and the directory the real pass writes to comes from `main`'s default
    instead. Compared as source because the alternative is a dry run needing
    a config, a checkpoint tree and a prompted pass to reach one string.
    """

    from selfsight.analysis.endpoint2 import blind_path

    assert blind_path(tmp_path, "naive", 8) == (
        tmp_path / "analysis" / "blind_observe" / "naive" / "step-00008.jsonl")
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'args.run / "analysis" / "blind_observe"' in source
    assert 'out_root / arm / f"step-{step:05d}.jsonl"' in source
