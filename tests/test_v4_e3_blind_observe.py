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
