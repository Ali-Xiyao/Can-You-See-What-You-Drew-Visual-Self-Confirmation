"""Tests for the endpoint 2 driver's not-done logic.

The arithmetic is tested in `tests/test_endpoint2.py`. What is tested here is
the decision the driver makes *before* any arithmetic: whether there is enough
of a blind pass to fit a slope at all, and what it says when there is not.
Deviation 11.3 makes "not done" a registered outcome with a required
consequence -- no substituted prompted gap -- so the path that reaches it is
worth as much care as the path that produces a number.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_endpoint2.py"


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_endpoint2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _module()


def _adjudicated(run: Path, arm: str, steps: list[int]) -> None:
    for step in steps:
        directory = run / "evaluations" / arm / f"step-{step:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verified.jsonl").write_text(
            json.dumps({"spec_id": "p1", "candidate_index": 0,
                        "image_correct": True, "resolution": "agreed"}) + "\n",
            encoding="utf-8")


def _blind(run: Path, arm: str, steps: list[int]) -> None:
    directory = run / "analysis" / "blind_observe" / arm
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (directory / f"step-{step:05d}.jsonl").write_text(
            json.dumps({"prompt_id": "p1", "candidate_index": 0,
                        "condition": "image_only", "s_select": 0.5}) + "\n",
            encoding="utf-8")


def test_no_blind_pass_at_all_is_not_done(tmp_path: Path):
    run = tmp_path / "e3-s20260906"
    _adjudicated(run, "naive", [0, 8, 16])
    _adjudicated(run, "blind_self", [0, 8, 16])
    with pytest.raises(driver.NotDone, match="no image_only pass"):
        driver.steps_for(run, "naive", "blind_self")


def test_two_checkpoints_cannot_carry_a_slope(tmp_path: Path):
    run = tmp_path / "e3-s20260906"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16])
        _blind(run, arm, [0, 8])
    with pytest.raises(driver.NotDone, match="covers 2 checkpoint"):
        driver.steps_for(run, "naive", "blind_self")


def test_a_blind_pass_with_no_verdicts_to_pair_against_is_not_done(tmp_path: Path):
    """A blind score with no external label is not half a data point.

    This is the shape a stale output directory takes: the blind pass ran, the
    adjudication for that checkpoint was later moved or never finished, and the
    scores sit there looking complete.
    """

    run = tmp_path / "e3-s20260906"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16])
        _blind(run, arm, [0, 8, 16, 24])
    with pytest.raises(driver.NotDone, match=r"step\(s\) \[24\]"):
        driver.steps_for(run, "naive", "blind_self")


def test_the_checkpoints_the_blind_pass_missed_are_reported(tmp_path: Path):
    # Deviation 13.1 point 6: name them, do not quietly fit a shorter series.
    run = tmp_path / "e3-s20260906"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16, 24, 32])
        _blind(run, arm, [0, 8, 16])
    usable, missing = driver.steps_for(run, "naive", "blind_self")
    assert usable == [0, 8, 16]
    assert missing == [24, 32]


def test_a_step_only_one_arm_measured_blind_is_not_used(tmp_path: Path):
    run = tmp_path / "e3-s20260906"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16, 24])
    _blind(run, "naive", [0, 8, 16, 24])
    _blind(run, "blind_self", [0, 8, 16])
    usable, missing = driver.steps_for(run, "naive", "blind_self")
    assert usable == [0, 8, 16]
    assert missing == [24]


def test_not_done_writes_the_registered_sentence_and_no_slope(tmp_path: Path):
    driver.write_not_done(tmp_path, ["e3-s20260906: no image_only pass"],
                          arm_a="naive", arm_b="blind_self")
    payload = json.loads((tmp_path / "endpoint2.json").read_text(encoding="utf-8"))
    assert payload["status"] == "not_done"
    assert payload["substituted_prompted_gap"] is False
    assert "slope" not in json.dumps(payload)
    text = (tmp_path / "endpoint2.txt").read_text(encoding="utf-8")
    assert "NOT DONE" in text
    assert "deviation 11.3" in text.lower()


def test_not_done_has_its_own_exit_code():
    # Not 0: a wrapper must not carry on as though endpoint 2 had an answer.
    # Not 1: it is a registered outcome, not a crash.
    assert driver.NOT_DONE == 2
