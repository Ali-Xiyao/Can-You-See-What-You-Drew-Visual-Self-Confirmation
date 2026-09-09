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

from selfsight.analysis.endpoint2 import FALSIFIED, SeedVerdict, across_seeds

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


# --- failure condition 2, deviation 15.1 -------------------------------------


def _seed_verdict(seed: int, *, a_collapses: bool, b_collapses: bool,
                  b_exceeds_a: bool) -> SeedVerdict:
    return SeedVerdict(seed=seed, steps=(0, 8, 16), missing_steps=(),
                       slope_a=-0.01, slope_b=-0.005 if b_collapses else 0.01,
                       a_interval=(-0.02, -0.005 if a_collapses else 0.005),
                       b_interval=(-0.02, -0.001 if b_collapses else 0.005),
                       difference_interval=(0.001 if b_exceeds_a else -0.01, 0.02),
                       a_collapses=a_collapses, b_collapses=b_collapses,
                       b_exceeds_a=b_exceeds_a,
                       dropped_checkpoints=0, discarded_resamples=0)


def _text(*, a_collapses: bool, b_collapses: bool, b_exceeds_a: bool = False) -> str:
    verdicts = [_seed_verdict(seed, a_collapses=a_collapses, b_collapses=b_collapses,
                              b_exceeds_a=b_exceeds_a)
                for seed in driver.REGISTERED_SEEDS]
    return driver.report(verdicts, across_seeds(verdicts), arm_a="naive",
                         arm_b="blind_self", resamples=200, bootstrap_seed=1)


def test_failure_condition_2_prints_the_registered_sentence():
    text = _text(a_collapses=True, b_collapses=True)
    assert "FAILURE CONDITION 2 HOLDS" in text
    assert FALSIFIED in text
    assert "5/5 seeds" in text


def test_b_collapsing_without_a_does_not_trigger_it():
    # "only A collapsing while B also collapses" -- both clauses, and the
    # report has to say which one is missing rather than print nothing.
    text = _text(a_collapses=False, b_collapses=True)
    assert "FAILURE CONDITION 2 HOLDS" not in text
    assert "B collapses but A does not" in text


def test_a_holding_b_leaves_the_condition_untriggered():
    text = _text(a_collapses=True, b_collapses=False)
    assert "not triggered" in text
    assert FALSIFIED not in text


def test_the_per_seed_row_carries_b_s_own_clause():
    text = _text(a_collapses=True, b_collapses=True, b_exceeds_a=True)
    # All three clauses on one row: B can collapse and still beat A.
    assert "A collapses, B collapses, B > A" in text


def test_the_condition_is_read_even_when_the_endpoint_confirms():
    """Confirmation does not suppress it. A and B can both collapse with B
    collapsing more slowly, which confirms endpoint 2's contrast and falsifies
    section 2's mechanism at the same time -- and the paper owes both.
    """

    text = _text(a_collapses=True, b_collapses=True, b_exceeds_a=True)
    assert "CONFIRMED across seeds" in text
    assert "FAILURE CONDITION 2 HOLDS" in text


def _collapsing_run(tmp_path: Path, seed: int, *, gaps_a: list[float],
                    gaps_b: list[float], steps=(0, 8, 16, 24)) -> Path:
    """A whole replicate: six prompts, half labelled right, both arms.

    The right half scores `gap` above the wrong half at each step, so each
    arm's blind gap follows the list it was given and its fitted slope is the
    trend in that list. Enough for `main()` to reach the payload, which is the
    only place `failure_condition_2` is written.
    """

    run = tmp_path / f"e3-s{seed}"
    prompts = [f"p{index:02d}" for index in range(6)]
    for arm, gaps in (("naive", gaps_a), ("blind_self", gaps_b)):
        for column, step in enumerate(steps):
            directory = run / "evaluations" / arm / f"step-{step:05d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "verified.jsonl").write_text(
                "".join(json.dumps({"spec_id": prompt, "candidate_index": 0,
                                    "image_correct": index < 3,
                                    "resolution": "agreed"}) + "\n"
                        for index, prompt in enumerate(prompts)), encoding="utf-8")
            blind = run / "analysis" / "blind_observe" / arm
            blind.mkdir(parents=True, exist_ok=True)
            (blind / f"step-{step:05d}.jsonl").write_text(
                "".join(json.dumps({
                    "prompt_id": prompt, "candidate_index": 0,
                    "condition": "image_only",
                    # A per-prompt tilt so the bootstrap has spread to find.
                    "s_select": (gaps[column] * (0.6 + 0.2 * index) if index < 3 else 0.0),
                }) + "\n" for index, prompt in enumerate(prompts)), encoding="utf-8")
    return run


def _run_main(monkeypatch, tmp_path: Path, runs: list[Path]) -> dict:
    outdir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "v4_e3_endpoint2.py", "--outdir", str(outdir), "--resamples", "300",
        "--runs", *[str(run) for run in runs]])
    assert driver.main() == 0
    return json.loads((outdir / "endpoint2.json").read_text(encoding="utf-8"))


def test_the_payload_records_failure_condition_2_when_both_arms_collapse(
        monkeypatch, tmp_path: Path):
    both = [0.60, 0.40, 0.20, 0.00]
    runs = [_collapsing_run(tmp_path, seed, gaps_a=both, gaps_b=both)
            for seed in driver.REGISTERED_SEEDS]
    payload = _run_main(monkeypatch, tmp_path, runs)
    assert payload["across_seeds"]["b_collapsing"] == 5
    assert payload["failure_condition_2"]["holds"] is True
    assert payload["failure_condition_2"]["wording"] == FALSIFIED


def test_the_payload_leaves_it_unset_when_b_holds(monkeypatch, tmp_path: Path):
    runs = [_collapsing_run(tmp_path, seed, gaps_a=[0.60, 0.40, 0.20, 0.00],
                            gaps_b=[0.55, 0.58, 0.61, 0.64])
            for seed in driver.REGISTERED_SEEDS]
    payload = _run_main(monkeypatch, tmp_path, runs)
    assert payload["across_seeds"]["b_collapsing"] == 0
    assert payload["failure_condition_2"]["holds"] is False
    assert payload["failure_condition_2"]["wording"] is None
