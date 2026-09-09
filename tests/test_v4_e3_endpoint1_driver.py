"""Tests for the endpoint 1 driver, which had none until deviation 15.2.

The arithmetic is tested in `tests/test_endpoint1.py`. The driver was left
untested because it looked like formatting, and then deviation 15.2 put a
registered sentence in it -- the consequence of failure condition 1, which
decides what the paper is allowed to claim. That is not formatting.

What is pinned here is the layer the sentence reads (deviation 14.6: study
level, `sign_test.confirmed`, not any one seed's `detected`), that it fires on
an exploratory seed set as well, and that a confirmed study does not print it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from selfsight.analysis.endpoint1 import NOT_DETECTED_CONSEQUENCE

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_endpoint1.py"

STEPS = (0, 8)
KEYS = tuple((f"p{index}", draw) for index in range(4) for draw in range(2))


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_endpoint1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _module()


def _checkpoint(run: Path, arm: str, step: int, values: tuple[bool, ...]) -> None:
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "verified.jsonl").write_text(
        "".join(json.dumps({"spec_id": spec_id, "candidate_index": index,
                            "image_correct": value, "resolution": "agreed"}) + "\n"
                for (spec_id, index), value in zip(KEYS, values)), encoding="utf-8")
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps({"spec_id": spec_id, "candidate_index": index}) + "\n"
                for spec_id, index in KEYS), encoding="utf-8")


def _run(tmp_path: Path, seed: int, *, b_leads: bool) -> Path:
    """A replicate whose B arm beats A, or does not, by a wide margin.

    Eight (prompt, draw) keys over four prompts. B leading everywhere puts the
    paired difference well above the registered 0.042; B trailing everywhere
    puts it well below zero. Endpoint 1's own tests cover the boundary -- what
    matters here is which branch of the driver runs.
    """

    run = tmp_path / f"e3-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    a = (True, False, False, False, False, False, False, False)
    b = (True, True, True, True, True, True, False, False) if b_leads else (False,) * 8
    for step in STEPS:
        _checkpoint(run, "naive", step, a)
        _checkpoint(run, "blind_self", step, b)
    return run


def _invoke(monkeypatch, tmp_path: Path, runs: list[Path],
            *, partial: bool = True) -> tuple[dict, str]:
    outdir = tmp_path / "out"
    argv = ["v4_e3_endpoint1.py", "--outdir", str(outdir), "--resamples", "200",
            "--runs", *[str(run) for run in runs]]
    if partial:
        argv.append("--allow-partial")
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    return (json.loads((outdir / "endpoint1.json").read_text(encoding="utf-8")),
            (outdir / "endpoint1.txt").read_text(encoding="utf-8"))


def test_the_consequence_is_the_registered_sentence():
    # The clause that limits the claim is the whole point of failure condition
    # 1; a wording that keeps the shape and drops "may not" would print, read
    # fine, and license the paper to say the opposite.
    assert "may not claim that BSV improves generation" in NOT_DETECTED_CONSEQUENCE
    assert "improves the quality of the selection signal" in NOT_DETECTED_CONSEQUENCE
    assert "retreats to inference time" in NOT_DETECTED_CONSEQUENCE
    assert "failure condition 1" in NOT_DETECTED_CONSEQUENCE


def test_a_confirmed_study_does_not_print_the_consequence(monkeypatch, tmp_path: Path):
    runs = [_run(tmp_path, seed, b_leads=True) for seed in driver.REGISTERED_SEEDS]
    payload, text = _invoke(monkeypatch, tmp_path, runs, partial=False)
    assert payload["sign_test"]["confirmed"] is True
    assert payload["failure_condition_1"]["holds"] is False
    assert payload["failure_condition_1"]["wording"] is None
    assert "not triggered" in text
    assert NOT_DETECTED_CONSEQUENCE not in text


def test_a_study_the_sign_test_did_not_confirm_prints_it(monkeypatch, tmp_path: Path):
    runs = [_run(tmp_path, seed, b_leads=False) for seed in driver.REGISTERED_SEEDS]
    payload, text = _invoke(monkeypatch, tmp_path, runs, partial=False)
    assert payload["sign_test"]["confirmed"] is False
    assert payload["failure_condition_1"]["holds"] is True
    assert payload["failure_condition_1"]["wording"] == NOT_DETECTED_CONSEQUENCE
    assert NOT_DETECTED_CONSEQUENCE in text


def test_the_consequence_reads_the_study_level_not_a_seed(monkeypatch, tmp_path: Path):
    """Four seeds detect and one does not, so the sign test fails and the
    consequence fires -- even though four of the five rows above it say
    detected. Deviation 14.6 fixes which of the two layers decides, and
    reading the per-seed flags here would give the opposite answer four times
    out of five.
    """

    runs = [_run(tmp_path, seed, b_leads=True) for seed in driver.REGISTERED_SEEDS[:4]]
    runs.append(_run(tmp_path, driver.REGISTERED_SEEDS[4], b_leads=False))
    payload, text = _invoke(monkeypatch, tmp_path, runs, partial=False)
    assert [seed["verdict"]["detected"] for seed in payload["seeds"]].count(True) == 4
    assert payload["sign_test"]["confirmed"] is False
    assert payload["failure_condition_1"]["holds"] is True
    assert "study level" in payload["failure_condition_1"]["read_at"]
    assert NOT_DETECTED_CONSEQUENCE in text


def test_an_exploratory_seed_set_still_triggers_it(monkeypatch, tmp_path: Path):
    """`confirmatory` gates the positive claim; it must not gate the downgrade.

    Two seeds both leading is not a confirmed endpoint 1, so the paper is not
    entitled to the generation claim, and the sentence saying so has to print.
    Suppressing it until five seeds exist would let an interrupted study look
    like one with no verdict rather than one with an unfavourable one.
    """

    runs = [_run(tmp_path, seed, b_leads=True) for seed in driver.REGISTERED_SEEDS[:2]]
    payload, text = _invoke(monkeypatch, tmp_path, runs)
    assert payload["confirmatory"] is False
    assert payload["sign_test"]["confirmed"] is False
    assert payload["failure_condition_1"]["holds"] is True
    assert NOT_DETECTED_CONSEQUENCE in text


def test_a_partial_seed_set_is_refused_without_the_flag(monkeypatch, tmp_path: Path):
    runs = [_run(tmp_path, seed, b_leads=True) for seed in driver.REGISTERED_SEEDS[:2]]
    with pytest.raises(SystemExit, match="allow-partial"):
        _invoke(monkeypatch, tmp_path, runs, partial=False)


def test_no_run_directory_at_all_is_refused(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", [
        "v4_e3_endpoint1.py", "--outdir", str(tmp_path / "out"),
        "--runs", str(tmp_path / "e3-s20269999")])
    with pytest.raises(SystemExit, match="No replicate run directories"):
        driver.main()
