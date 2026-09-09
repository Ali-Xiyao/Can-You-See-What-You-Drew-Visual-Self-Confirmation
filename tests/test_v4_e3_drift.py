"""Tests for the drift driver's two gates and its refusal to aggregate.

The arithmetic is tested in `tests/test_drift.py`. What is tested here is what
the driver decides around it: whether deviation 7.2 is in force at all
(deviation 14.6, read out of `endpoint1.json` and never recomputed), and what
it does with five verdicts when no rule was registered for combining them
(deviation 14.3). Both are places where the tempting behaviour and the
registered behaviour differ, which is the only reason a driver needs tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from selfsight.analysis.drift import NOT_SEPARABLE
from selfsight.analysis.endpoint1 import load_checkpoint, paired_difference

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_drift.py"

# A real split.json opens with a wall clock stamp. It is not part of the
# split identity and the guard excludes it, but the fixture carries one so
# that stays a checked fact rather than a fixture that never had the field.
SPLIT = {"created": "2026-09-08T14:33:14Z",
         "train": ["p1", "p2"], "outcome": ["p3", "p4"]}
KEYS = (("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1))
STEPS = (0, 8, 16)


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_drift", SCRIPT)
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


def _run_dir(tmp_path: Path, name: str) -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "split.json").write_text(json.dumps(SPLIT), encoding="utf-8")
    return run


def _main_run(tmp_path: Path, values=(True, False, False, False)) -> Path:
    run = _run_dir(tmp_path, "main")
    for step in STEPS:
        _checkpoint(run, "naive", step, values)
    return run


def _replicate(tmp_path: Path, seed: int, *, naive=(True, False, False, False),
               blind=(True, True, True, False)) -> Path:
    """One replicate whose last step carries the verdict and whose rounds vary.

    Step 0 gets both arms the same, which is what the runs actually produce and
    what deviation 14.5 excludes from the round bootstrap.
    """

    run = _run_dir(tmp_path, f"e3-s{seed}")
    _checkpoint(run, "naive", 0, naive)
    _checkpoint(run, "blind_self", 0, naive)
    _checkpoint(run, "naive", 8, naive)
    _checkpoint(run, "blind_self", 8, (True, True, False, False))
    _checkpoint(run, "naive", 16, naive)
    _checkpoint(run, "blind_self", 16, blind)
    return run


# theta_A = theta_A' = 0.25 so the drift is 0; theta_B = 0.75 clears it.
SEPARABLE_BLIND = (True, True, True, False)
# theta_A' = 1.0 puts the drift at 0.75 and leaves no effect to clear it.
FLAT = (True, True, True, True)


def _endpoint1(tmp_path: Path, *, confirmed: bool = True,
               detected: dict[int, bool] | None = None,
               status: str = "done") -> Path:
    path = tmp_path / "endpoint1.json"
    # "endpoint" is not decoration: the gate reads it to refuse endpoint 2's or
    # 3's file, and a fixture without it would let that guard be deleted.
    payload = {"endpoint": "E3 endpoint 1: final external correctness, B > A",
               "status": status, "sign_test": {"confirmed": confirmed},
               "seeds": [{"seed": seed, "verdict": {"detected": value}}
                         for seed, value in (detected or {}).items()]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _invoke(monkeypatch, tmp_path: Path, runs: list[Path], endpoint1: Path,
            main: Path) -> tuple[int, dict, str]:
    outdir = tmp_path / "out"
    argv = ["v4_e3_drift.py", "--main", str(main), "--endpoint1", str(endpoint1),
            "--outdir", str(outdir), "--resamples", "200", "--runs",
            *[str(run) for run in runs]]
    monkeypatch.setattr(sys, "argv", argv)
    code = driver.main()
    payload = json.loads((outdir / "drift.json").read_text(encoding="utf-8"))
    return code, payload, (outdir / "drift.txt").read_text(encoding="utf-8")


# --- deviation 14.6, the study-level gate ------------------------------------


def test_a_missing_endpoint_1_puts_the_rule_out_of_force(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    code, payload, text = _invoke(monkeypatch, tmp_path, [run],
                                  tmp_path / "absent.json", main)
    # The literal 2, not the constant: the supervisor branches on the exit
    # code, and "out of force" has to be distinguishable from "done".
    assert code == 2 == driver.OUT_OF_FORCE
    assert payload["status"] == "out_of_force"
    assert "replicates" not in payload
    assert "OUT OF FORCE" in text


def test_an_unfinished_endpoint_1_puts_the_rule_out_of_force(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, status="not_done", detected={20260906: True})
    code, payload, _ = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    assert code == driver.OUT_OF_FORCE
    assert "not done" in payload["reason"]


def test_a_sign_test_that_did_not_confirm_puts_the_rule_out_of_force(
        monkeypatch, tmp_path: Path):
    """Failure condition 1 has already taken over; 7.2 has nothing left to say.

    This is the branch worth pinning, because the per-replicate numbers exist
    and would print perfectly well. The rule says not to look at them.
    """

    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, confirmed=False, detected={20260906: True})
    code, payload, text = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    assert code == driver.OUT_OF_FORCE
    assert "sign test did not confirm" in payload["reason"]
    assert "separable" not in text


def test_the_gate_is_read_and_not_recomputed(tmp_path: Path):
    # Deviation 14.6 names two layers of endpoint1.json and says to read both.
    # Deriving `confirmed` from the per-seed flags here would be a second
    # implementation of the sign test, free to disagree with the first.
    endpoint1 = _endpoint1(tmp_path, confirmed=True,
                           detected={20260906: False, 20260907: False})
    confirmed, detected = driver.endpoint1_gate(endpoint1)
    assert confirmed is True
    assert detected == {20260906: False, 20260907: False}


def test_a_replicate_absent_from_endpoint_1_is_not_assumed_detected(
        monkeypatch, tmp_path: Path):
    # A seed endpoint 1 never listed has not been judged detected, and the
    # default for "not judged" is the same as for "judged and missed".
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, detected={20260907: True})
    code, payload, _ = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    assert code == 0
    assert payload["replicates"][0]["separable"] is None


def test_a_replicate_whose_endpoint_1_missed_gets_no_verdict(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, detected={20260906: False})
    code, payload, text = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    assert code == 0
    row = payload["replicates"][0]
    assert row["separable"] is None
    assert row["wording"] == driver.NOT_APPLICABLE
    assert "0 of 0 applicable replicates" in text


# --- deviation 14.3, the count nobody registered a rule for -------------------


def _two(monkeypatch, tmp_path: Path, *, second_blind, second_naive):
    main = _main_run(tmp_path)
    first = _replicate(tmp_path, 20260906, blind=SEPARABLE_BLIND)
    second = _replicate(tmp_path, 20260907, naive=second_naive, blind=second_blind)
    endpoint1 = _endpoint1(tmp_path, detected={20260906: True, 20260907: True})
    return _invoke(monkeypatch, tmp_path, [first, second], endpoint1, main)


def test_the_count_is_reported_and_not_tested(monkeypatch, tmp_path: Path):
    code, payload, text = _two(monkeypatch, tmp_path,
                               second_naive=FLAT, second_blind=FLAT)
    assert code == 0
    assert [row["separable"] for row in payload["replicates"]] == [True, False]
    assert "1 of 2 applicable replicates" in text
    assert "No aggregate rule is registered" in text
    assert "none registered" in payload["aggregate_rule"]


def test_disagreeing_replicates_force_the_inconsistency_sentence(
        monkeypatch, tmp_path: Path):
    _, _, text = _two(monkeypatch, tmp_path, second_naive=FLAT, second_blind=FLAT)
    assert "inconsistent across" in text
    assert NOT_SEPARABLE in text


def test_agreeing_replicates_do_not_get_the_inconsistency_sentence(
        monkeypatch, tmp_path: Path):
    _, payload, text = _two(monkeypatch, tmp_path,
                            second_naive=(True, False, False, False),
                            second_blind=SEPARABLE_BLIND)
    assert [row["separable"] for row in payload["replicates"]] == [True, True]
    assert "2 of 2 applicable replicates" in text
    assert "inconsistent across" not in text


def test_a_b_below_a_prime_is_reported_signed_and_not_separable(
        monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906, naive=FLAT, blind=(True, False, False, False))
    endpoint1 = _endpoint1(tmp_path, detected={20260906: True})
    _, payload, text = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    row = payload["replicates"][0]
    assert row["effect"] == pytest.approx(-0.75)
    assert row["separable"] is False
    assert "-0.7500" in text


# --- deviation 14.6b, the asymmetry with endpoints 2 and 3 --------------------


def test_fewer_than_five_replicates_is_exploratory_not_not_done(
        monkeypatch, tmp_path: Path):
    """Endpoints 2 and 3 write "not done" when a seed is missing. 7.2 does not.

    A replicate gives an upper bound; four give four upper bounds and no
    registered test has been interrupted. Deviation 14.6b writes the reason for
    the asymmetry down so it does not read as a double standard later.
    """

    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, detected={20260906: True})
    code, payload, text = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    assert code == 0
    assert payload["status"] == "done"
    assert payload["confirmatory"] is False
    assert "Exploratory only" in text


def test_the_registered_five_are_confirmatory(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    runs = [_replicate(tmp_path, seed) for seed in driver.REGISTERED_SEEDS]
    endpoint1 = _endpoint1(tmp_path, detected=dict.fromkeys(driver.REGISTERED_SEEDS, True))
    _, payload, text = _invoke(monkeypatch, tmp_path, runs, endpoint1, main)
    assert payload["confirmatory"] is True
    assert "Exploratory only" not in text


def test_no_replicate_directory_at_all_is_refused(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    endpoint1 = _endpoint1(tmp_path, detected={20260906: True})
    monkeypatch.setattr(sys, "argv", [
        "v4_e3_drift.py", "--main", str(main), "--endpoint1", str(endpoint1),
        "--outdir", str(tmp_path / "out"), "--runs", str(tmp_path / "e3-s20269999")])
    with pytest.raises(SystemExit, match="No replicate run directories"):
        driver.main()


# --- deviation 7.3, the round axis -------------------------------------------


def test_the_round_axis_excludes_step_zero(monkeypatch, tmp_path: Path):
    main = _main_run(tmp_path)
    run = _replicate(tmp_path, 20260906)
    endpoint1 = _endpoint1(tmp_path, detected={20260906: True})
    _, payload, text = _invoke(monkeypatch, tmp_path, [run], endpoint1, main)
    rounds = payload["replicates"][0]["rounds"]
    assert rounds["steps"] == [8, 16]
    assert rounds["excluded_step"] == 0
    assert rounds["seed"] == 20260908
    assert "no failure condition" in text


def test_the_round_differences_are_endpoint_1_s(tmp_path: Path):
    # The round axis and endpoint 1 report the same quantity at the same
    # checkpoints; computing it twice is only safe if it is computed once.
    run = _replicate(tmp_path, 20260906)
    # The B arm stops a checkpoint short, which is what an interrupted run
    # looks like. The round axis is the steps both arms reached, not either.
    for path in (run / "evaluations" / "blind_self" / "step-00016").iterdir():
        path.unlink()
    (run / "evaluations" / "blind_self" / "step-00016").rmdir()
    found = driver.round_differences(run, "naive", "blind_self")
    assert sorted(found) == [0, 8]
    for step, value in found.items():
        assert value == pytest.approx(
            paired_difference(load_checkpoint(run, step, "naive", "blind_self")))


# --- the two drivers against each other ------------------------------------
#
# Everything above builds endpoint1.json by hand, which pins what this driver
# expects rather than what endpoint 1 writes. Two defects found on 2026-09-09
# lived in exactly that gap: a launch gate reading a file nothing wrote, and
# this module's split guard hashing a timestamp. So the tests below run the
# real endpoint 1 driver and hand its actual output to the gate.

_e1_spec = importlib.util.spec_from_file_location(
    "v4_e3_endpoint1_for_drift", ROOT / "scripts" / "v4_e3_endpoint1.py")
endpoint1_driver = importlib.util.module_from_spec(_e1_spec)
assert _e1_spec.loader is not None
_e1_spec.loader.exec_module(endpoint1_driver)


def test_endpoint_1s_real_output_satisfies_the_gate(monkeypatch, tmp_path: Path):
    """Run endpoint 1 for real, then read its file with this driver's gate.

    The hand-built fixture above says `status: done`; endpoint 1 writes no
    `status` at all. That is harmless -- the gate uses `.get` -- but it is
    only known to be harmless because something looked at the real payload.
    The keys that matter, `sign_test.confirmed` and `seeds[].verdict.detected`,
    have no other check that the two drivers agree on where they live.
    """

    runs = []
    for index, seed in enumerate(endpoint1_driver.REGISTERED_SEEDS):
        run = _run_dir(tmp_path, f"e3-s{seed}")
        # Five real replicates split at five different moments.
        (run / "split.json").write_text(
            json.dumps({**SPLIT, "created": f"2026-09-1{index}T02:07:55Z"}),
            encoding="utf-8")
        for step in STEPS:
            _checkpoint(run, "naive", step, (True, False, False, False))
            _checkpoint(run, "blind_self", step,
                        (True, False, False, False) if step == 0
                        else (True, True, True, True))
        runs.append(run)

    outdir = tmp_path / "e1-out"
    monkeypatch.setattr(sys, "argv",
                        ["v4_e3_endpoint1.py", "--outdir", str(outdir),
                         "--resamples", "200",
                         "--runs", *[str(run) for run in runs]])
    endpoint1_driver.main()

    written = outdir / "endpoint1.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert "confirmed" in payload["sign_test"]
    assert payload["seeds"] and all("seed" in row and "detected" in row["verdict"]
                                    for row in payload["seeds"])

    confirmed, detected = driver.endpoint1_gate(written)
    assert confirmed is True
    assert set(detected) == set(endpoint1_driver.REGISTERED_SEEDS)
    assert all(detected.values())


def test_another_endpoints_verdict_is_refused_rather_than_out_of_force(tmp_path: Path):
    """Endpoint 2's file handed to endpoint 1's flag must not exit 2.

    Out of force is a registered outcome: deviation 14.6 says 7.2 does not
    apply when endpoint 1 is not detected, and the driver exits 2 to say so.
    An operator who passed the wrong path would otherwise get that same exit,
    and the record would show a rule that was evaluated and found not to
    apply rather than one nobody looked at. The three files sit side by side
    under review-packets/ with names one character apart.
    """

    path = tmp_path / "endpoint2.json"
    path.write_text(json.dumps({
        "endpoint": "E3 endpoint 2: blind discrimination gap vs step",
        "status": "done"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not endpoint 1"):
        driver.endpoint1_gate(path)


def test_a_file_that_names_no_endpoint_is_refused(tmp_path: Path):
    path = tmp_path / "endpoint1.json"
    path.write_text(json.dumps({"sign_test": {"confirmed": True}, "seeds": []}),
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="not endpoint 1"):
        driver.endpoint1_gate(path)
