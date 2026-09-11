"""The supervisor runs the two arms at once now, so the orchestration is tested.

Everything here is about `Pilot.chains`, which is the only part of the run that
became concurrent. The stages it launches are unchanged and already covered by
the tests for the scripts they call; what is new, and what a multi-day
unattended run would suffer from silently, is a chain that does not actually
overlap, a failure that gets swallowed, or an arm that lands on the wrong card.
"""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_decoupling_pilot", ROOT / "scripts" / "run_decoupling_pilot.py")
supervisor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supervisor)


class _Bare:
    """A Pilot with none of its state: chains touches no instance attributes."""

    chains = supervisor.Pilot.chains


# ------------------------------------------------------------------ overlap


def test_the_two_arm_chains_actually_run_at_the_same_time():
    """Sequential execution would still pass every other test in this file."""

    entered = threading.Barrier(2, timeout=5.0)
    order = []

    def body(name):
        def run():
            # Times out and raises BrokenBarrierError if the other chain is not
            # started yet, which is exactly what serial execution looks like.
            entered.wait()
            order.append(name)
        return run

    _Bare().chains({"naive": body("naive"), "rfo_gold": body("rfo_gold")})
    assert sorted(order) == ["naive", "rfo_gold"]


def test_a_slow_chain_does_not_delay_the_start_of_the_other():
    started = {}

    def slow():
        started["slow"] = time.monotonic()
        time.sleep(0.4)

    def quick():
        started["quick"] = time.monotonic()

    _Bare().chains({"a_slow": slow, "b_quick": quick})
    assert abs(started["quick"] - started["slow"]) < 0.2


# ------------------------------------------------------------------ failure


def test_one_chain_failing_is_re_raised_rather_than_swallowed():
    def boom():
        raise RuntimeError("generate died")

    with pytest.raises(RuntimeError, match="naive chain failed: generate died"):
        _Bare().chains({"naive": boom, "rfo_gold": lambda: None})


def test_the_surviving_chain_finishes_its_stage_instead_of_being_killed():
    """A half-written manifest costs more than the minutes an early exit saves."""

    finished = []

    def boom():
        raise RuntimeError("detector died")

    def slow():
        time.sleep(0.3)
        finished.append("rfo_gold")

    with pytest.raises(RuntimeError, match="chain failed"):
        _Bare().chains({"naive": boom, "rfo_gold": slow})
    assert finished == ["rfo_gold"], "the other arm was cut off mid-stage"


def test_both_chains_failing_still_names_one_and_keeps_its_cause():
    def boom(message):
        def run():
            raise ValueError(message)
        return run

    with pytest.raises(RuntimeError) as caught:
        _Bare().chains({"naive": boom("first"), "rfo_gold": boom("second")})
    assert "naive chain failed" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


# ------------------------------------------------------------------- cards


def test_each_arm_owns_a_different_card_and_every_arm_has_one():
    assert sorted(supervisor.ARM_DEVICE) == sorted(supervisor.ARMS)
    assert len(set(supervisor.ARM_DEVICE.values())) == len(supervisor.ARMS), \
        "two arms on one card would serialise the pass this change exists to overlap"


def test_no_evaluation_stage_still_hard_codes_a_card():
    """A missed cuda:0 would quietly pin both arms to the same GPU again."""

    source = (ROOT / "scripts" / "run_decoupling_pilot.py").read_text(encoding="utf-8")
    body = source[source.index("def adjudicate"):source.index("def report")]
    assert "cuda:0" not in body and "cuda:1" not in body, \
        "adjudicate/evaluate/probe must take the card from ARM_DEVICE"


# ------------------------------------------------------- publishing progress


def test_two_threads_publishing_state_never_leave_a_corrupt_file_behind(tmp_path):
    """The shared .tmp path let two writers interleave into one buffer.

    Measured against the previous implementation under this same hammering, in
    three trials: 340 writer failures, 8 reads of malformed JSON and 3 orphaned
    temporary files. What is asserted here is what the per-thread tmp name plus
    the bounded replace retry actually fix, and not more: a reader on Windows
    can still lose an occasional open() to a replace in flight, in this version
    and in any version, so reader PermissionError is counted and ignored.
    """

    target = tmp_path / "state.json"
    writes = 120
    write_failures = []
    corrupt_reads = []
    good_reads = []

    def write(tag):
        for index in range(writes // 2):
            try:
                supervisor.write_json(target, {"who": tag, "index": index})
            except OSError as exc:
                write_failures.append(exc)

    def read():
        for _ in range(writes):
            if not target.exists():
                continue
            try:
                good_reads.append(json.loads(target.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                corrupt_reads.append(exc)
            except OSError:
                pass          # inherent on Windows, and harmless: readers retry

    threads = [threading.Thread(target=write, args=("naive",)),
               threading.Thread(target=write, args=("rfo_gold",)),
               threading.Thread(target=read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert good_reads, "the reader never saw the file, so this proved nothing"
    assert not corrupt_reads, f"reader saw malformed JSON: {corrupt_reads[:2]}"
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was orphaned"
    # Generous: the point is that this collapsed from ~340, not that it is zero.
    assert len(write_failures) <= 5, f"{len(write_failures)} of {writes} writes failed"


def test_a_failed_status_write_does_not_take_the_run_down_with_it(monkeypatch, tmp_path):
    """state.json is telemetry; evidence is written by the stages themselves."""

    runner = object.__new__(supervisor.Pilot)
    runner.out = tmp_path
    runner.started = 0.0
    runner.limit = 1

    def refuse(*_args, **_kwargs):
        raise PermissionError("held open by a reader")

    monkeypatch.setattr(runner, "_write_state", refuse)
    runner.state("running", "naive.step-00000.generate")   # must not raise


def test_an_evidence_write_still_raises_when_it_cannot_publish(monkeypatch, tmp_path):
    """Only the status file is allowed to fail quietly."""

    monkeypatch.setattr(supervisor.os, "replace",
                        lambda *_: (_ for _ in ()).throw(PermissionError("locked")))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        supervisor.write_json(tmp_path / "stage-completion" / "x.json", {"stage": "x"})
    assert not list((tmp_path / "stage-completion").glob("*.tmp")), "tmp left behind"
