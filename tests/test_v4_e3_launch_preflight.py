"""The launch gate that reads state.json, and the three ways a run can end.

This gate decides whether 400 GPU-hours of replicates start. It runs
unattended at whatever hour the main run finishes, and its output is read by
someone who was asleep, so a wrong *explanation* costs nearly as much as a
wrong verdict: an operator who is told the 96 h line fired when it did not
goes looking in the wrong place, and an operator told a dead run is "still
running" goes back to bed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_e3_launch_preflight", ROOT / "scripts" / "v4_e3_launch_preflight.py")
preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)


def write_state(tmp_path: Path, **fields) -> Path:
    """A state.json with the keys run_decoupling_pilot.py actually writes."""

    path = tmp_path / "state.json"
    payload = {"supervisor_pid": 4242, "updated_unix": 1788977661.0,
               "started_unix": 1788877993.0, "elapsed_hours": 27.7, "through_round": 11}
    payload.update(fields)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_finished_run_passes(tmp_path):
    path = write_state(tmp_path, status="pilot_complete", completed_rounds=11,
                       stage="review_results")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is True
    assert "11 rounds" in message


def test_a_bounded_run_is_not_called_a_wall_clock_stop(tmp_path):
    """canary_complete means --through-round, not the 96 h line.

    run() raises RuntimeError when the deadline passes, and that exception
    leaves the round loop above the terminal state() call, so the 96 h case
    never writes canary_complete at all -- it writes nothing. The status is
    reached only by `self.limit != config["training"]["rounds"]`, which is
    the --through-round flag. STATUS 43.15 recorded this correctly for the
    2026-09-06 pilot resume; the first draft of this gate did not.
    """

    path = write_state(tmp_path, status="canary_complete", completed_rounds=3,
                       stage="review_results")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "through-round" in message
    lowered = message.lower()
    assert "96" not in lowered and "wall" not in lowered and "stop line" not in lowered


def test_a_run_whose_supervisor_is_gone_is_not_reported_as_still_running(tmp_path,
                                                                        monkeypatch):
    """This is the shape the 96 h line really leaves behind.

    Nothing rewrites state.json after run() raises, so the file says running
    for as long as the disk lasts. Reading that as "still going" is how an
    operator waits all night for a run that died before midnight.
    """

    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: False)
    path = write_state(tmp_path, status="running", stage="naive.step-00024.detect.qwen3vl")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "4242" in message
    assert "without writing a terminal status" in message


def test_a_live_run_says_which_stage_it_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: True)
    path = write_state(tmp_path, status="running", stage="naive.step-00024.detect.qwen3vl")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "naive.step-00024.detect.qwen3vl" in message
    assert "still running" in message


def test_an_unknown_liveness_answer_falls_back_to_the_ordinary_wording(tmp_path,
                                                                      monkeypatch):
    """None is "cannot tell", and it must not read as "dead"."""

    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: None)
    path = write_state(tmp_path, status="running", stage="round-003.train")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "still running" in message


def test_liveness_never_uses_os_kill():
    """os.kill(pid, 0) is not a liveness probe on Windows.

    CPython implements os.kill on Windows by opening the process and calling
    TerminateProcess for any signal that is not a console-control event, so
    the portable idiom would kill the multi-day run this gate protects. The
    hazard is invisible at the call site, which is why it is pinned here.
    """

    tree = ast.parse((ROOT / "scripts" / "v4_e3_launch_preflight.py").read_text(
        encoding="utf-8"))
    # Parsed, not grepped: the docstring that explains the hazard says
    # os.kill, and a substring search cannot tell an explanation from a call.
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "kill"]
    assert calls == [], f"os.kill called at line(s) {[node.lineno for node in calls]}"
    imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "os" and any(a.name == "kill" for a in node.names)]
    assert imported == []


@pytest.mark.parametrize("fields", [
    {"status": "pilot_complete", "completed_rounds": 10},
    {"status": "pilot_complete"},
    {"status": "canary_complete", "completed_rounds": 11},
    {"status": "running", "completed_rounds": 11},
    {"status": "failed", "completed_rounds": 11},
    {"status": None, "completed_rounds": 11},
])
def test_only_eleven_completed_rounds_under_pilot_complete_passes(tmp_path, monkeypatch,
                                                                  fields):
    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: True)
    passed, _ = preflight.gate_main_run_finished(write_state(tmp_path, **fields))
    assert passed is False


def test_an_absent_state_file_is_refused_by_path(tmp_path):
    passed, message = preflight.gate_main_run_finished(tmp_path / "state.json")
    assert passed is False
    assert "state.json" in message


# --- gate_card_schedule_decided -------------------------------------------
#
# This gate had no test, which is how it came to read a file and a key that
# nothing writes. `card_benchmark.py` lives in the arm B branch's review
# packet and this preflight lives here; until the merge rehearsal of
# 2026-09-09 the two had never been in one tree, so no run of anything could
# have noticed that the gate wanted `verdict.json["serialise_detect"]` and
# the benchmark wrote `card_benchmark.json["verdict"]["adopt_serial"]`.
#
# The fixture below is therefore built from the benchmark's own payload keys
# rather than from the gate's expectations, which is the only ordering that
# would have caught the original defect.

PACKET = "review-packets/card-scheduling-20260909"


def write_benchmark(root: Path, **fields) -> Path:
    """The shape `card_benchmark.py:main` writes, minus the timing detail."""

    payload = {"images": 64, "manifest": "runs/v4/.../manifest.jsonl",
               "no_load": True, "margin": 0.90,
               "results": [{"detector": "internvl"}, {"detector": "qwen3vl"}],
               "verdict": {"adopt_serial": True,
                           "reason": "serial clears the 10% margin on every detector"}}
    payload.update(fields)
    packet = root / PACKET
    packet.mkdir(parents=True, exist_ok=True)
    path = packet / "card_benchmark.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_the_gate_reads_the_file_the_benchmark_writes(tmp_path):
    write_benchmark(tmp_path)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is True
    assert "adopt_serial=True" in message
    # The reason travels with the decision: an operator reading this at four
    # in the morning should not have to open the JSON to learn which way the
    # criterion fell or why.
    assert "every detector" in message


def test_a_verdict_against_serialising_is_still_a_decision(tmp_path):
    """`adopt_serial: False` passes the gate. It is a decision, not a failure.

    The gate exists so the criterion cannot be skipped, not so it comes out
    a particular way -- the registered tie-break (keep the per-arm pinning)
    is one of its two legitimate answers.
    """

    write_benchmark(tmp_path, verdict={"adopt_serial": False,
                                       "reason": "serial clears the margin on "
                                                 "['internvl'], not on all"})
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is True
    assert "adopt_serial=False" in message


def test_a_missing_benchmark_names_the_file_and_the_outdir(tmp_path):
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    # Naming the outdir is the point: the benchmark takes --outdir as a
    # required argument and will happily write a correct answer somewhere
    # this gate does not look.
    assert "card_benchmark.json" in message
    assert "--outdir" in message and PACKET.replace("/", "\\") in message.replace("/", "\\")


def test_the_old_verdict_json_no_longer_satisfies_the_gate(tmp_path):
    """A hand-written file with the old key must not pass.

    The old key was invented by this gate and appears in no pre-registration
    and in no other script. Leaving it as an accepted alternative would keep
    open exactly the path the defect created: a decision typed by hand at the
    moment the measurement is inconvenient.
    """

    packet = tmp_path / PACKET
    packet.mkdir(parents=True)
    (packet / "verdict.json").write_text(json.dumps({"serialise_detect": True}),
                                         encoding="utf-8")
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "card_benchmark.json" in message


def test_a_benchmark_with_no_decision_is_refused(tmp_path):
    write_benchmark(tmp_path, verdict={})
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "adopt_serial" in message


def test_a_busy_card_measurement_is_refused(tmp_path):
    """`--allow-busy-cards` records `no_load: false` and the gate must read it.

    Section 0.3 is a no-load benchmark; the numbers in it that are already
    known to be contaminated by another process are the reason it exists.
    """

    write_benchmark(tmp_path, no_load=False)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "no_load" in message


def test_a_margin_other_than_the_registered_one_is_refused(tmp_path):
    write_benchmark(tmp_path, margin=0.95)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "0.9" in message


def test_the_registered_margin_is_the_number_section_0_3_fixed():
    # 0.90, fixed before the measurement. The gate does not re-derive the
    # criterion -- card_benchmark.py owns that -- but it does refuse a report
    # produced under a different one.
    assert preflight.CARD_MARGIN == 0.90


def test_a_report_with_no_no_load_key_is_refused(tmp_path):
    """Absent is not the same as true, and the safe default is refuse.

    `card_benchmark.py` always writes the key, so a report without it did
    not come from the benchmark as it stands -- an older copy, a hand-edited
    file, or something else entirely. Every one of those is a reason to look
    rather than to assume the cards were empty.
    """

    payload = json.loads(write_benchmark(tmp_path).read_text(encoding="utf-8"))
    del payload["no_load"]
    (tmp_path / PACKET / "card_benchmark.json").write_text(json.dumps(payload),
                                                           encoding="utf-8")
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "no_load" in message
