"""The detector waits for room on a shared card instead of barging in.

Four things went wrong with the L3 preview on 2026-09-04 and only one of them
was this: an 8B detector asked for 15.17 GiB on a card that reported 22.76 GiB
free, and lost the interval between the reading and the claim to somebody
else's job. A wait cannot close that interval. What it can do is refuse to
start against a card that visibly has no room, and stop us being the process
that squeezes a stranger's work off the machine.

The two properties worth pinning are the ones that fail silently:

  * the wait must not itself take memory, which is why the reading comes from
    nvidia-smi and not torch.cuda.mem_get_info -- the latter builds a CUDA
    context on the card it is asked about;
  * the wait must always end in starting, never in refusing, because a queue
    accident recorded as terminal condition B.3 reads as a statement about the
    experiment.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_run_pipeline", ROOT / "scripts" / "v4_run_pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pipeline)


class _Recorder:
    """Stands in for time.sleep and remembers what it was asked to wait."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, seconds: int) -> None:
        self.calls.append(seconds)


def _smi(readings: str):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=readings, stderr="")
    return run


# ----------------------------------------------------------------- free_mib


def test_a_cpu_device_has_no_free_memory_question(monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("nvidia-smi must not be run for a cpu device")
    monkeypatch.setattr(subprocess, "run", explode)
    assert pipeline.free_mib("cpu") is None


@pytest.mark.parametrize("device,expected", [
    ("cuda", 19431),
    ("cuda:0", 19431),
    ("cuda:1", 15184),
])
def test_the_reading_is_indexed_by_the_card_asked_about(monkeypatch, device, expected):
    monkeypatch.setattr(subprocess, "run", _smi("19431\n15184\n"))
    assert pipeline.free_mib(device) == expected


def test_a_card_index_the_machine_does_not_have_reads_as_unknown(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _smi("19431\n"))
    assert pipeline.free_mib("cuda:3") is None


def test_no_nvidia_smi_reads_as_unknown_rather_than_as_empty(monkeypatch):
    """Unknown must not be mistaken for zero: zero would wait out the patience."""
    def missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", missing)
    assert pipeline.free_mib("cuda:0") is None


def test_garbage_from_the_tool_reads_as_unknown(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _smi("[N/A]\n"))
    assert pipeline.free_mib("cuda:0") is None


def test_the_reading_does_not_build_a_cuda_context(monkeypatch):
    """torch.cuda.mem_get_info would cost a few hundred MiB to ask the question.

    Read off the parse tree rather than the text, because the reason for not
    calling it is written down in a docstring that has to say its name.
    """
    tree = ast.parse(
        (ROOT / "scripts" / "v4_run_pipeline.py").read_text(encoding="utf-8"))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "mem_get_info" not in called

    monkeypatch.setattr(subprocess, "run", _smi("19431\n15184\n"))
    assert pipeline.free_mib("cuda:0") == 19431


# ---------------------------------------------------------------- await_room


def test_a_card_with_room_is_not_waited_for(monkeypatch):
    monkeypatch.setattr(pipeline, "free_mib", lambda _d: 19431)
    slept = _Recorder()
    pipeline.await_room("cuda:0", need_mib=18000, sleep=slept)
    assert slept.calls == []


def test_a_busy_card_is_waited_for_until_it_clears(monkeypatch):
    readings = iter([5000, 5000, 5000, 20000])
    monkeypatch.setattr(pipeline, "free_mib", lambda _d: next(readings))
    slept = _Recorder()
    pipeline.await_room("cuda:0", need_mib=18000, poll_s=60, sleep=slept)
    assert slept.calls == [60, 60, 60]


def test_the_wait_ends_in_starting_and_never_in_refusing(monkeypatch):
    """A queue accident is not terminal condition B.3, so this may not raise."""
    monkeypatch.setattr(pipeline, "free_mib", lambda _d: 200)
    slept = _Recorder()
    pipeline.await_room("cuda:0", need_mib=18000, patience_s=300, poll_s=60,
                        sleep=slept)
    assert slept.calls == [60] * 5


def test_an_unknown_reading_starts_immediately(monkeypatch):
    """A machine without nvidia-smi must run the pipeline, not stall on it."""
    monkeypatch.setattr(pipeline, "free_mib", lambda _d: None)
    slept = _Recorder()
    pipeline.await_room("cuda:0", need_mib=18000, sleep=slept)
    assert slept.calls == []


def test_the_card_going_dark_mid_wait_stops_the_wait(monkeypatch):
    readings = iter([5000, None])
    monkeypatch.setattr(pipeline, "free_mib", lambda _d: next(readings))
    slept = _Recorder()
    pipeline.await_room("cuda:0", need_mib=18000, poll_s=60, sleep=slept)
    assert slept.calls == [60]


# ------------------------------------------------------------ the threshold


def test_the_threshold_leaves_a_stranger_room_to_work():
    """18000 of a 24576 MiB card: we start beside ~6 GiB of someone else, not more.

    Both ends matter. Below the failed 15.17 GiB warmup the gate would wave
    through the allocation that already died; above about 20000 nothing on a
    shared machine would ever pass and every round would wait out its patience.
    """
    assert 15534 < pipeline.DETECTOR_MIB < 20000


def test_both_detector_stages_ask_before_they_load():
    source = (ROOT / "scripts" / "v4_run_pipeline.py").read_text(encoding="utf-8")
    assert source.count("    await_room(args.device)\n") == 2
    for load_line in ("    detector = load(args.detector, device=args.device)",
                      "    detector = load(args.primary, device=args.device)"):
        before = source.split(load_line)[0]
        assert before.rstrip().endswith("await_room(args.device)")
