"""The registered card-scheduling criterion, which exists in exactly one place.

Section 3.4 fixed `gate_card_schedule_decided` by making it read
`card_benchmark.json` and *not* re-implement the rule -- a second copy of a
criterion is two copies that drift. That decision makes `card_benchmark.py`'s
`verdict()` the only implementation of the rule registered in section 0.3, and
it had no test and has never been run: the packet holds the script and nothing
else.

It runs once, on both cards, inside the ~1.5 h window between the main run
finishing and replicate 1 starting. Its timing half cannot be tested without
that window. Its decision half is a pure function of four numbers, and that is
what decides the schedule for 55 checkpoint blocks.

The module is loaded out of the arm B branch, because that is where it lives
until the merge; these tests skip if it is not in the checkout.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARM_B = "staging/arm-b-merged"
PACKET = "review-packets/card-scheduling-20260909"
SCRIPT = f"{PACKET}/card_benchmark.py"


def _source() -> str | None:
    result = subprocess.run(["git", "show", f"{ARM_B}:{SCRIPT}"], cwd=ROOT,
                            capture_output=True, check=False)
    return result.stdout.decode("utf-8") if result.returncode == 0 else None


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    source = _source()
    if source is None:
        pytest.skip(f"{ARM_B}:{SCRIPT} is not in this checkout")
    # Planted at its real depth so the module's `parents[2]` resolves to a
    # root rather than off the top of the filesystem.
    home = tmp_path_factory.mktemp("armb") / PACKET
    home.mkdir(parents=True)
    path = home / "card_benchmark.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("card_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(detector: str, *, serial: float, parallel: float, differing: int = 0) -> dict:
    """The fields `verdict` reads, named as `measure` writes them."""

    return {"detector": detector,
            "wall_serial_both_on_fastest": serial,
            "wall_parallel_one_arm_per_card": parallel,
            "cards_agree": {"compared": 64, "differing": differing, "examples": []}}


def both(**kwargs) -> list[dict]:
    return [row("internvl", **kwargs), row("qwen3vl", **kwargs)]


def test_the_margin_is_the_number_section_0_3_registered(benchmark):
    assert benchmark.MARGIN == 0.90


def test_the_gate_and_the_benchmark_hold_the_same_margin(benchmark):
    """Section 3.4 let the gate keep its own copy of the number, not the rule.

    One number in two files is the smallest thing that can drift, so it is
    checked rather than trusted.
    """

    preflight = (ROOT / "scripts/v4_e3_launch_preflight.py").read_text(encoding="utf-8")
    assert "CARD_MARGIN = 0.90" in preflight
    assert benchmark.MARGIN == 0.90


def test_clearing_the_margin_on_both_detectors_adopts_serial(benchmark):
    result = benchmark.verdict(both(serial=80.0, parallel=100.0))
    assert result["adopt_serial"] is True
    assert "10% margin" in result["reason"]


def test_exactly_at_the_margin_counts_as_clearing_it(benchmark):
    """The rule is `wall_S <= 0.90 * wall_P`, written that way before the
    measurement. A boundary read as `<` is a different registered rule.
    """

    result = benchmark.verdict(both(serial=90.0, parallel=100.0))
    assert result["adopt_serial"] is True


def test_a_hair_over_the_margin_does_not(benchmark):
    result = benchmark.verdict(both(serial=90.01, parallel=100.0))
    assert result["adopt_serial"] is False


def test_one_detector_clearing_is_the_registered_tie_break(benchmark):
    """A split verdict keeps the current design. Section 0.3 fixed that before
    the measurement precisely so a 1-1 split could not be argued either way.
    """

    results = [row("internvl", serial=80.0, parallel=100.0),
               row("qwen3vl", serial=99.0, parallel=100.0)]
    result = benchmark.verdict(results)
    assert result["adopt_serial"] is False
    assert "internvl" in result["reason"]
    assert "keeps" in result["reason"]


def test_no_detector_clearing_says_so_rather_than_naming_an_empty_list(benchmark):
    result = benchmark.verdict(both(serial=99.0, parallel=100.0))
    assert result["adopt_serial"] is False
    assert "no detector" in result["reason"]


def test_a_faster_serial_arrangement_is_still_refused_if_the_cards_disagree(benchmark):
    """The one case where timing must not decide.

    Two cards that detect different things make the schedule a change to the
    measurements. The timing here clears the margin comfortably and the
    verdict still has to be no, with a reason that says why.
    """

    results = [row("internvl", serial=50.0, parallel=100.0, differing=1),
               row("qwen3vl", serial=50.0, parallel=100.0)]
    result = benchmark.verdict(results)
    assert result["adopt_serial"] is False
    assert "do not detect the same things" in result["reason"]
    assert "internvl" in result["reason"]
    assert "correctness question before it is a scheduling one" in result["reason"]


def test_agreement_is_read_from_the_shape_the_detector_writes(benchmark, tmp_path):
    """`detections.internvl.jsonl` rows carry `image_path` and `detections`.

    Checked against those key names rather than against invented ones: a
    reader that keys off something the writer does not emit is the defect
    section 3.4 registered, and `agree` runs after all four timed passes, so
    a KeyError there costs the measurement it was meant to check.
    """

    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text(
        json.dumps({"image_path": "im/1.png", "detections": ["cat"], "reply": "x"}) + "\n"
        + json.dumps({"image_path": "im/2.png", "detections": ["dog"], "reply": "y"}) + "\n",
        encoding="utf-8")
    right.write_text(
        json.dumps({"image_path": "im/1.png", "detections": ["cat"], "reply": "z"}) + "\n"
        + json.dumps({"image_path": "im/2.png", "detections": ["dog"], "reply": "w"}) + "\n",
        encoding="utf-8")
    assert benchmark.agree(left, right) == {"compared": 2, "differing": 0, "examples": []}


def test_one_differing_image_is_reported_by_name(benchmark, tmp_path):
    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text(json.dumps({"image_path": "im/1.png", "detections": ["cat"]}) + "\n",
                    encoding="utf-8")
    right.write_text(json.dumps({"image_path": "im/1.png", "detections": ["cat", "dog"]}) + "\n",
                     encoding="utf-8")
    reported = benchmark.agree(left, right)
    assert reported["differing"] == 1
    assert reported["examples"] == ["1.png"]


def test_the_reply_text_is_not_what_is_compared(benchmark, tmp_path):
    """Detectors are asked to name objects, and the prose around the answer is
    not the answer. Comparing replies would report a disagreement on every
    image and make the correctness check unusable.
    """

    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text(json.dumps({"image_path": "im/1.png", "detections": ["cat"],
                                "reply": "I can see a cat."}) + "\n", encoding="utf-8")
    right.write_text(json.dumps({"image_path": "im/1.png", "detections": ["cat"],
                                 "reply": "There is one cat in this image."}) + "\n",
                     encoding="utf-8")
    assert benchmark.agree(left, right)["differing"] == 0


def test_the_payload_carries_every_key_the_gate_reads(benchmark):
    """The other half of section 3.4's fix, pinned from the writer's side."""

    source = (Path(benchmark.__file__)).read_text(encoding="utf-8")
    for key in ('"no_load"', '"margin"', '"verdict"', '"results"'):
        assert key in source, f"card_benchmark.py no longer writes {key}"
    assert '"card_benchmark.json"' in source
    assert '"adopt_serial"' in source


def test_a_busy_card_measurement_records_itself_as_one(benchmark):
    """`--allow-busy-cards` must not produce a payload that looks no-load.

    The gate refuses a report whose `no_load` is false or missing, which only
    works if the flag actually reaches the field.
    """

    source = (Path(benchmark.__file__)).read_text(encoding="utf-8")
    assert '"no_load": not args.allow_busy_cards' in source
