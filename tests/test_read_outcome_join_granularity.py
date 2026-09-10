"""The join granularity registered in `docs/prereg/2026-09-10-join-granularity.md`.

The main run's protocol section 3 fixed the granularity before any outcome
existed: the internal curve uses the first draw of each prompt, the other
three serve the external curve. `v4_train.py` implements that and stamps
`internal_curve_scope="first_draw_only"` on every row. `read_outcome` keeps a
spec only when *every* one of its images carries both measurements, which at
R=4 no spec can -- so the registered D* rule ran eight windows on an empty
pairing, and the supplementary screen's internal arm ran on an empty list.

The repair cannot land while the main run is alive: `run_decoupling_pilot.py`
checks `source_sha256` at the top of every stage and this file is in `SOURCES`.
So the tests that need it are marked `xfail(strict=True)`. When the repair
lands they turn into XPASS, which pytest reports as a failure, and whoever
lands it has to come here and remove the markers. That is the point of them.

Fixtures are built from what the writers put on disk, checked against
`runs/v4/decoupling-main-20260908/evaluations/naive/step-00040` on 2026-09-10:
manifest rows carry spec_id/candidate_index/image_path, verified rows carry
image_correct/resolution, and s_select rows key the spec as `prompt_id`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDS_REPAIR = pytest.mark.xfail(
    strict=True,
    reason="lands after the main run reaches pilot_complete; prereg 2026-09-10 sections 2 and 9")


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location(
        "frozen_report_under_test", ROOT / "scripts/v4_decoupling_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block(directory: Path, specs: dict[str, dict]) -> Path:
    """Write the three files a checkpoint directory holds, in the writers' shape.

    `specs` maps spec_id to {"verdicts": [...], "scores": {draw_index: value},
    "pending": [draw_index, ...]}. The draw count comes from len(verdicts),
    which is how the run does it: the manifest is written per (prompt, draw).
    """

    directory.mkdir(parents=True, exist_ok=True)
    manifest, verified, selection = [], [], []
    for spec_id, plan in specs.items():
        for index, correct in enumerate(plan["verdicts"]):
            image = str(directory / "images" / f"naive-s00040-{spec_id}-{index}.png")
            manifest.append({"spec_id": spec_id, "candidate_index": index,
                             "image_path": image, "prompt": f"a prompt for {spec_id}",
                             "spec": {"spec_id": spec_id}, "seed": 1000 + index})
            verified.append({"spec_id": spec_id, "candidate_index": index,
                             "image_path": image, "image_correct": bool(correct),
                             "resolution": ("pending_human" if index in plan.get("pending", ())
                                            else "agreed"),
                             "detections": [], "disputed": False, "report": "",
                             "seed": 1000 + index, "verifier_agreement": True})
            if index in plan.get("scores", {}):
                selection.append({"prompt_id": spec_id, "image_path": image,
                                  "s_select": plan["scores"][index],
                                  "available": 4, "total": 4, "correct": 3,
                                  "abstained": 0, "missing": 0, "errors": [],
                                  "questions": [], "observation": {},
                                  "question_digest": "d" * 8, "rgb_sha256": "a" * 64,
                                  "metadata": {"internal_curve_scope": "first_draw_only",
                                               "outcome_draws_per_prompt": len(plan["verdicts"])}})
    for name, rows in (("manifest.jsonl", manifest), ("verified.jsonl", verified),
                       ("s_select.jsonl", selection)):
        (directory / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return directory


# One spec drawn four times, scored on draw 0 only, every image adjudicated.
# 0.8 and two-of-four true are picked so that the internal mean, the external
# mean and the wrong divisor (0.8 / 4 = 0.2) are three different numbers.
R4 = {"spec-0": {"verdicts": [True, True, False, False], "scores": {0: 0.8}}}


@pytest.fixture
def r4(tmp_path):
    return block(tmp_path / "step-00040", R4)


@NEEDS_REPAIR
def test_a_spec_scored_on_its_first_draw_and_fully_adjudicated_is_counted(report, r4):
    out = report.read_outcome(r4)
    assert out["complete_specs"] == {
        "spec-0": {"s_select": 0.8, "external": 0.5, "n_images": 4}}


@NEEDS_REPAIR
def test_the_internal_mean_divides_by_the_scored_images_not_the_drawn_ones(report, r4):
    """0.8 over one scored image, not 0.8 over four drawn ones.

    Protocol section 3 says the internal curve is measured on the first draw.
    Dividing by 4 would silently redefine it as a per-image quantity whose
    other three values happen to be zero.
    """

    out = report.read_outcome(r4)
    assert out["complete_specs"]["spec-0"]["s_select"] == pytest.approx(0.8)


@NEEDS_REPAIR
def test_spec_measurements_carries_the_first_draw_score(report, r4):
    """The second outlet of the same defect, prereg section 9.

    `spec_measurements[...]["s_select"]` feeds the supplementary screen's
    internal list. None here is what made `n_internal_paired_specs` zero in
    every window of the main run's report while the external side had 44.
    """

    out = report.read_outcome(r4)
    assert out["spec_measurements"]["spec-0"]["s_select"] == pytest.approx(0.8)


def test_one_unadjudicated_image_drops_the_whole_spec(report, tmp_path):
    """Registered in section 2 and deliberately strict.

    Averaging external over only the adjudicated images would let the
    denominator drift with adjudication progress, which makes the order the
    images were adjudicated in a confounder.
    """

    directory = block(tmp_path / "step-00040",
                      {"spec-0": {"verdicts": [True, True, False, False],
                                  "scores": {0: 0.8}, "pending": [3]}})
    assert report.read_outcome(directory)["complete_specs"] == {}


def test_a_spec_with_no_scored_image_is_not_counted(report, tmp_path):
    directory = block(tmp_path / "step-00040",
                      {"spec-0": {"verdicts": [True, True, False, False], "scores": {}}})
    assert report.read_outcome(directory)["complete_specs"] == {}


def test_spec_measurements_external_still_requires_every_image(report, tmp_path):
    """Section 9 changes the internal half of this dictionary, not the external
    half. All four images are meant to be adjudicated, so a missing verdict is
    genuinely missing.
    """

    directory = block(tmp_path / "step-00040",
                      {"spec-0": {"verdicts": [True, True, False, False],
                                  "scores": {0: 0.8}, "pending": [3]}})
    assert report.read_outcome(directory)["spec_measurements"]["spec-0"]["external"] is None


def test_the_dead_bounds_are_left_alone(report, r4):
    """Section 9 explicitly declines to touch these.

    They treat the three unscored images as missing data and so are nearly
    uninformative at R>1, but nothing in the repository reads them, and
    changing a field with no consumer is changing something no test covers.
    """

    row = report.read_outcome(r4)["spec_measurements"]["spec-0"]
    assert row["s_select_low"] == pytest.approx(0.2)
    assert row["s_select_high"] == pytest.approx(0.95)


def test_the_pilot_shape_is_untouched(report, tmp_path):
    """R=1 is what the pilot ran and what the frozen predicate was written for.

    The repair has to be invisible there, or it is not a repair but a second
    definition of the internal curve.
    """

    directory = block(tmp_path / "step-00040", {"spec-0": {"verdicts": [True], "scores": {0: 0.8}}})
    out = report.read_outcome(directory)
    assert out["complete_specs"] == {"spec-0": {"s_select": 0.8, "external": 1.0, "n_images": 1}}
    assert out["spec_measurements"]["spec-0"]["s_select"] == pytest.approx(0.8)


def test_the_writer_still_stamps_first_draw_only(report):
    """If the writer stops scoring only the first draw, this rule needs re-reading.

    The granularity is not a choice this reader makes; it is `v4_train.py`'s,
    recorded in section 3 of the protocol. A reader pinned to an assumption the
    writer has dropped is the defect this whole file exists for, one turn later.
    """

    source = (ROOT / "scripts/v4_train.py").read_text(encoding="utf-8")
    assert '"internal_curve_scope": "first_draw_only"' in source
    assert "first_draw = {prompt_id: images[(prompt_id, 0)] for prompt_id in prompt_ids}" in source
PATCH = "review-packets/dstar-join-granularity-20260910/read_outcome_repair.patch"


def test_the_patch_is_checked_out_with_lf_endings():
    """A CRLF copy of the patch cannot apply, and nothing else would say so.

    `git apply` matches the context lines byte for byte against
    `scripts/v4_decoupling_report.py`, which `.gitattributes` pins to eol=lf.
    Verified on 2026-09-10: a CRLF copy of this patch dies with "patch does
    not apply" at the first hunk. `*.patch` was falling through to
    `* text=auto` under core.autocrlf=true, so a fresh Windows checkout would
    have produced exactly that copy -- and the failure would surface at
    pilot_complete, the one moment the patch is needed.

    The attribute is asserted as well as the bytes. Deleting the
    `.gitattributes` line leaves the working tree LF until the next checkout,
    so a byte check alone would stay green across the commit that breaks it.
    """

    assert b"\r" not in (ROOT / PATCH).read_bytes(), f"{PATCH} has CR bytes and will not apply"
    attributes = subprocess.run(
        ["git", "check-attr", "eol", "--", PATCH],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert attributes.strip().endswith(": eol: lf"), (
        f"git would not restore {PATCH} as LF; got {attributes.strip()!r}")
