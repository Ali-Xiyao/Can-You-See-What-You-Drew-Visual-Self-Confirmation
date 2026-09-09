"""Tests for the deviation 13.2 selection pass.

The GPU half cannot be tested without a GPU. Everything around it can, and the
failures that matter are all silent: a preamble that leaks into the blind
condition, a resumed run that appends to a half-written pair, a row shape the
analysis reads but reads backwards. The last test runs the writer and the
reader end to end and checks the sign of the thing endpoint 3 exists to
measure, because a sign error here produces a perfectly ordinary number.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfsight.analysis import endpoint3
from selfsight.v4.observe import PROMPTED_PREAMBLE
from selfsight.v4.questions import build_questions
from selfsight.v4.spec import SceneSpec

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_selection_observe.py"

SEED = 7
ARM = "naive"
STEP = 8


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_selection_observe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observe = _module()


# --- fixtures ---------------------------------------------------------------
#
# One prompt asks for one white notebook and two blue pens. The "wrong"
# candidate draws an extra of each, which is what makes `build_counting`
# produce an `image_differs_from_spec` trial: the drawn count and the requested
# count disagree, so agreeing with the picture means disagreeing with the
# description. Every other trial agrees with both. Four questions either way.


def _spec(spec_id: str) -> dict:
    return {"spec_id": spec_id, "prompt": "one white notebook, two blue pens",
            "objects": [{"object": "book", "color": "white", "count": 1},
                        {"object": "pen", "color": "blue", "count": 2}],
            "relations": [], "surface": "wooden table", "metadata": {}}


def _detections(pens: int, books: int) -> list[dict]:
    items = [{"object": "pen", "color": "blue", "bbox": [10 + 40 * i, 20, 40 + 40 * i, 300],
              "center": [25 + 40 * i, 160]} for i in range(pens)]
    items += [{"object": "book", "color": "white",
               "bbox": [200 + 60 * i, 30, 260 + 60 * i, 320],
               "center": [230 + 60 * i, 175]} for i in range(books)]
    return items


# Index 0 is the off-spec candidate on purpose: it makes the first-index
# tie-break and the uniform-random one land on different candidates, so a test
# that confuses them fails instead of agreeing by luck.
CANDIDATES = ((0, (3, 2), False), (1, (2, 1), True))


def _write_eval(run: Path, arm: str, step: int, *, prompts: tuple[str, ...] = ("s1", "s2"),
                unnameable: str | None = None, unverified: str | None = None) -> Path:
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest, verified = [], []
    for spec_id in prompts:
        for index, (pens, books), correct in CANDIDATES:
            image = f"{spec_id}-c{index}.png"
            manifest.append({"spec_id": spec_id, "candidate_index": index, "seed": SEED,
                             "prompt": _spec(spec_id)["prompt"], "image_path": image,
                             "spec": _spec(spec_id)})
            if image == unverified:
                continue
            settled = _detections(pens, books)
            if image == unnameable:
                settled = settled + [{"object": "unnameable", "color": None,
                                      "bbox": [0, 0, 1, 1], "center": [0, 0]}]
            verified.append({"spec_id": spec_id, "candidate_index": index,
                             "image_path": image, "detections": settled,
                             "image_correct": correct, "resolution": "agreed",
                             "disputed": False, "verifier_agreement": 1.0, "report": {}})
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    (directory / "verified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verified), encoding="utf-8")
    return directory


def _questions(spec_id: str, pens: int, books: int):
    return build_questions(SceneSpec.from_dict(_spec(spec_id)),
                           _detections(pens, books), seed=SEED)


class Backbone:
    """Answers by rule and remembers every question it was handed.

    `rule` gets the image and the atom and returns a raw reply. The atom keeps
    its `question_id` through `to_atomic` and through the preamble wrapping,
    which is what lets a rule answer differently per trial without parsing
    prose. The image has to come with it: `question_id` is built from the spec
    id and the family, so two candidates for the same prompt reuse it while
    meaning different things.
    """

    def __init__(self, rule=None):
        self.rule = rule or (lambda image, atom: "A")
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def observe_atoms(self, image_path, atoms):
        self.calls.append((image_path, tuple(atom.text for atom in atoms)))
        return SimpleNamespace(
            answers=[SimpleNamespace(raw_answer=self.rule(image_path, atom))
                     for atom in atoms])


def _run(tmp_path: Path, backbone: Backbone, **kwargs) -> tuple[Path, dict]:
    run = tmp_path / "run"
    _write_eval(run, ARM, STEP, **kwargs)
    out = run / "analysis" / "selection" / ARM
    summary = observe.observe_step(backbone, run, ARM, STEP, out, adapter_digest="deadbeef")
    return out / f"step-{STEP:05d}.jsonl", summary


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- what gets asked --------------------------------------------------------


def test_the_blind_condition_asks_the_builder_s_question_verbatim(tmp_path: Path):
    backbone = Backbone()
    _run(tmp_path, backbone)
    wanted = {question.prompt_text for question in _questions("s1", 3, 2)}
    asked = {text for _, texts in backbone.calls for text in texts}
    assert wanted <= asked


def test_the_preamble_goes_on_the_prompted_condition_and_nowhere_else(tmp_path: Path):
    """The one failure nothing downstream could catch.

    A blind question carrying the preamble produces an ordinary score under the
    label `blind`, and the context effect it feeds is then a difference between
    two prompted conditions -- reliably near zero, and reliably publishable as
    "no dose".
    """

    backbone = Backbone()
    _run(tmp_path, backbone)
    marker = PROMPTED_PREAMBLE.format(prompt="", question="").split("{")[0][:40]
    wrapped = [text for _, texts in backbone.calls for text in texts if marker in text]
    plain = [text for _, texts in backbone.calls for text in texts if marker not in text]
    assert len(wrapped) == len(plain), "one wrapped copy per bare copy"
    assert all("one white notebook, two blue pens" in text for text in wrapped)


def test_the_prompted_question_is_the_blind_one_inside_the_preamble(tmp_path: Path):
    # Not merely "both conditions ran": the two conditions have to differ in
    # the context and in nothing else, or the difference is not a context
    # effect.
    backbone = Backbone()
    path, _ = _run(tmp_path, backbone)
    rows = _rows(path)
    by_condition = {condition: {(row["image_path"], row["question"]) for row in rows
                                if row["condition"] == condition}
                    for condition in observe.CONDITIONS}
    assert by_condition["blind"] == by_condition["prompted"]


def test_the_condition_labels_are_the_ones_endpoint_3_reads():
    # The writer and the reader are in different files, and the reader raises
    # on any other label -- after the GPU hours have been spent.
    assert observe.CONDITIONS == endpoint3.CONDITIONS == ("blind", "prompted")


# --- which images supply trials ---------------------------------------------


def test_every_candidate_is_measured(tmp_path: Path):
    # The run's own evaluation records candidate 0 only. Endpoint 3's y axis is
    # a choice among candidates, so a pass that inherited that filter would
    # have nothing to choose between.
    path, _ = _run(tmp_path, Backbone())
    rows = _rows(path)
    assert {row["candidate_index"] for row in rows} == {0, 1}
    # 2 prompts x 2 candidates x 2 conditions x 4 questions
    assert len(rows) == 32


def test_an_unnameable_image_supplies_no_trials(tmp_path: Path):
    path, summary = _run(tmp_path, Backbone(), unnameable="s1-c0.png")
    assert summary["unnameable"] == 1
    assert not [row for row in _rows(path) if row["image_path"] == "s1-c0.png"]


def test_an_image_without_a_verdict_is_skipped(tmp_path: Path):
    path, summary = _run(tmp_path, Backbone(), unverified="s2-c1.png")
    assert summary["unverified"] == 1
    assert not [row for row in _rows(path) if row["image_path"] == "s2-c1.png"]


def test_steps_present_needs_both_the_manifest_and_the_verdicts(tmp_path: Path):
    """Half a step is not a step.

    The supervisor writes the manifest when it draws and the verdicts when the
    adjudication ladder finishes, so a step caught in between has one and not
    the other. Either half alone would be resolved into the plan and then fail
    on the missing file -- after the checkpoint had been loaded onto the GPU.
    """

    run = tmp_path / "run"
    _write_eval(run, ARM, STEP)
    half = run / "evaluations" / ARM
    (half / "step-00016").mkdir(parents=True)
    (half / "step-00016" / "manifest.jsonl").write_text("", encoding="utf-8")
    (half / "step-00024").mkdir(parents=True)
    (half / "step-00024" / "verified.jsonl").write_text("", encoding="utf-8")
    assert observe.steps_present(run, ARM) == [STEP]
    assert observe.steps_present(run, "rfo_gold") == []


# --- resuming ---------------------------------------------------------------


def test_a_finished_pair_is_not_asked_again(tmp_path: Path):
    run = tmp_path / "run"
    _write_eval(run, ARM, STEP)
    out = run / "analysis" / "selection" / ARM
    first = Backbone()
    observe.observe_step(first, run, ARM, STEP, out, adapter_digest="d")
    second = Backbone()
    summary = observe.observe_step(second, run, ARM, STEP, out, adapter_digest="d")
    assert second.calls == []
    assert summary["answers"] == 0
    assert summary["resumed_pairs"] == 8
    assert len(_rows(out / f"step-{STEP:05d}.jsonl")) == 32


def test_a_half_written_pair_is_dropped_and_asked_again(tmp_path: Path):
    """A pair is four lines, so a killed process can leave two of them.

    Counting rows rather than trusting the presence of one is the whole point:
    resuming on presence would leave that candidate scored on half its
    questions, which is a number, not an error.
    """

    run = tmp_path / "run"
    _write_eval(run, ARM, STEP)
    out = run / "analysis" / "selection" / ARM
    observe.observe_step(Backbone(), run, ARM, STEP, out, adapter_digest="d")
    path = out / f"step-{STEP:05d}.jsonl"
    rows = _rows(path)
    truncated = [row for row in rows
                 if not (row["image_path"] == "s1-c0.png" and row["condition"] == "blind")]
    victim = [row for row in rows
              if row["image_path"] == "s1-c0.png" and row["condition"] == "blind"]
    path.write_text("".join(json.dumps(row) + "\n" for row in truncated + victim[:2]),
                    encoding="utf-8")

    backbone = Backbone()
    summary = observe.observe_step(backbone, run, ARM, STEP, out, adapter_digest="d")
    assert summary["dropped_partial_rows"] == 2
    assert summary["answers"] == 4
    assert len(backbone.calls) == 1
    after = _rows(path)
    assert len(after) == 32
    keys = [(row["image_path"], row["condition"], row["question"]) for row in after]
    assert len(set(keys)) == 32, "a resumed pair was appended on top of itself"


def test_a_file_belonging_to_another_measurement_is_refused(tmp_path: Path):
    run = tmp_path / "run"
    _write_eval(run, ARM, STEP)
    out = run / "analysis" / "selection" / ARM
    out.mkdir(parents=True)
    (out / f"step-{STEP:05d}.jsonl").write_text(
        json.dumps({"image_path": "somewhere-else.png", "condition": "blind"}) + "\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="does not plan"):
        observe.observe_step(Backbone(), run, ARM, STEP, out, adapter_digest="d")


# --- the writer and the reader together -------------------------------------


def test_a_reciter_and_a_reader_produce_the_registered_signs(tmp_path: Path):
    """End to end, with the sign conventions hand-computed.

    The backbone here is the caricature endpoint 3 is about: blind it reads the
    pixels perfectly, prompted it recites the description perfectly. Deviation
    13.2 point 3 says x is blind minus prompted `correct` on conflict trials,
    so x = 1 - 0 = +1. Point 4 says y is the blind pick minus the prompted
    pick: blind, the on-spec candidate scores 4/4 against the off-spec
    candidate's 3/4 and is picked, so 1.0; prompted, both score 4/4, and the
    tie resolves to 0.5 under the uniform expectation and to the off-spec
    candidate -- index 0, incorrect -- under first-index. So y = 0.5 and the
    strict reading is 1.0.

    Every one of those numbers goes through the row shape this script writes.
    A field renamed on one side of the boundary lands here.
    """

    gold = {(f"{spec_id}-c{index}.png", question.question_id): question
            for spec_id in ("s1", "s2")
            for index, (pens, books), _ in CANDIDATES
            for question in _questions(spec_id, pens, books)}

    def rule(image, atom):
        question = gold[(image, atom.question_id)]
        prompted = "one white notebook, two blue pens" in atom.text
        if prompted and question.gold_source == endpoint3.CONFLICT_GOLD:
            return "B" if question.gold == "A" else "A"
        return question.gold

    path, _ = _run(tmp_path, Backbone(rule))
    rows = _rows(path)

    dose, counts = endpoint3.context_effect(rows)
    assert counts == {"blind_trials": 2, "blind_correct": 2,
                      "prompted_trials": 2, "prompted_correct": 0}
    assert dose == pytest.approx(1.0)

    point = endpoint3.checkpoint_point(rows, seed=20260906, arm=ARM, step=STEP)
    assert point.pools == 2
    assert point.candidates == 4
    assert point.context_effect == pytest.approx(1.0)
    assert point.selection_gain == pytest.approx(0.5)
    assert point.selection_gain_first_index == pytest.approx(1.0)
    assert point.cluster == (20260906, STEP)


def test_the_reader_finds_the_file_where_the_writer_puts_it(tmp_path: Path):
    run = tmp_path / "run"
    _write_eval(run, ARM, STEP)
    out = run / "analysis" / "selection" / ARM
    observe.observe_step(Backbone(), run, ARM, STEP, out, adapter_digest="d")
    assert endpoint3.selection_path(run, ARM, STEP).exists()
    assert endpoint3.available_steps(run, ARM) == [STEP]
