"""Regression tests for partial verifier truth and paired factual diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.v4.factual_truth import factual_answer, question_scope

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question(kind="count", phrase="red books", wanted="1"):
    text = (f"How many {phrase} are in this picture? Answer with a single number."
            if kind == "count" else
            f"Is there a {phrase} in this picture? Answer A or B only.\nA. yes\nB. no")
    return {"question_id": f"s:{kind}:0", "atom_id": f"s:{kind}:book",
            "text": text, "expected_answer": wanted,
            "choices": [] if kind == "count" else ["yes", "no"]}


def verified(items, resolution="agreed", disputed=()):
    return {"detections": items, "resolution": resolution, "disputed": list(disputed)}


RED = {"object": "book", "color": "red"}
BLUE = {"object": "notebook", "color": "blue"}


def test_legacy_existence_uses_colour_in_text_not_bare_atom_id():
    q = question("existence", "red book", "yes")
    assert factual_answer(q, verified([BLUE])).answer == "no"
    assert question_scope(q).colour == "red"


def test_legacy_total_and_colour_count_have_different_scopes():
    row = verified([RED, BLUE])
    assert factual_answer(question(phrase="books"), row).answer == "2"
    assert factual_answer(question(), row).answer == "1"


@pytest.mark.parametrize("kind,phrase", [("count", "red books"), ("existence", "red book")])
def test_disputed_key_is_unknown_and_crop_resolution_is_usable(kind, phrase):
    q = question(kind, phrase)
    row = verified([BLUE], "agreed_verdict", [RED])
    result = factual_answer(q, row)
    assert not result.known and result.reason == "disputed_scope"
    resolved = verified([RED, BLUE], "resolved_by_crop", [RED])
    assert factual_answer(q, resolved).known


def test_dispute_only_invalidates_its_scope_but_total_includes_all_colours():
    row = verified([RED], "agreed_verdict", [BLUE])
    assert factual_answer(question(), row).answer == "1"
    assert not factual_answer(question(phrase="books"), row).known


def test_unnameable_is_not_a_zero_count():
    result = factual_answer(question(), verified([{"object": "unnameable", "color": ""}], "human"))
    assert result.answer is None and not result.known and result.reason == "unnameable"


def test_official_detection_normalization_filters_surfaces():
    row = verified([{"object": "books", "color": " RED "}, {"object": "table", "color": "red"}])
    assert factual_answer(question(), row).answer == "1"
    assert factual_answer(question(phrase="red tables"), row).answer == "0"


def test_missing_and_unsupported_facts_do_not_become_absence():
    assert factual_answer(question(), None).reason == "missing_verification"
    assert factual_answer({"text": "What happened?"}, verified([])).reason == "unsupported_question"


def pool(prompt="p"):
    return {"prompt_id": prompt, "questions": [question()],
            "candidates": [{"candidate_id": "good", "image_path": prompt + "-good", "correct": True},
                           {"candidate_id": "bad", "image_path": prompt + "-bad", "correct": False}],
            "selection": {"dropped": None, "scores": {
                "naive": {"good": 1.0, "bad": 1.0},
                "rfo": {"good": 1.0, "bad": 0.0}}}}


def test_ideal_and_actual_share_complete_pool_denominator():
    ceiling = script("v4_observation_ceiling")
    p, excluded = pool(), pool("u")
    found = {"p-good": verified([RED]), "p-bad": verified([]),
             "u-good": verified([RED]), "u-bad": verified([], "agreed_verdict", [RED])}
    report = ceiling.paired_diagnostic({"p": p, "u": excluded}, found)
    assert report["eligible_ids"] == ["p"]
    assert report["known_questions"] == 3 and report["total_questions"] == 4
    assert report["unknown_reasons"] == {"disputed_scope": 1}
    assert {m["n_pools"] for m in report["metrics"].values()} == {1}
    empty = ceiling.paired_diagnostic({"u": excluded}, found)
    assert all(m is None for m in empty["metrics"].values())


def test_residual_distinguishes_partial_from_all_pairs_and_keeps_unknown():
    residual = script("v4_residual_diagnosis")
    p = pool()
    p["candidates"].append({"candidate_id": "also-bad", "image_path": "same", "correct": False})
    found = {"p-good": verified([RED]), "p-bad": verified([]), "same": verified([RED])}
    assert residual.classify(p, found)[0] == "some_pairs_distinct"
    found["same"] = verified([], "agreed_verdict", [RED])
    assert residual.classify(p, found) == ("unknown", {"disputed_scope": 1})


def test_residual_includes_legacy_colour_existence_information():
    residual = script("v4_residual_diagnosis")
    p = pool()
    p["questions"] = [question("existence", "red book", "yes"), question(phrase="books")]
    found = {"p-good": verified([RED]), "p-bad": verified([BLUE])}
    assert residual.classify(p, found)[0] == "all_pairs_distinct"


def test_count_report_marks_unnameable_and_disputed_unknown(tmp_path):
    report_script = script("v4_resolution_verdict")
    run = tmp_path / "source"
    run.mkdir()
    rows = [dict(verified([RED]), image_path="known"),
            dict(verified([], "agreed_verdict", [RED]), image_path="disputed"),
            dict(verified([{"object": "unnameable"}], "human"), image_path="blob")]
    (run / "verified.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    (tmp_path / "runs.json").write_text(json.dumps({"runs": [str(run)]}), encoding="utf-8")
    p = {"prompt_id": "p", "questions": [question()],
         "candidates": [{"candidate_id": r["image_path"], "image_path": r["image_path"]} for r in rows]}
    (tmp_path / "pools.jsonl").write_text(json.dumps(p), encoding="utf-8")
    obs = [{"prompt_id": "p", "candidate_id": r["image_path"], "observation": {"answers": [
        {"question_id": "s:count:0", "normalized_answer": "1", "abstain": False}]}} for r in rows]
    for arm in ("naive", "rfo"):
        (tmp_path / f"observations.{arm}.jsonl").write_text("\n".join(map(json.dumps, obs)), encoding="utf-8")
    for result in report_script.counting_report(tmp_path).values():
        assert result["known"] == 1 and result["total"] == 3
        assert result["miscounted"] == 0
        assert result["unknown_reasons"] == {"disputed_scope": 1, "unnameable": 1}
