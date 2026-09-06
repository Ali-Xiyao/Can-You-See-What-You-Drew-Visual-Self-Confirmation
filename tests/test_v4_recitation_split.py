"""Regression coverage for frozen-question, partial-truth recitation diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report_script():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_recitation_split.py"
    spec = importlib.util.spec_from_file_location("recitation_split_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question(text="How many red books are in this picture?", wanted="one"):
    return {"question_id": "s:count:0", "atom_id": "legacy:count:book",
            "family": "count", "text": text, "expected_answer": wanted,
            "choices": []}


def pool(candidates, q=None):
    return {"prompt_id": "p", "questions": [q or question()],
            "candidates": [{"candidate_id": name, "image_path": name}
                           for name in candidates]}


def verdict(items, resolution="agreed", disputed=()):
    return {"detections": items, "resolution": resolution, "disputed": list(disputed)}


def observation(candidate, value="one", **answer_fields):
    answer = {"question_id": "s:count:0", "normalized_answer": value,
              "error": None, "abstain": False, **answer_fields}
    return {"prompt_id": "p", "candidate_id": candidate,
            "observation": {"answers": [answer]}}


RED = {"object": "notebook", "color": "red"}
BLUE = {"object": "book", "color": "blue"}


def test_actual_text_number_words_and_partial_truth(report_script):
    pools = [pool(["known", "disputed", "crop"])]
    found = {
        "known": verdict([RED, BLUE]),
        "disputed": verdict([BLUE], "agreed_verdict", [RED]),
        "crop": verdict([RED], "resolved_by_crop", [RED]),
    }
    report = report_script.recitation_report(
        pools, found, {"naive": [observation(name) for name in found]})
    arm = report["arms"]["naive"]
    assert arm["total_questions"] == 3
    assert arm["known_questions"] == 2
    assert arm["unknown_reasons"] == {"disputed_scope": 1}
    assert arm["cells"]["picture_matches_request"]["total"] == 2
    assert arm["cells"]["picture_matches_request"]["said_truth"] == 2
    assert arm["cells"]["picture_differs_from_request"]["total"] == 0
    assert arm["cells"]["picture_differs_from_request"]["said_truth_rate_available"] is None
    assert [r["fact_answer"] for r in arm["records"]] == ["1", None, "1"]


def test_legacy_bare_count_uses_total_and_reports_unrepresentable_truth(report_script):
    q = question("How many books are in this picture?", "two")
    q["choices"] = ["one", "two"]
    report = report_script.recitation_report(
        [pool(["image"], q)], {"image": verdict([RED, BLUE, BLUE])},
        {"rfo": [observation("image", "two")]})
    cell = report["arms"]["rfo"]["cells"]["picture_differs_from_request"]
    assert cell["total"] == cell["said_request"] == cell["truth_not_in_choices"] == 1
    assert cell["said_truth"] == 0
    assert cell["wrong_answer_minus_truth"] == {"-1": 1}


def test_unavailable_answers_keep_expected_denominator(report_script):
    names = ["ok", "absent", "missing", "obs_error", "answer_error", "abstain", "null"]
    rows = [observation("ok", "two"),
            {"prompt_id": "p", "candidate_id": "missing", "observation": {"answers": []}},
            {"prompt_id": "p", "candidate_id": "obs_error", "error": "failed"},
            observation("answer_error", "one", error="failed"),
            observation("abstain", "one", abstain=True), observation("null", None)]
    report = report_script.recitation_report(
        [pool(names)], {name: verdict([RED, RED]) for name in names}, {"naive": rows})
    cell = report["arms"]["naive"]["cells"]["picture_differs_from_request"]
    assert cell["total"] == 7
    assert cell["answer_status"] == {
        "available": 1, "missing_observation": 1, "missing_answer": 1,
        "observation_error": 1, "answer_error": 1, "abstain": 1, "null_answer": 1}
    assert cell["said_truth_rate_available"] == 1
    assert cell["said_truth_rate_all_requested"] == 1 / 7
    assert cell["wrong_answer_minus_truth"] == {}


def test_unknown_facts_and_empty_input_do_not_become_wrong_answers(report_script):
    report = report_script.recitation_report(
        [pool(["missing", "unnameable"])],
        {"unnameable": verdict([{"object": "unnameable"}])},
        {"naive": [observation("missing"), observation("unnameable")]})
    arm = report["arms"]["naive"]
    assert arm["known_questions"] == 0
    assert arm["unknown_reasons"] == {"missing_verification": 1, "unnameable": 1}
    assert arm["cells"]["fact_unknown"]["said_request"] == 2
    assert arm["cells"]["fact_unknown"]["said_truth_rate_available"] is None
    empty = report_script.recitation_report([], {}, {"naive": []})["arms"]["naive"]
    assert empty["total_questions"] == 0
    assert empty["fact_coverage"] is None


def test_duplicate_or_unexpected_observations_rejected(report_script):
    with pytest.raises(ValueError, match="Duplicate"):
        report_script.recitation_report([pool(["a"])], {"a": verdict([RED])},
                                        {"naive": [observation("a"), observation("a")]})
    with pytest.raises(ValueError, match="Unexpected"):
        report_script.recitation_report([pool(["a"])], {},
                                        {"naive": [observation("outside")]})
    duplicate = observation("a")
    duplicate["observation"]["answers"] *= 2
    with pytest.raises(ValueError, match="Duplicate answer"):
        report_script.recitation_report([pool(["a"])], {}, {"naive": [duplicate]})


def test_unsupported_questions_and_nonnumeric_replies_are_explicit(report_script):
    unsupported = question("Please give the book count.")
    report = report_script.recitation_report(
        [pool(["a"], unsupported)], {"a": verdict([RED])},
        {"naive": [observation("a")]})
    assert report["arms"]["naive"]["unknown_reasons"] == {"unsupported_question": 1}
    report = report_script.recitation_report(
        [pool(["a", "blank"])], {name: verdict([RED, RED]) for name in ("a", "blank")},
        {"naive": [observation("a", "unclear"), observation("blank", " ")]})
    cell = report["arms"]["naive"]["cells"]["picture_differs_from_request"]
    assert cell["wrong_answer_minus_truth"] == {"non_numeric": 1}
    assert cell["answer_status"] == {"available": 1, "empty_answer": 1}
    assert cell["available"] == 1


def test_directory_and_cli_preserve_inputs_and_refuse_overwrite(report_script, tmp_path, capsys):
    source, analysis = tmp_path / "source", tmp_path / "analysis"
    source.mkdir()
    analysis.mkdir()
    (source / "verified.jsonl").write_text(json.dumps({"image_path": "a", **verdict(
        [BLUE], "agreed_verdict", [RED])}) + "\n", encoding="utf-8")
    (analysis / "runs.json").write_text(json.dumps({"runs": [str(source)]}), encoding="utf-8")
    (analysis / "pools.jsonl").write_text(json.dumps(pool(["a"])) + "\n", encoding="utf-8")
    (analysis / "observations.naive.jsonl").write_text(json.dumps(observation("a")) + "\n",
                                                       encoding="utf-8")
    inputs = {p: p.read_bytes() for folder in (source, analysis) for p in folder.iterdir()}
    built = report_script.build_report(analysis)
    assert built["arms"]["naive"]["unknown_reasons"] == {"disputed_scope": 1}
    assert built["arms"]["rfo"]["answer_status"]["missing_observation"] == 1
    report_script.main([str(analysis)])
    assert json.loads(capsys.readouterr().out)["arms"]["naive"]["known_questions"] == 0
    output = tmp_path / "review" / "report.json"
    report_script.main([str(analysis), "--output", str(output)])
    assert output.exists()
    with pytest.raises(SystemExit):
        report_script.main([str(analysis), "--output", str(output)])
    with pytest.raises(SystemExit):
        report_script.main([str(analysis), "--output", str(source / "new.json")])
    assert all(p.read_bytes() == data for p, data in inputs.items())
