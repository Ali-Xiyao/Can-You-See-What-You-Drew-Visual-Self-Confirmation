"""Fixed factual cohorts, image weighting and unavailable-answer regressions."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_factual_trajectory.py"
    spec = importlib.util.spec_from_file_location("factual_trajectory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question(qid="s:count:red", color="red", wanted="1", family="count"):
    return {"question_id": qid, "text": (
        f"How many {color} books are in this picture?" if family == "count" else
        f"Is there a {color} book in this picture? Answer A or B only.\nA. yes\nB. no"),
        "expected_answer": wanted, "choices": [] if family == "count" else ["yes", "no"],
        "family": family, "question_format": "open" if family == "count" else "forced_choice"}


def item(color="red"):
    return {"object": "book", "color": color}


def checkpoint(step, specs):
    """specs: (spec_id, actual questions, said values, detection items)."""
    data = {"manifest": [], "observations": [], "verifications": []}
    for sid, questions, responses, detections in specs:
        image = f"{sid}-{step}.png"
        data["manifest"].append({"spec_id": sid, "image_path": image})
        data["observations"].append({
            "prompt_id": sid, "image_path": image, "questions": copy.deepcopy(questions),
            "observation": {"answers": [{"question_id": q["question_id"],
                "normalized_answer": response, "error": None, "abstain": False}
                for q, response in zip(questions, responses)]}})
        data["verifications"].append({"spec_id": sid, "image_path": image,
            "detections": copy.deepcopy(detections), "resolution": "agreed", "disputed": []})
    return data


def four_points(specs):
    return {step: checkpoint(step, specs) for step in (0, 8, 16, 24)}


def test_unknown_scope_removed_from_every_checkpoint_and_crop_remains_known(script):
    red, blue = question(), question("s:count:blue", "blue")
    data = four_points([("s", [red, blue], ["1", "1"], [item(), item("blue")])])
    disputed = data[8]["verifications"][0]
    disputed.update(detections=[item("blue")], resolution="agreed_verdict", disputed=[item()])
    crop = data[16]["verifications"][0]
    crop.update(resolution="resolved_by_crop", disputed=[item("blue")])
    result = script.arm_report(data)
    group = result["groups"]["count"]
    assert group["cohort"]["retained_questions"] == 1
    assert group["cohort"]["excluded_questions"] == 1
    assert group["cohort"]["partially_retained_image_slots"] == ["s"]
    assert group["cohort"]["excluded_question_reason_union"] == {"disputed_scope": 1}
    assert group["checkpoint_coverage_only"]["8"]["known_questions"] == 1
    assert group["checkpoint_coverage_only"]["16"]["known_questions"] == 2
    assert all(p["n_questions"] == 1 and p["means"]["response_matches_fact"] == 1
               for p in group["fixed_cohort_trajectories"].values())
    excluded = next(r for r in result["records"] if r["question_id"] == red["question_id"])
    assert excluded["checkpoints"]["8"]["fact_answer"] is None
    assert excluded["checkpoints"]["8"]["response_matches_fact"] is None


def test_three_metrics_average_within_image_then_weight_images_equally(script):
    qs = [question(f"b:count:{color}", color) for color in ("red", "blue", "green")]
    data = four_points([
        ("a", [question("a:count:red")], ["1"], [item()]),
        ("b", qs, ["1", "1", "1"], []),
    ])
    p = script.arm_report(data)["groups"]["all"]["fixed_cohort_trajectories"]["0"]
    assert p["n_images"] == 2 and p["n_questions"] == 4
    assert p["means"] == {"response_matches_request": 1.0,
                          "fact_matches_request": 0.5, "response_matches_fact": 0.5}
    assert [image["n_questions"] for image in p["per_image"]] == [1, 3]
    assert p["means"]["fact_matches_request"] != 1 / 4  # Question pooling is wrong.


def test_missing_error_and_abstain_responses_stay_in_fixed_denominator(script):
    names = ["ok", "row_missing", "answer_missing", "row_error", "observation_error",
             "answer_error", "abstain", "null", "empty"]
    data = four_points([(sid, [question(f"{sid}:count:red")], ["1"], [item()]) for sid in names])
    rows = {row["prompt_id"]: row for row in data[8]["observations"]}
    data[8]["observations"].remove(rows["row_missing"])
    rows["answer_missing"]["observation"]["answers"] = []
    rows["row_error"]["error"] = "failed"
    del rows["row_error"]["questions"]  # Recover stable schedule from other checkpoints.
    rows["observation_error"]["observation"]["error"] = "failed"
    rows["answer_error"]["observation"]["answers"][0]["error"] = "failed"
    rows["abstain"]["observation"]["answers"][0]["abstain"] = True
    rows["null"]["observation"]["answers"][0]["normalized_answer"] = None
    rows["empty"]["observation"]["answers"][0]["normalized_answer"] = " "
    result = script.arm_report(data)["groups"]["count"]
    assert result["cohort"]["retained_questions"] == len(names)
    p = result["fixed_cohort_trajectories"]["8"]
    assert p["means"] == {"response_matches_request": 1 / 9,
                          "fact_matches_request": 1.0, "response_matches_fact": 1 / 9}
    assert p["answer_status"] == {"available": 1, "missing_observation": 1,
        "missing_answer": 1, "observation_error": 2, "answer_error": 1,
        "abstain": 1, "null_answer": 1, "empty_answer": 1}


@pytest.mark.parametrize("field,value", [
    ("text", "How many blue books are in this picture?"),
    ("expected_answer", "2"), ("choices", ["one", "two"]),
    ("question_id", "s:count:other"), ("question_format", "forced_choice"),
])
def test_question_identity_or_meaning_drift_rejected(script, field, value):
    data = four_points([("s", [question()], ["1"], [item()])])
    data[16]["observations"][0]["questions"][0][field] = value
    with pytest.raises(ValueError, match="Question meaning or ID changed"):
        script.arm_report(data)


def test_no_common_facts_has_null_means_and_explicit_excluded_images(script):
    data = four_points([("s", [question()], ["1"], [item()])])
    data[8]["verifications"] = []
    data[16]["verifications"][0]["detections"] = [{"object": "unnameable"}]
    result = script.arm_report(data)
    group = result["groups"]["all"]
    assert group["cohort"]["retained_questions"] == 0
    assert group["cohort"]["excluded_image_slots"] == ["s"]
    assert group["cohort"]["excluded_question_reason_union"] == {
        "missing_verification": 1, "unnameable": 1}
    assert all(p["means"] == dict.fromkeys(script.METRICS)
               for p in group["fixed_cohort_trajectories"].values())


def test_arm_cohorts_remain_separate_and_existence_is_a_separate_family(script):
    qs = [question(), question("s:exists:red", wanted="yes", family="existence")]
    naive = four_points([("s", qs, ["1", "yes"], [item()])])
    gold = copy.deepcopy(naive)
    gold[8]["verifications"][0].update(resolution="agreed_verdict", disputed=[item()])
    result = script.factual_trajectory_report({"naive": naive, "rfo_gold": gold})
    assert result["arms"]["naive"]["groups"]["all"]["cohort"]["retained_questions"] == 2
    assert result["arms"]["rfo_gold"]["groups"]["all"]["cohort"]["retained_questions"] == 0
    assert result["arms"]["naive"]["groups"]["existence"]["cohort"]["retained_questions"] == 1


def write_data(root, data):
    (root / "split.json").write_text(json.dumps({"outcome": [
        row["spec_id"] for row in next(iter(data.values()))["manifest"]]}), encoding="utf-8")
    for step, cp in data.items():
        directory = root / "evaluations" / "naive" / f"step-{step:05d}"
        directory.mkdir(parents=True)
        for key, filename in (("manifest", "manifest.jsonl"), ("observations", "s_select.jsonl"),
                              ("verifications", "verified.jsonl")):
            (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in cp[key]),
                                               encoding="utf-8")


def test_build_hashes_inputs_and_refuses_changed_source(script, tmp_path, monkeypatch):
    data = four_points([("s", [question()], ["1"], [item()])])
    write_data(tmp_path, data)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.jsonl")}
    report = script.build_report(tmp_path, arms=("naive",))
    assert len(report["input_sha256"]) == 13
    assert report["input_hashes_unchanged_after_analysis"]
    assert all(p.read_bytes() == content for p, content in before.items())
    original_read = script._read_jsonl

    def changing_read(path):
        result = original_read(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(script, "_read_jsonl", changing_read)
    with pytest.raises(ValueError, match="Input files changed"):
        script.build_report(tmp_path, arms=("naive",))


def test_build_rejects_same_missing_spec_at_every_checkpoint(script, tmp_path):
    data = four_points([("s", [question()], ["1"], [item()])])
    write_data(tmp_path, data)
    (tmp_path / "split.json").write_text(json.dumps({"outcome": ["s", "missing"]}),
                                          encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest does not match frozen outcome IDs"):
        script.build_report(tmp_path, arms=("naive",))


def test_cli_only_writes_new_file_outside_runs(script, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "build_report", lambda path: {"ok": True})
    run = tmp_path / "run"
    output = tmp_path / "review" / "report.json"
    script.main([str(run), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    assert "Wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main([str(run), "--output", str(output)])
    with pytest.raises(SystemExit):
        script.main([str(run), "--output", str(run / "new.json")])
    with pytest.raises(SystemExit):
        script.main([str(run), "--output", str(script.PROJECT / "runs" / "new.json")])
