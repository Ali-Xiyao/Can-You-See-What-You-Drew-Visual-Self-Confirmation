"""Historical sensitivity must retain partial truth and a paired baseline."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("accuracy_sensitivity_under_test",
                                                REPO / "scripts/v4_accuracy_sensitivity.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture(kind="count", choices=None):
    question = {"question_id": "q", "atom_id": "legacy:exists:book",
                "text": "How many red books are in this picture?" if kind == "count"
                        else "Is there a red book in this picture?",
                "expected_answer": "1" if kind == "count" else "yes", "choices": choices or []}
    if choices:
        question["expected_answer"] = choices[0]
    pools, found, observations = {}, {}, []
    for prompt in ("known", "disputed"):
        pools[prompt] = {"prompt_id": prompt, "questions": [question],
                         "candidates": [{"candidate_id": label, "image_path": f"{prompt}-{label}",
                                         "correct": label == "good"} for label in ("good", "bad")],
                         "selection": {"dropped": False, "scores": {"naive": {"good": 1.0, "bad": 1.0}}}}
        for label in ("good", "bad"):
            found[f"{prompt}-{label}"] = {
                "resolution": "agreed", "detections": [{"object": "book", "color": "red"
                                                           if label == "good" else "blue"}]}
            observations.append({"prompt_id": prompt, "candidate_id": label, "observation": {"answers": [
                {"question_id": "q", "normalized_answer": question["expected_answer"],
                 "abstain": False, "error": None}]}})
    found["disputed-bad"] = {"resolution": "agreed_verdict", "detections": [],
                             "disputed": [{"object": "book", "color": "red"}]}
    return pools, found, observations


def test_disputed_core_is_unknown_and_every_simulation_uses_the_same_complete_pools():
    pools, found, observations = fixture()
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["eligible_pool_ids"] == ["known"]
    assert report["coverage"]["total_questions"] == 4
    assert report["coverage"]["known_questions"] == 3
    assert report["coverage"]["unknown_reasons"] == {"disputed_scope": 1}
    assert report["coverage"]["eligible_questions"] == 2
    actual, ideal = report["curve"]
    assert actual["fact_accuracy"]["mean"] == 0.5
    assert ideal["fact_accuracy"]["mean"] == 1.0
    assert actual["metrics"]["sep_equal"]["mean"] == 1.0
    assert ideal["metrics"]["sep_higher"]["mean"] == 1.0
    assert actual["n_pools"] == ideal["n_pools"] == 1


def test_legacy_atom_id_does_not_remove_colour_from_existence_fact():
    pools, found, observations = fixture(kind="existence")
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["curve"][0]["fact_accuracy"]["mean"] == 0.5
    assert report["curve"][1]["metrics"]["sep_higher"]["mean"] == 1.0


def test_legacy_number_words_are_mapped_back_to_the_recorded_choice_representation():
    pools, found, observations = fixture(choices=["one", "zero"])
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["baseline_score_mismatches"] == 0
    assert report["curve"][0]["metrics"]["ceiling"]["mean"] == 1.0
    assert report["curve"][1]["metrics"]["ceiling"]["mean"] == 0.5


@pytest.mark.parametrize("verification,reason", [
    (None, "missing_verification"),
    ({"resolution": "human", "detections": [{"object": "unnameable"}]}, "unnameable"),
    ({"resolution": "pending_human", "detections": []}, "unresolved_verification"),
])
def test_missing_unnameable_and_unresolved_facts_never_become_zero(verification, reason):
    pools, found, observations = fixture()
    found["known-bad"] = verification
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["coverage"]["unknown_reasons"][reason] == 1
    assert report["coverage"]["total_questions"] == 4
    assert report["eligible_pool_ids"] == []
    assert all(row["fact_accuracy"]["mean"] is None for row in report["curve"])


def test_resolved_crop_can_retain_historical_disputed_keys_without_exclusion():
    pools, found, observations = fixture()
    found["disputed-bad"]["resolution"] = "resolved_by_crop"
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["eligible_pool_ids"] == ["disputed", "known"]
    assert report["coverage"]["unknown_reasons"] == {}
    assert report["curve"][1]["fact_accuracy"]["mean"] == 1.0


@pytest.mark.parametrize("bad_answer,reason", [
    (None, "missing_answer"),
    ({"normalized_answer": None, "abstain": True}, "abstained_answer"),
    ({"normalized_answer": "1", "error": "failed"}, "answer_error"),
])
def test_unavailable_actual_answers_exclude_the_same_pool_from_every_p(bad_answer, reason):
    pools, found, observations = fixture()
    row = next(r for r in observations if r["prompt_id"] == "known" and r["candidate_id"] == "bad")
    row["observation"]["answers"] = [] if bad_answer is None else [{"question_id": "q", **bad_answer}]
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["coverage"]["unavailable_actual_reasons"] == {reason: 1}
    assert report["coverage"]["total_questions"] == 4
    assert all(row["n_pools"] == 0 for row in report["curve"])


def test_unexpressible_legacy_truth_is_known_but_not_mixed_into_closed_choices():
    pools, found, observations = fixture(choices=["one", "zero"])
    found["known-bad"]["detections"] = [{"object": "book", "color": "red"}] * 2
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0, 1.0], seeds=2)
    assert report["coverage"]["known_questions"] == 3
    assert report["coverage"]["truth_not_in_choices_questions"] == 1
    assert report["excluded_pool_reasons"]["known"] == ["truth_not_in_choices"]


def test_baseline_mismatch_fails_instead_of_silently_changing_the_old_score():
    pools, found, observations = fixture()
    pools["known"]["selection"]["scores"]["naive"]["bad"] = 0.0
    with pytest.raises(ValueError, match="p=0 score differs"):
        module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0], seeds=2)


def test_duplicate_observed_answers_fail_closed():
    pools, found, observations = fixture()
    observations[0]["observation"]["answers"] *= 2
    with pytest.raises(ValueError, match="Duplicate observed question"):
        module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0], seeds=2)


def test_replacement_masks_are_nested_and_do_not_depend_on_excluded_pools_or_order():
    pools, found, observations = fixture()
    script = module()
    fractions = [0.0, 0.1, 0.5, 1.0]
    full = script.sensitivity_report(pools, found, observations, arm="naive", grid=fractions, seeds=20)
    reduced = script.sensitivity_report({"known": pools["known"]}, found,
                                        list(reversed([r for r in observations if r["prompt_id"] == "known"])),
                                        arm="naive", grid=fractions, seeds=20)
    assert full["curve"] == reduced["curve"]
    for seed in range(20):
        lower = full["curve"][1]["replicates"][seed]
        upper = full["curve"][2]["replicates"][seed]
        assert lower["replaced_answers"] <= upper["replaced_answers"]
        assert lower["fact_accuracy"] <= upper["fact_accuracy"]


def test_cli_preserves_fact_unknown_and_all_answer_error_levels_without_input_writes(tmp_path):
    pools, found, observations = fixture()
    observations[0]["observation"] = None
    observations[1]["error"] = ""
    observations[2]["observation"]["error"] = ""
    observations[3]["observation"]["answers"][0]["normalized_answer"] = "   "
    outdir, source = tmp_path / "pool-run", tmp_path / "verification-run"
    outdir.mkdir()
    source.mkdir()
    files = {outdir / "pools.jsonl": list(pools.values()),
             outdir / "selection.jsonl": [{"prompt_id": pid, **p["selection"]} for pid, p in pools.items()],
             outdir / "observations.naive.jsonl": observations,
             source / "verified.jsonl": [{"image_path": path, **record} for path, record in found.items()]}
    for path, rows in files.items():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    runs = outdir / "runs.json"
    runs.write_text(json.dumps({"runs": [str(source)]}), encoding="utf-8")
    inputs = [*files, runs]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    script = module()
    output = tmp_path / "new-diagnostic.json"
    script.main([str(outdir), "--arm", "naive", "--json-output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["coverage"]["known_questions"] == 3
    assert report["coverage"]["unknown_reasons"] == {"disputed_scope": 1}
    assert report["coverage"]["unavailable_actual_reasons"] == {
        "missing_observation": 1, "record_error": 1, "observation_error": 1, "empty_answer": 1}
    assert report["eligible_pool_ids"] == []
    for directory in (outdir, source):
        with pytest.raises(SystemExit) as error:
            script.main([str(outdir), "--arm", "naive", "--json-output", str(directory / "new.json")])
        assert error.value.code == 2
        assert not (directory / "new.json").exists()
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs} == before


def test_missing_record_and_empty_answer_error_are_reported_independently_of_truth():
    pools, found, observations = fixture()
    observations.pop(0)
    observations[0]["observation"]["answers"][0]["error"] = ""
    report = module().sensitivity_report(pools, found, observations, arm="naive", grid=[0.0], seeds=2)
    assert report["coverage"]["unknown_reasons"] == {"disputed_scope": 1}
    assert report["coverage"]["unavailable_actual_reasons"] == {"missing_observation": 1, "answer_error": 1}
