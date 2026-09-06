"""Selection gain must share unknown labels and preserve frozen evidence."""
from __future__ import annotations

import importlib.util
import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from selfsight.utils.hashing import rgb_sha256, sha256_file

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("selection_gain_under_test", REPO / "scripts/v4_selection_gain.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def candidates(labels):
    return [{"candidate_id": f"c{i}", "label": label,
             "unknown_reason": "pending" if label is None else None} for i, label in enumerate(labels)]


def test_selected_unknown_is_shared_with_the_uniform_denominator():
    report = SCRIPT.pool_gain(candidates([None, True, None, False]), "c0")
    assert report["selected_correctness"]["lower"] == 0
    assert report["selected_correctness"]["upper"] == 1
    assert report["uniform_correctness"]["lower"] == 0.25
    assert report["uniform_correctness"]["upper"] == 0.75
    assert report["paired_gain"]["lower_exact"] == "-1/2"
    assert report["paired_gain"]["upper_exact"] == "1/2"
    assert report["paired_gain"]["point"] is None


def test_nonselected_unknown_changes_only_the_random_term():
    report = SCRIPT.pool_gain(candidates([False, True, None, False]), "c0")
    assert report["paired_gain"]["lower_exact"] == "-1/2"
    assert report["paired_gain"]["upper_exact"] == "-1/4"


def test_fully_known_pool_has_an_exact_gain():
    assert SCRIPT.pool_gain(candidates([True, False, True, False]), "c0")["paired_gain"]["point"] == 0.5
    assert SCRIPT.pool_gain(candidates([True, False, True, False]), "c1")["paired_gain"]["point"] == -0.5


def test_joint_bounds_equal_exhaustive_unknown_completions():
    for n in range(2, 5):
        for labels in itertools.product((False, True, None), repeat=n):
            unknown = [i for i, value in enumerate(labels) if value is None]
            for selected in range(n):
                gains = []
                for completion in itertools.product((False, True), repeat=len(unknown)):
                    values = list(labels)
                    for index, value in zip(unknown, completion):
                        values[index] = value
                    gains.append(Fraction(int(values[selected])) - Fraction(sum(values), n))
                result = SCRIPT.pool_gain(candidates(labels), f"c{selected}")
                assert Fraction(result["paired_gain"]["lower_exact"]) == min(gains)
                assert Fraction(result["paired_gain"]["upper_exact"]) == max(gains)


@pytest.mark.parametrize("rows,selected,message", [
    ([{"candidate_id": "a"}], "a", "Missing explicit"),
    ([{"candidate_id": "a", "label": None}], "a", "explicit reason"),
    ([{"candidate_id": "a", "label": 1}], "a", "boolean"),
    ([{"candidate_id": "a", "label": True}] * 2, "a", "Duplicate candidate"),
    (candidates([True]), "absent", "Selected candidate is absent"),
])
def test_missing_duplicate_and_invalid_labels_fail_closed(rows, selected, message):
    with pytest.raises(ValueError, match=message):
        SCRIPT.pool_gain(rows, selected)


def test_repeated_pixels_share_one_unknown_label_and_known_conflicts_fail():
    rows = [{**row, "rgb_sha256": "same"} for row in candidates([None, None])]
    assert SCRIPT.pool_gain(rows, "c0")["paired_gain"]["point"] == 0.0
    rows = [{**row, "rgb_sha256": "same"} for row in candidates([True, False])]
    with pytest.raises(ValueError, match="Conflicting known labels"):
        SCRIPT.pool_gain(rows, "c0")


def test_aggregate_weights_prompts_equally_even_when_pool_sizes_differ():
    left = {"prompt_id": "p1", "canonical_scene_sha256": "s1",
            **SCRIPT.pool_gain(candidates([True, False]), "c0")}
    right = {"prompt_id": "p2", "canonical_scene_sha256": "s2",
             **SCRIPT.pool_gain(candidates([False, True, True, True]), "c0")}
    report = SCRIPT.aggregate([left, right])
    assert report["paired_gain"]["point"] == -0.125
    assert report["prompt_weight"]["exact"] == "1/2"


def make_run(tmp_path):
    run = tmp_path / "run"
    base = run / "rounds/round-000"
    ladder = base / "ladder/rfo_gold"
    ladder.mkdir(parents=True)
    (base / "observations").mkdir()
    (run / "runtime-check").mkdir()
    manifest, verified, observations, missing = [], [], [], []
    decisions = {"naive": [], "rfo_gold": []}
    for pi, pid in enumerate(("good-pool", "no-gold-pool")):
        ids = {"naive": [], "rfo_gold": []}
        for index in range(2):
            seed = 100 + pi * 10 + index
            images = {}
            for arm in ids:
                cid = f"{arm}-r000-{pid}-{seed}-{index}"
                ids[arm].append(cid)
                path = base / "candidates" / arm / f"{cid}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (3, 3), (20 + pi * 40 + index, 40, 60)).save(path)
                images[arm] = path
            metadata = {"spec_id": pid, "seed": seed, "candidate_index": index,
                        "image_path": str(images["rfo_gold"]),
                        "spec": {"objects": [{"object": "book", "color": "red" if pi == 0 else "blue", "count": 1}]}}
            manifest.append(metadata)
            if pi == 1 and index == 1:
                missing.append({"spec_id": pid, "seed": seed, "candidate_index": index,
                                "candidate_id": ids["rfo_gold"][-1]})
            else:
                verified.append({"image_path": metadata["image_path"], "spec_id": pid,
                                 "image_correct": pi == 0, "resolution": "pending_human" if index == 1 else "agreed"})
            observations.append({"candidate_id": ids["naive"][-1], "prompt_id": pid,
                                 "image_path": str(images["naive"]),
                                 "observation": {"rgb_sha256": rgb_sha256(images["naive"])}})
        decisions["naive"].append({"prompt_id": pid, "candidate_pool_ids": ids["naive"],
                                   "selected_candidate_id": ids["naive"][1 if pi == 0 else 0]})
        decisions["rfo_gold"].append({"prompt_id": pid, "candidate_pool_ids": ids["rfo_gold"],
                                      "selected_candidate_id": ids["rfo_gold"][0] if pi == 0 else None})
    for path, rows in ((ladder / "manifest.jsonl", manifest), (ladder / "verified.jsonl", verified),
                       (base / "observations/naive.jsonl", observations)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selection = base / "selection.json"
    selection.write_text(json.dumps({"round": 0, "paired_prompt_ids": ["good-pool"], "decisions": decisions}), encoding="utf-8")
    (base / "DONE.json").write_text(json.dumps({"round": 0, "paired": 1,
                                                "arms": [{"selected_samples": 1}, {"selected_samples": 1}]}), encoding="utf-8")
    (run / "runtime-check/first-update-integrity.json").write_text(json.dumps({
        "provenance": {"selection_sha256": sha256_file(selection), "verified_sha256": sha256_file(ladder / "verified.jsonl")},
        "verification_coverage_audit": {"missing": missing}}), encoding="utf-8")
    return run


def test_end_to_end_report_retains_all_and_gold_filtered_subsets_without_input_writes(tmp_path):
    run = make_run(tmp_path)
    before = {str(path): sha256_file(path) for path in run.rglob("*") if path.is_file()}
    output = tmp_path / "report.json"
    SCRIPT.main([str(run), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["all_prompts"]["n_prompts"] == 2
    assert report["trained_prompts"]["n_prompts"] == 1
    assert report["paired_rgb_verified_candidates"] == 4
    assert report["unknown_reasons"] == {"pending_human": 1, "audited_missing_verification": 1}
    assert report["all_prompts"]["paired_gain"]["lower_exact"] == "-1/2"
    assert report["all_prompts"]["paired_gain"]["upper_exact"] == "0"
    assert report["inputs_unchanged_after_analysis"] is True
    assert before == {str(path): sha256_file(path) for path in run.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        SCRIPT.main([str(run), "--output", str(run / "new.json")])
    with pytest.raises(SystemExit):
        SCRIPT.main([str(run), "--output", str(output)])


def test_pixel_mismatch_rejects_transferring_gold_labels(tmp_path):
    run = make_run(tmp_path)
    image = next((run / "rounds/round-000/candidates/rfo_gold").glob("*.png"))
    Image.new("RGB", (3, 3), (255, 0, 0)).save(image)
    with pytest.raises(ValueError, match="RGB correspondence failed"):
        SCRIPT.build_round000_report(run)


def test_unregistered_missing_verification_label_is_rejected(tmp_path):
    run = make_run(tmp_path)
    verified = run / "rounds/round-000/ladder/rfo_gold/verified.jsonl"
    lines = verified.read_text(encoding="utf-8").splitlines()
    verified.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    audit_path = run / "runtime-check/first-update-integrity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["provenance"]["verified_sha256"] = sha256_file(verified)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing verification label was not explicitly audited"):
        SCRIPT.build_round000_report(run)
