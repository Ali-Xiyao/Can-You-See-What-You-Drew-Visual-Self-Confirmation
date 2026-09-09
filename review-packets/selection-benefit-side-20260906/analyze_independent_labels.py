"""Paired selected-vs-sampled-control diagnosis; pending labels remain unknown."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selfsight.v4.spec import canonical_noun


def interval(low, high):
    return {"lower": float(low), "upper": float(high), "lower_exact": str(low), "upper_exact": str(high), "point": float(low) if low == high else None}


def joint_gain(terms, n):
    """Same pixels and gold specification share a truth variable across terms."""
    if n < 1:
        raise ValueError("No pools")
    groups = defaultdict(list)
    for term in terms:
        if term["label"] is not None and type(term["label"]) is not bool:
            raise ValueError("Label must be boolean or None")
        groups[term["identity"]].append(term)
    lower = upper = F(0)
    for rows in groups.values():
        labels = {r["label"] for r in rows if r["label"] is not None}
        if len(labels) > 1:
            raise ValueError("Conflicting settled labels for identical pixels and gold")
        coefficient = sum((F(r["coefficient"]) for r in rows), F(0)) / n
        label = next(iter(labels)) if labels else None
        if label is None:
            lower += min(F(0), coefficient)
            upper += max(F(0), coefficient)
        else:
            lower += coefficient * int(label)
            upper += coefficient * int(label)
    return interval(lower, upper)


def interpret(record):
    if record is None:
        return None, "missing_verdict"
    resolution = record.get("resolution")
    if resolution == "pending_human":
        return None, "pending_human"
    if resolution not in {"agreed", "agreed_verdict", "resolved_by_crop", "human"}:
        raise ValueError(f"Unrecognized resolution: {resolution}")
    if type(record.get("image_correct")) is not bool:
        raise ValueError("Settled verdict lacks a boolean label")
    return record["image_correct"], None


def self_test():
    def t(key, coefficient, label):
        return {"identity": key, "coefficient": coefficient, "label": label}
    assert joint_gain([t("same", 1, None), t("same", -1, None)], 1)["point"] == 0
    assert joint_gain([t("a", 1, True), t("b", -1, False)], 1)["point"] == 1
    assert joint_gain([t("a", 1, False), t("b", -1, True)], 1)["point"] == -1
    assert joint_gain([t("a", 1, None), t("b", -1, None)], 1) == interval(F(-1), F(1))
    assert joint_gain([t("a", 1, True), t("b", -1, None)], 2) == interval(F(0), F(1, 2))
    assert joint_gain([t("a", 1, None), t("a", -1, None), t("b", 1, True), t("c", -1, False)], 2)["point"] == .5
    assert joint_gain([t("a", 1, None), t("a", -1, True)], 1)["point"] == 0
    assert interpret({"resolution": "pending_human", "image_correct": True}) == (None, "pending_human")
    assert interpret({"resolution": "agreed_verdict", "image_correct": False}) == (False, None)
    assert interpret(None) == (None, "missing_verdict")
    for terms in ([t("a", 1, True), t("a", -1, False)], [t("a", 1, "false")]):
        try:
            joint_gain(terms, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid labels accepted")
    for record in ({"resolution": "unexpected", "image_correct": True}, {"resolution": "human", "image_correct": 1}):
        try:
            interpret(record)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid verdict accepted")
    print("14 targeted checks passed: pending, missing, same-image correlation, shared-label bounds, sign and denominator.")


def main():
    inputs = {}
    def read(path, jsonl=False):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line] if jsonl else json.loads(raw)
    def scene(spec):
        counts = Counter()
        for obj in spec["objects"]:
            counts[canonical_noun(obj["object"]), obj.get("color")] += obj["count"]
        payload = json.dumps(sorted((noun, color, n) for (noun, color), n in counts.items()))
        return hashlib.sha256(payload.encode()).hexdigest()

    folder = SIDE / "independent-labels"
    state = read(folder / "state.json")
    assert state["status"] == "detector_ladder_complete", "Wait for the complete detector ladder"
    for stage in ("detect.qwen3vl", "detect.internvl", "crop", "verify"):
        assert read(folder / f"{stage}.complete.json")["exit_code"] == 0
    public = read(SIDE / "followup-label-request.public.jsonl", True)
    manifest = read(folder / "manifest.jsonl", True)
    assert public == manifest
    by_id = {r["review_id"]: r for r in public}
    assert len(by_id) == len(public) == 44
    pairs = read(SIDE / "followup-label-pairs.private.json")
    verdict_rows = read(folder / "verified.jsonl", True)
    verdicts = {r["image_path"]: r for r in verdict_rows}
    assert len(verdicts) == len(verdict_rows)
    assert set(verdicts) <= {r["image_path"] for r in public}
    # Reuse only requested-spec metadata for the two same-image pairs which
    # require no image label to know their selection-minus-control gain is zero.
    specs = {}
    for index in (1, 2):
        rows = read(ROOT / f"runs/v4/decoupling-pilot-20260906/rounds/round-{index:03d}/ladder/rfo_gold/manifest.jsonl", True)
        for row in rows:
            key = (index, row["spec_id"])
            assert key not in specs or specs[key] == row["spec"]
            specs[key] = row["spec"]
    per_pool, review_needed = [], {}
    for pair in pairs:
        spec = specs[pair["round"], pair["prompt_id"]]
        scene_id = scene(spec)
        terms, details = [], {}
        for role, coefficient in (("selected", 1), ("uniform_control", -1)):
            if pair["same_candidate"]:
                key = f"{scene_id}:unsubmitted-same-candidate:{pair['selected_candidate_id']}"
                label, reason, case = None, "same_candidate_not_submitted", None
            else:
                case = by_id[pair["labels"][role]["review_id"]]
                assert case["spec"] == spec
                path = Path(case["image_path"])
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                assert inputs[str(path)] == case["image_file_sha256"]
                key = scene_id + ":" + case["recorded_rgb_sha256"]
                record = verdicts.get(case["image_path"])
                if record is not None:
                    assert record["spec_id"] == pair["prompt_id"]
                    assert (record["seed"], record["candidate_index"]) == (case["seed"], case["candidate_index"])
                label, reason = interpret(record)
                if label is None:
                    review_needed[case["review_id"]] = {**case, "unknown_reason": reason, "verdict": record}
            terms.append({"identity": key, "coefficient": coefficient, "label": label})
            details[role] = {"candidate_id": pair[role + "_candidate_id"], "label": label, "unknown_reason": reason, "review_id": case["review_id"] if case else None}
        per_pool.append({"round": pair["round"], "selector_optimizer_step": pair["selector_optimizer_step"], "prompt_id": pair["prompt_id"], "scene_sha256": scene_id, "entered_training": pair["entered_training"], "same_candidate": pair["same_candidate"], "terms": terms, "labels": details, "paired_gain": joint_gain(terms, 1)})
    scopes = {}
    for index in (1, 2, None):
        for trained in (False, True):
            rows = [r for r in per_pool if (index is None or r["round"] == index) and (not trained or r["entered_training"])]
            name = (f"round_{index}" if index else "pooled_two_rounds") + ("_entered_training" if trained else "_all_pools")
            terms = [t for row in rows for t in row["terms"]]
            points = [row["paired_gain"]["point"] for row in rows]
            scopes[name] = {"n_pools": len(rows), "n_scenes": len({r["scene_sha256"] for r in rows}),
                           "sampled_control_gain_completion_bounds": joint_gain(terms, len(rows)),
                           "wins": sum(p is not None and p > 0 for p in points), "losses": sum(p is not None and p < 0 for p in points), "ties": sum(p == 0 for p in points), "unknown_gain_pools": sum(p is None for p in points),
                           "selected_known_correct": sum(r["labels"]["selected"]["label"] is True for r in rows), "selected_known_wrong": sum(r["labels"]["selected"]["label"] is False for r in rows),
                           "control_known_correct": sum(r["labels"]["uniform_control"]["label"] is True for r in rows), "control_known_wrong": sum(r["labels"]["uniform_control"]["label"] is False for r in rows)}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in inputs.items())
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis": "Post-training diagnostic using the previously fixed sampled uniform controls", "scopes": scopes, "n_images_needing_further_review": len(review_needed), "per_pool": per_pool, "input_sha256": inputs, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": ["Completion bounds concern unknown labels in these sampled pairs; they are not confidence intervals.", "Each control is one preselected uniform draw; this adds control-sampling uncertainty relative to enumerating the full pool.", "Small selected cohorts do not establish population-level selection advantage or a reliable training-time divergence.", "Rounds have different prompts; pooled results are a mixture of two checkpoints, not a time point.", "The entered-training cohort is conditioned on the Gold arm having usable candidates.", "No new pass threshold, event declaration, replacement policy or fabricated human label."]}
    for name, data in (("paired-gain-initial.json", result), ("paired-gain-initial.human-needed.json", list(review_needed.values()))):
        with (SIDE / name).open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps({"scopes": scopes, "n_images_needing_further_review": len(review_needed)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else main()
