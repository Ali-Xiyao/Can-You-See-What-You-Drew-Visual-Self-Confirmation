"""Audit first-round Naive selection gain over uniform sampling, using only CPU.

Unknown labels stay explicit. The selected label also enters the random-policy
denominator, so its uncertainty must not be varied independently in the two
terms. Completion bounds are not confidence intervals or a new acceptance gate.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.v4.spec import canonical_noun


def _interval(low: Fraction, high: Fraction) -> dict[str, Any]:
    return {"lower": float(low), "upper": float(high), "lower_exact": str(low),
            "upper_exact": str(high), "point": float(low) if low == high else None}


def pool_gain(candidates: Sequence[dict[str, Any]], selected_id: str) -> dict[str, Any]:
    """Require explicit labels; None plus a reason is different from an absent label."""
    if not candidates:
        raise ValueError("Candidate pool is empty")
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate ID")
    if selected_id not in ids:
        raise ValueError("Selected candidate is absent from the pool")
    for row in candidates:
        if "label" not in row:
            raise ValueError("Missing explicit candidate label")
        if row["label"] is not None and not isinstance(row["label"], bool):
            raise ValueError("Labels must be boolean or explicit None")
        if row["label"] is None and not row.get("unknown_reason"):
            raise ValueError("Unknown labels need an explicit reason")
    n = len(candidates)
    # Repeated identical pixels within one prompt share a single truth variable.
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        groups[row.get("rgb_sha256", row["candidate_id"])].append(row)
    known_gain = random_low = random_high = Fraction(0)
    gain_low = gain_high = Fraction(0)
    selected_low = selected_high = Fraction(0)
    for rows in groups.values():
        known = {row["label"] for row in rows if row["label"] is not None}
        if len(known) > 1:
            raise ValueError("Conflicting known labels for identical pixels in one prompt")
        label = next(iter(known)) if known else None
        is_selected = any(row["candidate_id"] == selected_id for row in rows)
        weight = Fraction(len(rows), n)
        coefficient = Fraction(int(is_selected)) - weight
        if label is None:
            gain_low += min(Fraction(0), coefficient)
            gain_high += max(Fraction(0), coefficient)
            random_high += weight
            selected_high += int(is_selected)
        else:
            value = int(label)
            known_gain += coefficient * value
            random_low += weight * value
            random_high += weight * value
            selected_low += int(is_selected) * value
            selected_high += int(is_selected) * value
    return {"n_candidates": n, "n_unique_rgb_or_label_groups": len(groups),
            "correct": sum(row["label"] is True for row in candidates),
            "wrong": sum(row["label"] is False for row in candidates),
            "unknown": sum(row["label"] is None for row in candidates),
            "selected_candidate_id": selected_id,
            "selected_label": next(row["label"] for row in candidates if row["candidate_id"] == selected_id),
            "uniform_candidate_weight": {"value": float(Fraction(1, n)), "exact": str(Fraction(1, n))},
            "selected_correctness": _interval(selected_low, selected_high),
            "uniform_correctness": _interval(random_low, random_high),
            "paired_gain": _interval(known_gain + gain_low, known_gain + gain_high),
            "bound_type": "joint label-completion bounds, not a confidence interval"}


def aggregate(pools: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not pools:
        return {"n_prompts": 0, "status": "no_prompts"}
    n = len(pools)
    result = {"n_prompts": n, "n_candidates": sum(row["n_candidates"] for row in pools),
              "prompt_ids": [row["prompt_id"] for row in pools],
              "prompt_weight": {"value": 1 / n, "exact": str(Fraction(1, n))},
              "n_canonical_scenes": len({row["canonical_scene_sha256"] for row in pools}),
              "selected_correct": sum(row["selected_label"] is True for row in pools),
              "selected_wrong": sum(row["selected_label"] is False for row in pools),
              "selected_unknown": sum(row["selected_label"] is None for row in pools)}
    for key in ("correct", "wrong", "unknown"):
        result[key + "_candidates"] = sum(row[key] for row in pools)
    for key in ("selected_correctness", "uniform_correctness", "paired_gain"):
        low = sum((Fraction(row[key]["lower_exact"]) for row in pools), Fraction(0)) / n
        high = sum((Fraction(row[key]["upper_exact"]) for row in pools), Fraction(0)) / n
        result[key] = _interval(low, high)
    result["bound_scope"] = "equal-prompt mean of joint per-pool bounds; conservative if labels depend across pools"
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: Sequence[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {name}: {value}")
        result[value] = row
    return result


def build_round000_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    directory = run_dir / "rounds/round-000"
    files = {"selection": directory / "selection.json", "done": directory / "DONE.json",
             "manifest": directory / "ladder/rfo_gold/manifest.jsonl",
             "verified": directory / "ladder/rfo_gold/verified.jsonl",
             "observations": directory / "observations/naive.jsonl",
             "integrity": run_dir / "runtime-check/first-update-integrity.json"}
    source_hashes = {str(path): sha256_file(path) for path in files.values()}
    selection = json.loads(files["selection"].read_text(encoding="utf-8"))
    done = json.loads(files["done"].read_text(encoding="utf-8"))
    integrity = json.loads(files["integrity"].read_text(encoding="utf-8"))
    if selection["round"] != 0 or done["round"] != 0:
        raise ValueError("This interface only audits the frozen initial round")
    for key in ("selection", "verified"):
        if integrity["provenance"][key + "_sha256"] != source_hashes[str(files[key])]:
            raise ValueError(f"First-update integrity audit no longer matches {key}")
    manifest = _index(_jsonl(files["manifest"]), "image_path", "manifest image")
    verified = _index(_jsonl(files["verified"]), "image_path", "verification label")
    if set(verified) - set(manifest):
        raise ValueError("Verification labels fall outside the manifest")
    gold_images = _index([{**row, "candidate_id": Path(path).stem} for path, row in manifest.items()],
                         "candidate_id", "Gold candidate")
    naive_observations = _index(_jsonl(files["observations"]), "candidate_id", "Naive observation")
    decisions = {arm: _index(selection["decisions"][arm], "prompt_id", arm + " prompt")
                 for arm in ("naive", "rfo_gold")}
    if set(decisions["naive"]) != set(decisions["rfo_gold"]):
        raise ValueError("Arms do not cover identical prompt IDs")
    expected_naive = [cid for row in decisions["naive"].values() for cid in row["candidate_pool_ids"]]
    expected_gold = [cid for row in decisions["rfo_gold"].values() for cid in row["candidate_pool_ids"]]
    if (len(set(expected_naive)) != len(expected_naive) or len(set(expected_gold)) != len(expected_gold)
            or set(expected_naive) != set(naive_observations) or set(expected_gold) != set(gold_images)):
        raise ValueError("Missing or duplicate candidate membership/observation")
    trained_ids = selection["paired_prompt_ids"]
    if len(trained_ids) != len(set(trained_ids)):
        raise ValueError("Duplicate paired prompt ID")
    common = {pid for pid in decisions["naive"] if all(
        decisions[arm][pid]["selected_candidate_id"] is not None for arm in decisions)}
    if (set(trained_ids) != common or done["paired"] != len(common)
            or any(row["selected_samples"] != len(common) for row in done["arms"])):
        raise ValueError("Saved trained subset disagrees with paired selection")
    audited_missing = _index(integrity["verification_coverage_audit"]["missing"],
                             "candidate_id", "audited missing label")
    absent_ids = {Path(path).stem for path in set(manifest) - set(verified)}
    if absent_ids != set(audited_missing):
        raise ValueError("Missing verification label was not explicitly audited")
    results = []
    image_sources = {}
    for pid in sorted(decisions["naive"]):
        decision, gold_decision = decisions["naive"][pid], decisions["rfo_gold"][pid]
        candidates = []
        scene = None
        for cid in decision["candidate_pool_ids"]:
            if not cid.startswith("naive-"):
                raise ValueError("Unexpected initial Naive candidate ID")
            gid = "rfo_gold" + cid[len("naive"):]
            if gid not in gold_decision["candidate_pool_ids"]:
                raise ValueError("Candidate has no matching Gold seed/index")
            item, observation = gold_images[gid], naive_observations[cid]
            if item["spec_id"] != pid or observation["prompt_id"] != pid:
                raise ValueError("Candidate prompt identity mismatch")
            expected_suffix = f"-{item['seed']}-{item['candidate_index']}"
            if not cid.endswith(expected_suffix):
                raise ValueError("Candidate seed/index mismatch")
            spec_counts = collections.Counter()
            for obj in item["spec"]["objects"]:
                spec_counts[(canonical_noun(obj["object"]), obj.get("color"))] += int(obj["count"])
            candidate_scene = sha256_json(sorted((noun, color, count) for (noun, color), count in spec_counts.items()))
            if scene is not None and scene != candidate_scene:
                raise ValueError("Pool spans different canonical scene specifications")
            scene = candidate_scene
            left, right = Path(observation["image_path"]), Path(item["image_path"])
            for path in (left, right):
                image_sources[str(path)] = sha256_file(path)
            left_hash, right_hash = rgb_sha256(left), rgb_sha256(right)
            if left_hash != right_hash or observation["observation"]["rgb_sha256"] != left_hash:
                raise ValueError("Naive/Gold decoded RGB correspondence failed")
            record = verified.get(item["image_path"])
            if record is None:
                missing = audited_missing[gid]
                if (missing["spec_id"], missing["seed"], missing["candidate_index"]) != (pid, item["seed"], item["candidate_index"]):
                    raise ValueError("Audited missing label identity mismatch")
                label, reason = None, "audited_missing_verification"
                label_source = {"path": str(files["integrity"]), "record_sha256": sha256_json(missing)}
            else:
                if record["spec_id"] != pid or "image_correct" not in record:
                    raise ValueError("Missing or mismatched verification label")
                if record.get("resolution") == "pending_human":
                    label, reason = None, "pending_human"
                elif isinstance(record["image_correct"], bool):
                    label, reason = record["image_correct"], None
                else:
                    raise ValueError("Invalid non-pending verification label")
                label_source = {"path": str(files["verified"]), "record_sha256": sha256_json(record),
                                "resolution": record.get("resolution")}
            candidates.append({"candidate_id": cid, "gold_candidate_id": gid, "label": label,
                               "unknown_reason": reason, "seed": item["seed"], "candidate_index": item["candidate_index"],
                               "naive_image_path": str(left), "gold_image_path": str(right), "rgb_sha256": left_hash,
                               "naive_file_sha256": image_sources[str(left)], "gold_file_sha256": image_sources[str(right)],
                               "label_source": label_source})
        if gold_decision["selected_candidate_id"] is not None:
            picked_gold = next((c for c in candidates if c["gold_candidate_id"] == gold_decision["selected_candidate_id"]), None)
            if picked_gold is None or picked_gold["label"] is not True:
                raise ValueError("Recorded Gold selection is not an adjudicated correct candidate")
        elif any(c["label"] is True for c in candidates):
            raise ValueError("Gold abstention conflicts with available correct candidates")
        gain = pool_gain(candidates, decision["selected_candidate_id"])
        for candidate in candidates:
            candidate["uniform_weight"] = gain["uniform_candidate_weight"]
            candidate["gain_coefficient_exact"] = str(Fraction(int(candidate["candidate_id"] == decision["selected_candidate_id"]))
                                                        - Fraction(1, len(candidates)))
        results.append({"prompt_id": pid, "canonical_scene_sha256": scene, "entered_training": pid in common,
                        "gold_selected_candidate_id": gold_decision["selected_candidate_id"],
                        "all_prompt_weight_exact": str(Fraction(1, len(decisions["naive"]))),
                        "training_prompt_weight_exact": str(Fraction(1, len(common))) if pid in common else "0",
                        **gain, "candidates": candidates})
    source_hashes.update(image_sources)
    if any(sha256_file(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Input files changed during CPU analysis")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir), "round": 0, "pools": results,
            "all_prompts": aggregate(results), "trained_prompts": aggregate([r for r in results if r["entered_training"]]),
            "unknown_reasons": dict(collections.Counter(c["unknown_reason"] for r in results for c in r["candidates"] if c["label"] is None)),
            "paired_rgb_verified_candidates": sum(len(r["candidates"]) for r in results),
            "input_sha256": source_hashes, "inputs_unchanged_after_analysis": True,
            "analysis_source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
            "limitations": ["completion bounds, not confidence intervals or a hypothesis test",
                            "a single small first-round sample cannot confirm positive or negative population gain",
                            "the trained subset is conditional on Gold having an available certified correct image",
                            "unknown labels are not imputed; historical selected images and tie-breaks are unchanged",
                            "duplicate pixels or related prompts do not increase independent sample size",
                            "this is retrospective descriptive evidence, not an independent new validation"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (args.run_dir, ROOT / "runs")):
        parser.error("--output must be outside run directories")
    if output.exists():
        parser.error("--output must be a new file; existing reports are never overwritten")
    report = build_round000_report(args.run_dir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(f"Created {output}; CPU only; unknown-completion bounds are not confidence intervals")
    for label in ("all_prompts", "trained_prompts"):
        row = report[label]
        if row["n_prompts"]:
            print(f"{label}: n={row['n_prompts']}, paired gain={row['paired_gain']['lower']:+.3%} to {row['paired_gain']['upper']:+.3%}")
        else:
            print(f"{label}: no prompts")


if __name__ == "__main__":
    main()
