"""CPU-only, retrospective selection-benefit audit, bounded to steps 0/8/16/24.

Writes only this side review's artifacts. Does not alter training, labels, or
registered reports. Reuses the first-round RGB provenance audit and independently
enumerates its unknown labels. The fixed probe bank is a separate population.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/v4/decoupling-pilot-20260906"
STEPS = (0, 8, 16, 24)
SEED, RESAMPLES = 20260906, 2000
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    payload = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def value(x):
    return {"value": float(x), "exact": str(x)}


def bounds(xs):
    return {"low": value(min(xs)), "high": value(max(xs))}


def cluster_bootstrap(values, groups):
    """Resample whole scenes; preserve prompt weights within each draw."""
    vals = np.asarray(values, dtype=float)
    clusters = sorted(set(groups))
    members = [np.flatnonzero(np.asarray(groups) == group) for group in clusters]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(clusters), size=(RESAMPLES, len(clusters)))
    means = [float(vals[np.concatenate([members[i] for i in draw])].mean()) for draw in draws]
    return {"point": float(vals.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)), "n_clusters": len(clusters),
            "resamples": RESAMPLES, "seed": SEED,
            "scope": "Retrospective percentile scene bootstrap, conditional on this selected bank; no repeated-look adjustment or training-seed uncertainty."}


def random_tail(probabilities, observed):
    """Exact count distribution of independently uniform choices per fixed pool."""
    distribution = [F(1)]
    for prob in probabilities:
        nxt = [F(0)] * (len(distribution) + 1)
        for k, old in enumerate(distribution):
            nxt[k] += old * (1 - prob)
            nxt[k + 1] += old * prob
        distribution = nxt
    assert sum(distribution) == 1
    return value(sum(distribution[observed:]))


def initial_round():
    source = ROOT / "review-packets/selector-gain-20260906/round000.json"
    report = read(source)
    decisions = read(RUN / "rounds/round-000/selection.json")
    decision_by_id = {row["prompt_id"]: row for row in decisions["decisions"]["naive"]}
    # Verify the small raw evidence files. Prior audit covers decoded RGB pairs.
    for filename, digest in report["input_sha256"].items():
        path = Path(filename)
        if path.suffix.lower() != ".png":
            assert sha(path) == digest, filename
            INPUTS[str(path)] = digest
    pools = report["pools"]
    assert all(pool["n_unique_rgb_or_label_groups"] == pool["n_candidates"] for pool in pools)
    unknowns = {}
    for pool in pools:
        assert pool["selected_candidate_id"] == decision_by_id[pool["prompt_id"]]["selected_candidate_id"]
        scores = decision_by_id[pool["prompt_id"]]["scores"]
        independent_choice = max(pool["candidates"], key=lambda c: (scores[c["candidate_id"]], -c["seed"], c["candidate_id"]))
        assert independent_choice["candidate_id"] == pool["selected_candidate_id"]
        for candidate in pool["candidates"]:
            key = pool["canonical_scene_sha256"] + ":" + candidate["rgb_sha256"]
            candidate["label_group"] = key
            if candidate["label"] is None:
                unknowns.setdefault(key, []).append((pool, candidate))
    keys = sorted(unknowns)
    assert len(keys) <= 16, "Do not turn a bounded exact audit into a large enumeration"
    completions = []
    for bits in itertools.product((False, True), repeat=len(keys)):
        labels = dict(zip(keys, bits))
        metrics = {}
        for scope, subset in (("all_prompts", pools),
                              ("trained_prompts", [p for p in pools if p["entered_training"]])):
            per_pool = []
            for pool in subset:
                ys = {c["candidate_id"]: int(c["label"] if c["label"] is not None else labels[c["label_group"]]) for c in pool["candidates"]}
                selected = F(ys[pool["selected_candidate_id"]])
                uniform = F(sum(ys.values()), len(ys))
                scores = decision_by_id[pool["prompt_id"]]["scores"]
                top = [cid for cid, score in scores.items() if score == max(scores.values())]
                assert pool["selected_candidate_id"] in top
                tie_uniform = F(sum(ys[cid] for cid in top), len(top))
                per_pool.append((selected, uniform, selected - uniform, tie_uniform - uniform))
            metrics[scope] = [sum(x[k] for x in per_pool) / len(per_pool) for k in range(4)]
        completions.append((labels, metrics))
    result = {"source": str(source), "n_unknown_label_groups": len(keys),
              "exact_joint_completions": len(completions),
              "actual_policy": "max atomic score, then minimum sampling_seed, then maximum candidate_id; independently reproduced on all 12 pools",
              "uniform_top_tie_role": "Post-hoc counterfactual expectation; not the actual deterministic policy and not a proposal applied to training.",
              "scope": "Finite-cohort label completion, not a confidence interval or a new PASS gate."}
    for scope in ("all_prompts", "trained_prompts"):
        result[scope] = {"n_pools": report[scope]["n_prompts"]}
        for k, metric in enumerate(("selected_correctness", "uniform_correctness", "paired_gain", "uniform_top_tie_gain_diagnostic")):
            found = bounds([metrics[scope][k] for _, metrics in completions])
            if k < 3:
                assert found["low"]["exact"] == report[scope][metric]["lower_exact"]
                assert found["high"]["exact"] == report[scope][metric]["upper_exact"]
            result[scope][metric] = found
    priorities = []
    for key in keys:
        rows = unknowns[key]
        if not any(c["candidate_id"] == p["selected_candidate_id"] for p, c in rows):
            continue
        pool, candidate = next((p, c) for p, c in rows if c["candidate_id"] == p["selected_candidate_id"])
        item = {"prompt_id": pool["prompt_id"], "candidate_id": candidate["candidate_id"],
                "image_path": candidate["naive_image_path"], "gold_image_path": candidate["gold_image_path"],
                "rgb_sha256": candidate["rgb_sha256"], "unknown_reason": candidate["unknown_reason"],
                "label_source": candidate["label_source"], "conditional_gain_bounds": {}}
        for label in (False, True):
            item["conditional_gain_bounds"][str(label)] = {
                scope: bounds([metrics[scope][2] for labels, metrics in completions if labels[key] is label])
                for scope in ("all_prompts", "trained_prompts")}
        priorities.append(item)
    label_paths = {Path(item["label_source"]["path"]) for item in priorities}
    for path in label_paths:
        raw = path.read_bytes(); INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        manifest_path = path.with_name("manifest.jsonl")
        raw_manifest = manifest_path.read_bytes(); INPUTS[str(manifest_path)] = hashlib.sha256(raw_manifest).hexdigest()
        manifest = [json.loads(line) for line in raw_manifest.decode("utf-8").splitlines() if line.strip()]
        for item in priorities:
            if item["label_source"]["path"] != str(path):
                continue
            item["original_pending_verdict"] = next(row for row in records if row["image_path"] == item["gold_image_path"])
            item["original_specification"] = next(row for row in manifest if row["image_path"] == item["gold_image_path"])["spec"]
    result["selected_unknown_review_priorities"] = priorities
    return result


def fixed_bank():
    bank = read(RUN / "probe-bank/bank.json")
    audit = read(RUN / "audit-splits/scene_overlap.json")
    assert audit["provenance"]["bank_sha256"] == INPUTS[str(RUN / "probe-bank/bank.json")]
    pools = {pool["prompt_id"]: pool for pool in bank["pools"]}
    scenes = {spec: group["scene_sha256"] for group in audit["within_probe_bank_clusters"] for spec in group["spec_ids"]}
    excluded = {row["spec_id"] for row in audit["overlap"]["actual_probe_bank"]}
    snapshots = {}
    for arm, step in [("base", 0)] + [(arm, step) for arm in ("naive", "rfo_gold") for step in STEPS[1:]]:
        directory = RUN / f"gradient-probes/{arm}/step-{step:05d}"
        selection, meta = read(directory / "selection.json"), read(directory / "report.json")
        assert selection["fingerprint"] == meta["fingerprint"]
        assert meta["bank_fingerprint"] == bank["fingerprint"]
        assert meta["checkpoint"]["step"] == step and meta["checkpoint"]["arm"] == arm
        rows = {row["prompt_id"]: row for row in selection["rows"]}
        assert len(rows) == len(selection["rows"]) == len(pools) == 16
        assert set(rows) == set(pools) and all(not row["dropped"] for row in rows.values())
        snapshots[arm, step] = rows
    result = {"n_bank": len(pools), "n_images": sum(len(p["candidates"]) for p in pools.values()),
              "bank_fingerprint": bank["fingerprint"], "selection_rule": bank["selection_rule"],
              "checkpoint_provenance_scope": "Uses completed probe reports and prior main-thread integrity audits. This side review does not re-read weight/manifest files; the baseline checkpoint manifest was inaccessible under its read permissions.",
              "excluded_scene_overlap_specs": sorted(excluded), "cohorts": {}}
    for cohort, ids in (("original_16", sorted(pools)),
                       ("scene_disjoint_14", sorted(pid for pid, pool in pools.items() if pool["spec_id"] not in excluded))):
        assert len(ids) == (16 if cohort == "original_16" else 14)
        truth = {pid: {c["candidate_id"]: c["correct"] for c in pools[pid]["candidates"]} for pid in ids}
        assert all(type(y) is bool for ys in truth.values() for y in ys.values())
        probs = [F(sum(truth[pid].values()), len(truth[pid])) for pid in ids]
        groups = [scenes[pools[pid]["spec_id"]] for pid in ids]
        assert len(set(groups)) == len(groups)
        n = len(ids)
        cohort_result = {"n_pools": n, "uniform_random_expected_correctness": value(sum(probs) / n), "arms": {}}
        for arm in ("naive", "rfo_gold"):
            steps, chosen_by_step = [], {}
            base_chosen = [int(truth[pid][snapshots["base", 0][pid]["selected"]["naive"]]) for pid in ids]
            base_tie_gain = None
            for step in STEPS:
                rows = snapshots["base", 0] if step == 0 else snapshots[arm, step]
                per_pool = []
                for i, pid in enumerate(ids):
                    row = rows[pid]
                    scores = row["scores"]["naive"]
                    assert set(scores) == set(truth[pid])
                    selected = row["selected"]["naive"]
                    top = [cid for cid, score in scores.items() if score == max(scores.values())]
                    assert selected in top
                    y = int(truth[pid][selected])
                    per_pool.append({"prompt_id": pid, "spec_id": pools[pid]["spec_id"],
                                     "selected_candidate_id": selected, "selected_correct": y,
                                     "random_correctness": value(probs[i]), "gain": value(F(y) - probs[i]),
                                     "n_top_ties": len(top), "n_candidates": len(scores),
                                     "n_full_score": sum(score == 1 for score in scores.values()),
                                     "uniform_top_tie_correctness": value(F(sum(truth[pid][cid] for cid in top), len(top)))})
                ys = [row["selected_correct"] for row in per_pool]
                chosen_by_step[step] = ys
                gain = [F(y) - prob for y, prob in zip(ys, probs)]
                delta = [y - base for y, base in zip(ys, base_chosen)]
                tie_gain = [F(row["uniform_top_tie_correctness"]["exact"]) - prob for row, prob in zip(per_pool, probs)]
                if step == 0:
                    base_tie_gain = tie_gain
                steps.append({"step": step, "selected_correct": sum(ys), "selection_correctness": value(F(sum(ys), n)),
                              "gain_over_random": value(sum(gain) / n),
                              "gain_bootstrap": cluster_bootstrap(gain, groups),
                              "change_from_base": cluster_bootstrap(delta, groups),
                              "uniform_top_tie_gain_diagnostic": value(sum(tie_gain) / n),
                              "uniform_top_tie_gain_bootstrap": cluster_bootstrap(tie_gain, groups),
                              "uniform_top_tie_gain_change_from_base": cluster_bootstrap([cur - old for cur, old in zip(tie_gain, base_tie_gain)], groups),
                              "base_to_current_improved": sum(x > 0 for x in delta),
                              "base_to_current_worsened": sum(x < 0 for x in delta),
                              "uniform_policy_probability_at_least_observed": random_tail(probs, sum(ys)),
                              "n_pools_with_top_ties": sum(row["n_top_ties"] > 1 for row in per_pool),
                              "n_all_tied_pools": sum(row["n_top_ties"] == row["n_candidates"] for row in per_pool),
                              "per_pool": per_pool})
            cohort_result["arms"][arm] = steps
        cohort_result["arms_have_identical_selected_candidates_at_each_step"] = all(
            [(row["prompt_id"], row["selected_candidate_id"], row["selected_correct"])
             for row in cohort_result["arms"]["naive"][i]["per_pool"]]
            == [(row["prompt_id"], row["selected_candidate_id"], row["selected_correct"])
                for row in cohort_result["arms"]["rfo_gold"][i]["per_pool"]]
            for i in range(len(STEPS)))
        result["cohorts"][cohort] = cohort_result
    return result


def missing_later_truth():
    result = []
    for index in (1, 2):
        directory = RUN / f"rounds/round-{index:03d}"
        selection = read(directory / "selection.json")
        observations_path = directory / "observations/naive.jsonl"
        payload = observations_path.read_bytes()
        INPUTS[str(observations_path)] = hashlib.sha256(payload).hexdigest()
        observations = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        own_verified = directory / "ladder/naive/verified.jsonl"
        result.append({"round": index, "naive_candidates": len(observations),
                       "naive_direct_verified_exists": own_verified.exists(),
                       "paired_training_prompts": len(selection["paired_prompt_ids"]),
                       "selection_gain_identified": False,
                       "reason": "Only Gold-arm generated candidates have a verification file in this round; their labels are not transferred to distinct Naive images."})
        assert not own_verified.exists(), "New direct labels arrived: inspect before retaining this limitation"
    return result


def main():
    initial = initial_round()
    bank = fixed_bank()
    later = missing_later_truth()
    assert all(sha(Path(path)) == digest for path, digest in INPUTS.items()), "A source changed during this audit"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis": "retrospective CPU-only selection benefit audit",
              "initial_round_independent_arithmetic": initial, "fixed_bank_trajectory": bank,
              "later_actual_training_pool_label_gap": later,
              "input_sha256": INPUTS, "analysis_source_sha256": sha(Path(__file__)),
              "limitations": ["No change to registered labels, score policy, training, or stopping rules.",
                              "First-round completion bounds describe unknown labels, not population uncertainty.",
                              "The probe bank was selected for balanced known-correct/known-wrong pools and is not representative of newly generated training candidates.",
                              "Fixed-bank scores use the current checkpoint's Naive criterion; training arm Gold is not the frozen RFO observer.",
                              "Bootstrap is exploratory, conditional on selected scenes; no adjustment for repeated looks, bank selection, or training seed variability.",
                              "Uniform-top-tie calculations are diagnostics, not a replacement selection rule.",
                              "Review priority does not authorize fabricating or replacing any unknown gold label."]}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    queue = initial["selected_unknown_review_priorities"]
    (OUT / "priority-review.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"initial_round": {scope: initial[scope] for scope in ("all_prompts", "trained_prompts")},
                      "fixed_bank": {cohort: {"random": data["uniform_random_expected_correctness"],
                          "trajectory": [{k: row[k] for k in ("step", "selected_correct", "gain_over_random", "gain_bootstrap", "change_from_base", "base_to_current_improved", "base_to_current_worsened", "uniform_policy_probability_at_least_observed", "n_pools_with_top_ties", "n_all_tied_pools")} for row in data["arms"]["naive"]]}
                          for cohort, data in bank["cohorts"].items()}, "priority_review": queue}, indent=2))


if __name__ == "__main__":
    main()
