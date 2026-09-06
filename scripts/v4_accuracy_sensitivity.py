"""Conditional answer-replacement sensitivity on the same adjudicable pools.

Facts follow the actual frozen question text and complete verification records.
Unknown facts never become zeros. The historical 74.8% baseline and 1.2pp
pricing are not valid baselines for this diagnostic. Random answer correction
is not a stronger-observer prediction, sample-size guarantee, or new verdict.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfsight.v4.factual_truth import canonical_answer, factual_answer, load_verifications
from v4_resolution_verdict import REGISTERED, load_pools
from v4_selector_resolution import Candidate, read_jsonl, resolution

SEEDS = 20
GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
METRICS = [key for key, *_ in REGISTERED] + ["sep_lower", "unique_top", "mean_gap"]


def replacement_uniform(arm: str, seed: int, prompt_id: str, candidate_id: str,
                        question_id: str) -> float:
    """Stable nested masks, independent of p, dictionary order and exclusions."""
    key = json.dumps([arm, seed, prompt_id, candidate_id, question_id], separators=(",", ":"))
    integer = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 11
    return integer / 2**53


def truth_representation(question: dict[str, Any], answer: str) -> str | None:
    """Map factual counts back to old choices, preserving original p=0 scoring."""
    choices = question.get("choices")
    if choices:
        return next((choice for choice in choices if canonical_answer(question, choice) == answer), None)
    expected = question["expected_answer"]
    return expected if canonical_answer(question, expected) == answer else answer


def _spread(values: list[float]) -> dict[str, float | None]:
    return {"mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def sensitivity_report(pools: dict[str, dict[str, Any]], found: dict[str, dict[str, Any]],
                       observations: Sequence[dict[str, Any]], *, arm: str = "rfo",
                       grid: Sequence[float] = GRID, seeds: int = SEEDS) -> dict[str, Any]:
    if seeds < 1 or not grid or any(not math.isfinite(p) or not 0 <= p <= 1 for p in grid):
        raise ValueError("Require a positive seed count and replacement fractions in [0, 1]")
    expected_keys = {(pid, c["candidate_id"]) for pid, p in pools.items() for c in p["candidates"]}
    observed = {}
    for row in observations:
        key = (row["prompt_id"], row["candidate_id"])
        if key in observed:
            raise ValueError(f"Duplicate observation: {key}")
        if key not in expected_keys:
            raise ValueError(f"Observation outside frozen pools: {key}")
        observed[key] = row
    counts = collections.Counter()
    unknown, unavailable = collections.Counter(), collections.Counter()
    eligible, excluded, mismatches = {}, {}, []
    for pid in sorted(pools):
        pool = pools[pid]
        questions = pool["questions"]
        qids = [q["question_id"] for q in questions]
        if len(set(qids)) != len(qids):
            raise ValueError(f"Duplicate frozen question ID: {pid}")
        reasons = set()
        if not questions or not pool["candidates"]:
            reasons.add("empty_pool_or_questions")
        if pool["selection"].get("dropped"):
            reasons.add("selection_dropped")
        prepared = []
        for candidate in sorted(pool["candidates"], key=lambda row: row["candidate_id"]):
            counts["total_candidates"] += 1
            facts = [factual_answer(q, found.get(candidate["image_path"])) for q in questions]
            counts["total_questions"] += len(facts)
            counts["known_questions"] += sum(f.known for f in facts)
            unknown.update(f.reason for f in facts if not f.known)
            if not all(f.known for f in facts):
                counts["unknown_candidates"] += 1
                reasons.add("unknown_facts")
            replacements = [truth_representation(q, f.answer) if f.known else None
                            for q, f in zip(questions, facts)]
            unrepresentable = sum(f.known and replacement is None
                                 for f, replacement in zip(facts, replacements))
            counts["truth_not_in_choices_questions"] += unrepresentable
            if unrepresentable:
                reasons.add("truth_not_in_choices")
            record = observed.get((pid, candidate["candidate_id"]))
            observation = record.get("observation") if record is not None else None
            record_reason = ("missing_observation" if record is None else
                             "record_error" if record.get("error") is not None else
                             "missing_observation" if not isinstance(observation, dict) else
                             "observation_error" if observation.get("error") is not None else None)
            answers = (observation.get("answers") or []) if record_reason is None else []
            answer_ids = [answer["question_id"] for answer in answers]
            if len(set(answer_ids)) != len(answer_ids):
                raise ValueError(f"Duplicate observed question ID: {pid}/{candidate['candidate_id']}")
            if set(answer_ids) - set(qids):
                raise ValueError(f"Observed question outside frozen set: {pid}/{candidate['candidate_id']}")
            answer_of = {answer["question_id"]: answer for answer in answers}
            rows, candidate_available = [], True
            for q, fact, replacement in zip(questions, facts, replacements):
                answer = answer_of.get(q["question_id"])
                normalized = answer.get("normalized_answer") if answer is not None else None
                reason = (record_reason or ("missing_answer" if answer is None else
                          "answer_error" if answer.get("error") is not None else
                          "abstained_answer" if answer.get("abstain") else
                          "empty_answer" if normalized is None or not str(normalized).strip() else None))
                if reason:
                    unavailable[reason] += 1
                    candidate_available = False
                    reasons.add("unavailable_actual_answers")
                else:
                    counts["available_actual_answers"] += 1
                    if fact.known:
                        counts["known_facts_with_actual_answer"] += 1
                        counts["known_facts_actual_correct"] += canonical_answer(q, answer["normalized_answer"]) == fact.answer
                rows.append({"question": q, "got": answer.get("normalized_answer") if answer else None,
                             "fact": fact.answer, "replacement": replacement})
            if candidate_available and questions:
                # Complete non-abstaining observations reproduce the historical
                # available-answer denominator, rather than changing its policy.
                baseline = sum(row["got"] == row["question"]["expected_answer"] for row in rows) / len(rows)
                stored = pool["selection"].get("scores", {}).get(arm, {}).get(candidate["candidate_id"])
                if stored is None or not math.isfinite(float(stored)):
                    reasons.add("unavailable_stored_score")
                elif not math.isclose(baseline, float(stored), rel_tol=0, abs_tol=1e-12):
                    mismatches.append(f"{pid}/{candidate['candidate_id']}")
            prepared.append({"candidate_id": candidate["candidate_id"], "correct": bool(candidate["correct"]),
                             "answers": rows})
        if reasons:
            excluded[pid] = sorted(reasons)
        else:
            eligible[pid] = prepared
    if mismatches:
        raise ValueError(f"Reconstructed p=0 score differs from frozen selection: {mismatches[:5]}")
    counts["eligible_candidates"] = sum(len(pool) for pool in eligible.values())
    counts["eligible_questions"] = sum(len(c["answers"]) for pool in eligible.values() for c in pool)
    coverage = {key: counts[key] for key in (
        "total_candidates", "total_questions", "known_questions", "unknown_candidates",
        "available_actual_answers", "known_facts_with_actual_answer", "known_facts_actual_correct",
        "truth_not_in_choices_questions", "eligible_candidates", "eligible_questions")}
    coverage.update({"unknown_reasons": dict(unknown), "unavailable_actual_reasons": dict(unavailable),
                     "unknown_questions": counts["total_questions"] - counts["known_questions"],
                     "all_known_available_fact_accuracy": counts["known_facts_actual_correct"] / counts["known_facts_with_actual_answer"]
                     if counts["known_facts_with_actual_answer"] else None})
    curve = []
    for p in grid:
        replicates = []
        for seed in range(seeds if 0 < p < 1 else 1):
            scored, agree, replaced = [], 0, 0
            for pid, pool in eligible.items():
                cells = []
                for candidate in pool:
                    hits = 0
                    for row in candidate["answers"]:
                        q = row["question"]
                        replace = replacement_uniform(arm, seed, pid, candidate["candidate_id"], q["question_id"]) < p
                        answer = row["replacement"] if replace else row["got"]
                        replaced += replace
                        agree += canonical_answer(q, answer) == row["fact"]
                        hits += answer == q["expected_answer"]
                    cells.append(Candidate(candidate["candidate_id"], hits / len(candidate["answers"]), candidate["correct"]))
                scored.append(cells)
            if scored:
                metrics = resolution(scored)
                replicates.append({"seed": seed, "fact_correct": agree, "replaced_answers": replaced,
                                   "fact_accuracy": agree / counts["eligible_questions"],
                                   "metrics": {key: metrics[key] for key in METRICS}})
        curve.append({"replacement_fraction": p, "n_pools": len(eligible),
                      "n_candidates": counts["eligible_candidates"], "n_questions": counts["eligible_questions"],
                      "simulation_replicates": len(replicates),
                      "fact_accuracy": _spread([r["fact_accuracy"] for r in replicates]),
                      "fact_correct_answers": _spread([r["fact_correct"] for r in replicates]),
                      "metrics": {key: _spread([r["metrics"][key] for r in replicates]) for key in METRICS},
                      "replicates": replicates})
    return {"schema_version": 2, "arm": arm, "total_pools": len(pools), "eligible_pool_ids": list(eligible),
            "excluded_pool_reasons": excluded, "coverage": coverage, "curve": curve,
            "baseline_score_mismatches": len(mismatches),
            "eligibility": "all facts known and representable, all original answers available, matching frozen scores",
            "comparison_scope": "actual p=0 and every simulated p use exactly the same complete pool subset",
            "replacement_policy": "per-answer Bernoulli mask, stable SHA256 IDs and seed, nested across p",
            "score_policy": "historical equal-weight answer scoring, original representations; p=0 checked per candidate",
            "reference_thresholds": {key: {"direction": direction, "threshold": threshold}
                                     for key, _name, direction, threshold, *_ in REGISTERED},
            "limitations": ["conditional diagnostic; no new registered pass/fail decision",
                            "unknown exclusions may be informative; eligible pools do not represent all pools",
                            "seed min/max describes simulation randomness, not a sampling confidence interval",
                            "random factual replacement does not predict stronger-observer errors or gains",
                            "no training-effect, predictive-utility or sample-size guarantee",
                            "factual answers and image verdicts share the verification source",
                            "historical 74.8% baseline and 1.2pp pricing cannot be reused"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--arm", default="rfo", choices=["naive", "rfo"])
    parser.add_argument("--json-output", type=Path, help="Create a new diagnostic file; never overwrite an existing file")
    args = parser.parse_args(argv)
    if args.json_output:
        output = args.json_output.resolve()
        source_runs = json.loads((args.outdir / "runs.json").read_text(encoding="utf-8"))["runs"]
        protected = [args.outdir.resolve(), *(Path(name).resolve() for name in source_runs)]
        if any(output.is_relative_to(directory) for directory in protected):
            parser.error("--json-output must be outside the input and verification run directories")
        if output.exists():
            parser.error("--json-output must be a new file; existing reports are never overwritten")
    report = sensitivity_report(load_pools(args.outdir), load_verifications(args.outdir),
                                read_jsonl(args.outdir / f"observations.{args.arm}.jsonl"), arm=args.arm)
    coverage = report["coverage"]
    print(f"{args.outdir} [{args.arm}]: actual/simulated share {len(report['eligible_pool_ids'])}/{report['total_pools']} pools")
    print(f"Known facts {coverage['known_questions']}/{coverage['total_questions']}; unknown {coverage['unknown_reasons']}")
    print(f"Eligible answers {coverage['eligible_questions']}; unavailable answers {coverage['unavailable_actual_reasons']}")
    print("p       fact agreement   correct>wrong  equal     all-tied  ceiling")
    for row in report["curve"]:
        accuracy = row["fact_accuracy"]["mean"]
        if accuracy is None:
            print(f"{row['replacement_fraction']:5.1%}   unavailable: no eligible pools")
            continue
        values = [row["metrics"][key]["mean"] for key, *_ in REGISTERED]
        print(f"{row['replacement_fraction']:5.1%}   {accuracy:8.2%}       " + "  ".join(f"{value:7.2%}" for value in values))
    print("Conditional simulation only: no registered pass, stronger-observer prediction, or sample-size guarantee.")
    print("Historical 74.8% / +1.2pp pricing is not reused; simulation seed spread is not a confidence interval.")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Created {args.json_output}")


if __name__ == "__main__":
    main()
