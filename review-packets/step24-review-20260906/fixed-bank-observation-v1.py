"""Describe saved observers on fixed pixels/questions; never execute a model.

Run with envs/core/python.exe review-packets/step24-review-20260906/fixed-bank-observation.py.
The JSON output must be a new file outside all runs directories.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from selfsight.data.questions import normalize_answer
from selfsight.schemas import AtomicQuestion
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.v4.factual_truth import canonical_answer, factual_answer, question_scope


class Inputs:
    def __init__(self):
        self.hashes = {}

    def record(self, path):
        path = Path(path).resolve()
        digest = sha256_file(path)
        old = self.hashes.setdefault(str(path), digest)
        if old != digest:
            raise ValueError(f"Input changed during read: {path}")
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text(encoding="utf-8"))

    def jsonl(self, path):
        return [json.loads(line) for line in self.record(path).read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def verify(self):
        changed = [path for path, digest in self.hashes.items()
                   if sha256_file(path) != digest]
        if changed:
            raise ValueError(f"Inputs changed during analysis: {changed}")


def indexed(rows, key, name):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError(f"Duplicate {name}: {value}")
        result[value] = row
    return result


def rate(values):
    if not values:
        return {"value": None, "exact": None, "n_images": 0}
    value = sum(values, Fraction()) / len(values)
    return {"value": float(value), "exact": str(value), "n_images": len(values)}


def summarize(facts, answers):
    result = {}
    for family in ("count", "existence", "all"):
        subset = [f for f in facts if f["known"] and
                  (family == "all" or f["family"] == family)]
        images = collections.defaultdict(list)
        for fact in subset:
            images[(fact["prompt_id"], fact["candidate_id"])].append(fact)
        image_rates = {metric: [] for metric in (
            "response_request", "response_fact", "fact_request")}
        micro = collections.Counter()
        statuses = collections.Counter()
        for rows in images.values():
            hits = collections.Counter()
            for fact in rows:
                key = (fact["prompt_id"], fact["candidate_id"], fact["question_id"])
                response, status = answers[key]
                statuses[status] += 1
                # Invalid/missing responses count as unsuccessful agreement.
                hits["response_request"] += status == "valid" and response == fact["request"]
                hits["response_fact"] += status == "valid" and response == fact["fact"]
                hits["fact_request"] += fact["fact"] == fact["request"]
            micro.update(hits)
            for metric in image_rates:
                image_rates[metric].append(Fraction(hits[metric], len(rows)))
        result[family] = {
            "n_fact_known_questions": len(subset),
            "n_images_with_fact_known_questions": len(images),
            "n_pools_with_fact_known_questions": len({key[0] for key in images}),
            "response_status_on_fixed_known_set": dict(statuses),
            **{metric: rate(values) for metric, values in image_rates.items()},
            "question_weighted_micro_diagnostic": {
                metric: {"hits": micro[metric], "denominator": len(subset),
                         "value": micro[metric] / len(subset) if subset else None}
                for metric in image_rates},
        }
    return result


def build(run_dir):
    inputs = Inputs()
    for relative in (
        "src/selfsight/v4/factual_truth.py", "src/selfsight/v4/spec.py",
        "src/selfsight/data/questions.py", "src/selfsight/schemas.py",
        "src/selfsight/utils/hashing.py",
    ):
        inputs.record(ROOT / relative)
    inputs.record(__file__)
    bank = inputs.json(run_dir / "probe-bank/bank.json")
    source = Path(bank["source"]).resolve()
    for name, digest in bank["source_hashes"].items():
        if sha256_file(inputs.record(source / name)) != digest:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    original = indexed(inputs.jsonl(source / "pools.jsonl"),
                       lambda row: row["prompt_id"], "source pool")
    runs = inputs.json(source / "runs.json")["runs"]
    verifications = {}
    verification_sources = {}
    for name in runs:
        path = (ROOT / name).resolve() / "verified.jsonl"
        for row in inputs.jsonl(path):
            key = str(Path(row["image_path"]).resolve())
            if key in verifications:
                raise ValueError(f"Duplicate verification image path: {key}")
            verifications[key] = row
            verification_sources[key] = str(path)
    pools = indexed(bank["pools"], lambda row: row["prompt_id"], "bank pool")
    images, questions, facts = {}, {}, []
    for pid, pool in pools.items():
        old = original[pid]
        if pool["questions"] != old["questions"]:
            raise ValueError(f"Frozen question content differs from source: {pid}")
        old_candidates = indexed(old["candidates"],
                                 lambda row: row["candidate_id"], "source candidate")
        if {c["candidate_id"] for c in pool["candidates"]} != set(old_candidates):
            raise ValueError(f"Frozen candidates differ from source: {pid}")
        for question in pool["questions"]:
            questions[(pid, question["question_id"])] = question
        for candidate in pool["candidates"]:
            cid = candidate["candidate_id"]
            key = (pid, cid)
            if key in images:
                raise ValueError(f"Duplicate bank candidate: {key}")
            previous = old_candidates[cid]
            for field in ("image_path", "sampling_seed", "correct"):
                if candidate[field] != previous[field]:
                    raise ValueError(f"Candidate source mismatch: {key} {field}")
            path = inputs.record(candidate["image_path"])
            if rgb_sha256(path) != candidate["rgb_sha256"]:
                raise ValueError(f"Frozen RGB mismatch: {key}")
            verification = verifications.get(str(path))
            if verification is not None and verification["spec_id"] != pool["spec_id"]:
                raise ValueError(f"Verification spec mismatch: {key}")
            images[key] = candidate
            for question in pool["questions"]:
                scope = question_scope(question)
                if scope is None or scope.kind not in ("count", "existence"):
                    raise ValueError(f"Unsupported bank question: {question}")
                truth = factual_answer(question, verification)
                request = canonical_answer(question, question["expected_answer"])
                facts.append({
                    "prompt_id": pid, "candidate_id": cid,
                    "question_id": question["question_id"], "family": scope.kind,
                    "request": request, "fact": truth.answer, "known": truth.known,
                    "reason": truth.reason,
                    "verification_resolution": verification.get("resolution") if verification else None,
                    "verification_file": verification_sources.get(str(path)),
                    "verification_record_sha256": sha256_json(verification) if verification else None,
                })
    if (len(pools), len(images), len(facts)) != (16, 80, 404):
        raise ValueError("This retrospective report requires the frozen 16/80/404 bank")
    key_of = lambda f: (f["prompt_id"], f["candidate_id"], f["question_id"])
    if len({key_of(f) for f in facts}) != len(facts):
        raise ValueError("Duplicate image-question key")
    if len({str(Path(c["image_path"]).resolve()) for c in images.values()}) != 80:
        raise ValueError("Bank image paths are not unique")
    fixed_known = [key_of(f) for f in facts if f["known"]]
    checkpoints = []
    checkpoint_specs = [("base", 0)] + [(arm, step)
        for arm in ("naive", "rfo_gold") for step in (8, 16, 24)]
    for arm, step in checkpoint_specs:
        path = run_dir / "gradient-probes" / arm / f"step-{step:05d}"
        report = inputs.json(path / "report.json")
        meta = inputs.json(path / "run.meta.json")
        sentinel = inputs.json(run_dir / "stage-completion" /
                               f"{arm}.step-{step:05d}.gradient.json")
        if sentinel["stage"] != f"{arm}.step-{step:05d}.gradient":
            raise ValueError("Invalid successful stage sentinel")
        if (report["checkpoint"]["arm"], report["checkpoint"]["step"]) != (arm, step):
            raise ValueError("Wrong checkpoint identity")
        if (report["bank_fingerprint"] != bank["fingerprint"] or
                meta["bank_fingerprint"] != bank["fingerprint"] or
                meta["checkpoint"] != report["checkpoint"] or
                report["prompt_ids"] != list(pools)):
            raise ValueError("Checkpoint bank or metadata mismatch")
        normalizer_hash = meta["implementation_sha256"]["src/selfsight/data/questions.py"]
        if normalizer_hash != inputs.hashes[str((ROOT / "src/selfsight/data/questions.py").resolve())]:
            raise ValueError("Saved answer normalizer differs from current implementation")
        observed = indexed(inputs.jsonl(path / "observations.naive.jsonl"),
                           lambda row: (row["prompt_id"], row["candidate_id"]),
                           "saved image observation")
        if set(observed) - set(images):
            raise ValueError("Unexpected image observation outside the frozen bank")
        answers = {}
        all_status = collections.Counter()
        for image_key, candidate in images.items():
            pid, cid = image_key
            row = observed.get(image_key)
            actual = {}
            if row is not None:
                observation = row["observation"]
                if (observation["observer_id"], observation["observer_revision"]) != (
                        report["model_id"], report["model_revision"]):
                    raise ValueError("Saved answer observer identity mismatch")
                if observation["rgb_sha256"] != candidate["rgb_sha256"]:
                    raise ValueError("Saved answer RGB mismatch")
                actual = indexed(observation["answers"], lambda a: a["question_id"],
                                 "saved atomic answer")
                expected_order = [q["question_id"] for q in pools[pid]["questions"]]
                if set(actual) - set(expected_order):
                    raise ValueError("Unexpected question outside the frozen bank")
                if list(actual) != [qid for qid in expected_order if qid in actual]:
                    raise ValueError("Saved question order differs from the frozen bank")
            for question in pools[pid]["questions"]:
                answer = actual.get(question["question_id"])
                response = None
                if row is None:
                    status = "missing_image_observation"
                elif answer is None:
                    status = "missing_atomic_answer"
                elif answer.get("error"):
                    status = "error"
                elif answer.get("abstain"):
                    status = "abstain"
                elif answer.get("normalized_answer") is None:
                    status = "missing_normalized_answer"
                elif normalize_answer(answer["raw_answer"], AtomicQuestion.from_dict(question)) != answer["normalized_answer"]:
                    status = "normalization_mismatch"
                else:
                    status = "valid"
                    response = canonical_answer(question, answer["normalized_answer"])
                answers[(pid, cid, question["question_id"])] = (response, status)
                all_status[status] += 1
        checkpoints.append({
            "arm": arm, "step": step, "checkpoint": report["checkpoint"],
            "adapter_parameter_digest": report["adapter_parameter_digest"],
            "observation_file_sha256": inputs.hashes[str((path / "observations.naive.jsonl").resolve())],
            "fixed_known_set_sha256": sha256_json(fixed_known),
            "coverage": {"expected_images": 80, "observed_images": len(observed),
                         "expected_atomic_answers": 404,
                         "observed_atomic_answers": sum(len(x["observation"]["answers"]) for x in observed.values()),
                         "all_fixed_questions_status": dict(all_status)},
            "metrics": summarize(facts, answers),
        })
    for family in ("count", "existence", "all"):
        constants = {cp["metrics"][family]["fact_request"]["exact"] for cp in checkpoints}
        if len(constants) != 1:
            raise AssertionError("Frozen external fact=request unexpectedly changed")
    coverage = {}
    for family in ("count", "existence", "all"):
        subset = [f for f in facts if family == "all" or f["family"] == family]
        coverage[family] = {
            "total_questions": len(subset), "known": sum(f["known"] for f in subset),
            "unknown": sum(not f["known"] for f in subset),
            "unknown_reasons": dict(collections.Counter(f["reason"] for f in subset if not f["known"])),
        }
    inputs.verify()
    return {
        "schema_version": "fixed-bank-observation-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_only",
        "source_run": str(run_dir), "original_verification_source": str(source),
        "bank_fingerprint": bank["fingerprint"],
        "fixed_known_set_sha256": sha256_json(fixed_known),
        "weighting": "For each family, average agreements over its fact-known questions within each image; then equally average eligible images. All primary metrics use exactly the same fixed fact-known question set at every checkpoint. Images with zero known questions in a family are excluded once for that family.",
        "invalid_response_policy": "Missing image/answer, error, abstention, absent normalization and normalization mismatch contribute zero agreement while staying in the fixed known-question/image denominators.",
        "baseline_policy": "One saved base step0 shared by both training-arm trajectories; observation condition is the saved prompted Naive observer for every checkpoint.",
        "fact_coverage": coverage, "fixed_facts": facts,
        "checkpoints": checkpoints, "external_fact_request_constant": True,
        "input_sha256": inputs.hashes, "inputs_unchanged_after_analysis": True,
        "limitations": [
            "No new model answers, gradients or images; no confidence intervals, hypothesis tests, D_g or lead-time claims.",
            "Fixed-bank descriptive changes do not establish general observer ability or a causal training mechanism.",
            "Verifier-derived facts are partial and share provenance with image adjudication; unresolved facts are not absence.",
            "Image-equal primary means differ from pooled-question micro diagnostics; neither weights pools equally.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/v4/decoupling-pilot-20260906")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output exists; refuse to overwrite")
    if output.is_relative_to(ROOT / "runs") or output.is_relative_to(run_dir):
        parser.error("Output must be outside all run directories")
    report = build(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Created {output}; fixed pixels/questions; descriptive only")
    for cp in report["checkpoints"]:
        print(cp["arm"], cp["step"], {
            family: {key: round(cp["metrics"][family][key]["value"] * 100, 6)
                     for key in ("response_request", "response_fact", "fact_request")}
            for family in ("count", "existence", "all")})


if __name__ == "__main__":
    main()

