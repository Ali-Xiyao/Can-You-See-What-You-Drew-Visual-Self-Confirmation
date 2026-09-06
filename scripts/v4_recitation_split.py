"""Describe count answers by whether the observed picture matches the request.

Facts come from actual frozen question text and complete verifier records.
Unresolved scopes and unavailable answers retain separate coverage counts.
These observational strata do not identify prompt leakage or determine whether
a different observer, resolution or sampling strategy would improve accuracy.

    envs/core/python.exe scripts/v4_recitation_split.py runs/v4/gate-b-openct2

The CLI prints JSON, or writes it to a new --output path outside its input runs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfsight.v4.factual_truth import (
    canonical_answer,
    factual_answer,
    load_verifications,
    question_scope,
)

CELL_NAMES = ("picture_matches_request", "picture_differs_from_request", "fact_unknown")
LIMITATIONS = [
    "Descriptive observational strata; not a causal estimate of prompt leakage.",
    "Within-arm strata may differ in image difficulty, requested counts, colours and answer priors.",
    "Between-arm differences may also change the model and prompt condition together.",
    "Requested-answer concentration can reflect priors, limited choices or systematic counting errors.",
    "This report cannot determine whether stronger observers, repeated sampling or higher resolution would help.",
    "Facts are detector/crop-derived operational labels, not independent human truth.",
    "Unknown facts never enter factual-accuracy denominators; unavailable answers have separate denominators.",
    "Legacy choices may make the factual answer impossible to express; this is reported separately.",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _is_count(question: Mapping[str, Any]) -> bool:
    scope = question_scope(question)
    return (question.get("family") == "count" or (scope is not None and scope.kind == "count")
            or ":count:" in str(question.get("question_id", "")))


def _answer_status(row: Mapping[str, Any] | None,
                   answer: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing_observation"
    observation = row.get("observation")
    if row.get("error") is not None or (
            isinstance(observation, Mapping) and observation.get("error") is not None):
        return "observation_error"
    if not isinstance(observation, Mapping):
        return "missing_observation"
    if answer is None:
        return "missing_answer"
    if answer.get("error") is not None:
        return "answer_error"
    if answer.get("abstain"):
        return "abstain"
    if answer.get("normalized_answer") is None:
        return "null_answer"
    if not str(answer["normalized_answer"]).strip():
        return "empty_answer"
    return "available"


def _new_cell() -> dict[str, Any]:
    return {"total": 0, "answer_status": collections.Counter(),
            "said_request": 0, "said_truth": 0, "truth_not_in_choices": 0,
            "wrong_answer_minus_truth": collections.Counter()}


def _finish_cell(cell: dict[str, Any], *, facts_known: bool) -> dict[str, Any]:
    available = cell["answer_status"]["available"]
    return {**cell, "answer_status": dict(cell["answer_status"]),
            "wrong_answer_minus_truth": dict(sorted(cell["wrong_answer_minus_truth"].items())),
            "available": available, "answer_coverage": _ratio(available, cell["total"]),
            "said_request_rate_available": _ratio(cell["said_request"], available),
            "said_request_rate_all_requested": _ratio(cell["said_request"], cell["total"]),
            "said_truth_rate_available": _ratio(cell["said_truth"], available) if facts_known else None,
            "said_truth_rate_all_requested": _ratio(cell["said_truth"], cell["total"]) if facts_known else None}


def recitation_report(
    pools: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    found: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Pure count-level report over every expected candidate/question.

    Iterate frozen pools, not observed answers: missing records cannot silently
    shrink the denominator. Per-question records retain factual and answer
    availability, so aggregate cells can be rebuilt.
    """
    pool_rows = list(pools.values()) if isinstance(pools, Mapping) else list(pools)
    candidates: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    question_ids: dict[str, set[str]] = {}
    for pool in pool_rows:
        prompt_id = str(pool["prompt_id"])
        if prompt_id in question_ids:
            raise ValueError(f"Duplicate pool prompt_id: {prompt_id}")
        ids = [str(q["question_id"]) for q in pool["questions"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate question in pool: {prompt_id}")
        question_ids[prompt_id] = set(ids)
        for candidate in pool["candidates"]:
            key = (prompt_id, str(candidate["candidate_id"]))
            if key in candidates:
                raise ValueError(f"Duplicate candidate: {key}")
            candidates[key] = (pool, candidate)

    arms = {}
    for arm, rows in observations.items():
        observed: dict[tuple[str, str], Mapping[str, Any]] = {}
        answer_tables: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
        observer_ids: set[tuple[str, str]] = set()
        for row in rows:
            key = (str(row["prompt_id"]), str(row["candidate_id"]))
            if key not in candidates:
                raise ValueError(f"Unexpected observed candidate in {arm}: {key}")
            if key in observed:
                raise ValueError(f"Duplicate observed candidate in {arm}: {key}")
            observed[key] = row
            observation = row.get("observation")
            observation = observation if isinstance(observation, Mapping) else {}
            if observation.get("observer_id") is not None:
                observer_ids.add((str(observation["observer_id"]),
                                  str(observation.get("observer_revision", ""))))
            answers = {}
            for answer in observation.get("answers") or []:
                qid = str(answer["question_id"])
                if qid not in question_ids[key[0]]:
                    raise ValueError(f"Unexpected answer in {arm}: {key}, {qid}")
                if qid in answers:
                    raise ValueError(f"Duplicate answer in {arm}: {key}, {qid}")
                answers[qid] = answer
            answer_tables[key] = answers

        cells = {name: _new_cell() for name in CELL_NAMES}
        unknown: collections.Counter[str] = collections.Counter()
        status_counts: collections.Counter[str] = collections.Counter()
        records = []
        for key, (pool, candidate) in candidates.items():
            image_path = str(candidate["image_path"])
            for question in pool["questions"]:
                if not _is_count(question):
                    continue
                fact = factual_answer(question, found.get(image_path))
                wanted = canonical_answer(question, question["expected_answer"])
                truth = canonical_answer(question, fact.answer) if fact.known else None
                answer = answer_tables.get(key, {}).get(str(question["question_id"]))
                status = _answer_status(observed.get(key), answer)
                said = (canonical_answer(question, answer["normalized_answer"])
                        if status == "available" else None)
                name = ("fact_unknown" if not fact.known else "picture_matches_request"
                        if truth == wanted else "picture_differs_from_request")
                cell = cells[name]
                cell["total"] += 1
                cell["answer_status"][status] += 1
                status_counts[status] += 1
                if not fact.known:
                    unknown[fact.reason] += 1
                choices = question.get("choices") or []
                representable = (not choices or truth in {
                    canonical_answer(question, choice) for choice in choices}) if fact.known else None
                cell["truth_not_in_choices"] += representable is False
                if status == "available":
                    cell["said_request"] += said == wanted
                    cell["said_truth"] += fact.known and said == truth
                    if name == "picture_differs_from_request" and said != truth:
                        try:
                            difference = str(int(said) - int(truth))
                        except (TypeError, ValueError):
                            difference = "non_numeric"
                        cell["wrong_answer_minus_truth"][difference] += 1
                records.append({"prompt_id": key[0], "candidate_id": key[1],
                                "image_path": image_path, "question_id": question["question_id"],
                                "question_text": question["text"], "requested_answer": wanted,
                                "fact_known": fact.known, "fact_answer": truth,
                                "fact_reason": fact.reason, "cell": name,
                                "answer_status": status, "said": said,
                                "truth_representable_in_choices": representable})

        known = sum(cells[name]["total"] for name in CELL_NAMES[:2])
        known_available = sum(cells[name]["answer_status"]["available"] for name in CELL_NAMES[:2])
        correct = sum(cells[name]["said_truth"] for name in CELL_NAMES[:2])
        total = len(records)
        arms[arm] = {
            "total_questions": total, "known_questions": known,
            "unknown_questions": total - known, "unknown_reasons": dict(unknown),
            "fact_coverage": _ratio(known, total), "answer_status": dict(status_counts),
            "known_available_answers": known_available,
            "factual_accuracy_available": _ratio(correct, known_available),
            "factual_correct_over_all_known_requests": _ratio(correct, known),
            "observer_identities": [{"id": name, "revision": revision}
                                    for name, revision in sorted(observer_ids)],
            "cells": {name: _finish_cell(cell, facts_known=name != "fact_unknown")
                      for name, cell in cells.items()}, "records": records,
        }
    return {"schema_version": 2, "scope": "descriptive count-answer stratification",
            "n_pools": len(pool_rows), "n_candidates": len(candidates),
            "cell_definitions": {
                "picture_matches_request": "Known answer to this count question equals its requested answer; not whole-image correctness.",
                "picture_differs_from_request": "Known answer to this count question differs from its requested answer.",
                "fact_unknown": "This question's factual answer is unresolved and excluded from both known-fact strata.",
            },
            "arms": arms, "limitations": list(LIMITATIONS)}


def _source_runs(out_dir: Path) -> list[Path]:
    return [Path(name).resolve() for name in
            json.loads((out_dir / "runs.json").read_text(encoding="utf-8"))["runs"]]


def build_report(out_dir: str | Path, arms: Sequence[str] = ("naive", "rfo")) -> dict[str, Any]:
    """Read an existing pool run without changing any historical output."""
    out_dir = Path(out_dir).resolve()
    paths = [out_dir / "pools.jsonl", out_dir / "runs.json"]
    paths += [source / "verified.jsonl" for source in _source_runs(out_dir)]
    paths += [out_dir / f"observations.{arm}.jsonl" for arm in arms]

    def identities() -> dict[str, str | None]:
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                if path.exists() else None for path in paths}

    before = identities()
    observations = {arm: _read_jsonl(out_dir / f"observations.{arm}.jsonl")
                    if (out_dir / f"observations.{arm}.jsonl").exists() else [] for arm in arms}
    report = recitation_report(_read_jsonl(out_dir / "pools.jsonl"),
                               load_verifications(out_dir), observations)
    if identities() != before:
        raise ValueError("Input files changed while building the diagnostic")
    report["outdir"] = str(out_dir)
    report["input_sha256"] = before
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--output", type=Path, help="new JSON file outside the input runs")
    args = parser.parse_args(argv)
    report = build_report(args.outdir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    output = args.output.resolve()
    protected = [args.outdir.resolve(), *_source_runs(args.outdir)]
    if any(output.is_relative_to(directory) for directory in protected):
        parser.error("--output must be outside the input run directories")
    if output.exists():
        parser.error("--output must be a new file; existing reports are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
