"""Canonical-scene overlap between planned training exposure and a frozen probe bank.

Writes the scene audit that `v4_gradient_sensitivity.py --audit` consumes: which
probe-bank prompts describe a scene the model will also have been trained on, so
that the supplementary sensitivity analysis can drop them.

WHY THIS IS NOT WRITTEN TO `RUN/audit-splits/scene_overlap.json`
`v4_decoupling_report.py` picks that path up implicitly and uses it to choose an
outcome subset, which is only sound if the audit was fixed before anyone could
see an outcome. It enforces that by refusing any audit whose
`created_before_any_outcome_evaluation_artifact` is not true. An audit written
after a run has started -- which is the case this script exists for -- must
declare `false`, so it belongs on an explicit path passed to the one consumer
whose input it can legitimately be. Two consumers, two different standards of
evidence; the flag is the boundary and this script never writes `true`.

For the same reason this file emits no outcome-side scene clusters and no
outcome spec_id subset. The exclusion set for an outcome curve chosen after
seeing outcomes is not a thing that should exist in a readable form, so the
audit is built structurally unable to supply one even if the flag were defeated.

The probe-bank exclusion has no such hazard: it is a function of the frozen
split, the frozen bank and the frozen schedule alone, all three of which predate
any gradient measurement, and it is applied to a gradient instrument rather than
to the outcome the paper reports.
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json
from selfsight.v4.spec import SceneSpec
from selfsight.v4.train import build_schedule, load_training_corpus, restrict_replay

VERSION = "v4-scene-overlap-audit-1"

CANONICAL_SCENE_KEY_DEFINITION = {
    "algorithm": (
        "Aggregate counts by (canonical_noun(object), color); sort noun/color; serialize "
        "[[noun,color_or_null,total_count],...] using selfsight.utils.hashing.sha256_json."
    ),
    "scope": (
        "Equivalence under the object/color/count correctness instrument. Prompt order, "
        "quantifier spelling, noun synonyms, and unscored support-surface/context wording are "
        "ignored; identical key does not claim pixel-identical/full-scene semantics."
    ),
}


def canonical_scene(spec: SceneSpec) -> list[list[Any]]:
    """The scene key: what the correctness instrument can tell apart, and nothing else.

    SpecObject.from_dict has already folded synonyms and compounds into one noun,
    so two specs that differ only in wording aggregate to the same counter here.
    """
    counter: collections.Counter[tuple[str, str | None]] = collections.Counter()
    for obj in spec.objects:
        counter[(obj.object, obj.color)] += obj.count
    return [[noun, color, total] for (noun, color), total
            in sorted(counter.items(), key=lambda item: (item[0][0], item[0][1] or ""))]


def cluster(specs: dict[str, SceneSpec], spec_ids: list[str]) -> list[dict[str, Any]]:
    """Group spec_ids by scene key, ordered by the key so the file is stable."""
    groups: dict[str, dict[str, Any]] = {}
    for spec_id in spec_ids:
        scene = canonical_scene(specs[spec_id])
        digest = sha256_json(scene)
        group = groups.setdefault(digest, {"scene_sha256": digest, "canonical_scene": scene,
                                           "spec_ids": []})
        group["spec_ids"].append(spec_id)
    for group in groups.values():
        group["spec_ids"].sort()
    return [groups[digest] for digest in sorted(groups)]


def planned_exposure(config: dict[str, Any], split: dict[str, Any]) -> dict[str, int]:
    """Which training prompts the frozen schedule reaches, and in which round.

    build_schedule reshuffles the whole train partition and takes
    rounds*prompts_per_round slots from it, so when those two numbers multiply
    out to the size of the partition every prompt is reached exactly once and
    membership stops depending on the seed. That is the case this run is in --
    11 x 12 against 132 -- but the caller checks rather than assumes it, because
    a shorter schedule would leave prompts unexposed and the audit would then
    silently over-exclude.
    """
    training = config["training"]
    schedule = build_schedule(
        split["train"],
        rounds=int(training["rounds"]),
        prompts_per_round=int(training["prompts_per_round"]),
        candidate_k=int(training["candidate_k"]),
        seed=int(config["seed"]),
        max_epochs=1,
    )
    # No overwrite guard: at max_epochs=1 build_schedule refuses a schedule
    # longer than the prompt list, so a prompt cannot appear twice and the guard
    # would be a branch no input can reach.
    return {entry.prompt_id: int(entry.round_index) for entry in schedule}


def verify_executed_rounds(run: Path, scheduled: dict[str, int]) -> list[int]:
    """Check the reconstruction against the rounds the run has already executed.

    The schedule is recomputed here rather than read off disk, so it is worth
    proving the recomputation lands on what actually happened wherever the run
    gives us something to compare against.
    """
    verified = []
    for selection_path in sorted(run.glob("rounds/round-*/selection.json")):
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        index = int(selection["round"])
        executed = {decision["prompt_id"]
                    for arm in selection["decisions"].values() for decision in arm}
        predicted = {prompt for prompt, where in scheduled.items() if where == index}
        if executed != predicted:
            raise ValueError(
                f"Reconstructed round {index} is not the round the run executed: "
                f"only-predicted={sorted(predicted - executed)} "
                f"only-executed={sorted(executed - predicted)}")
        verified.append(index)
    return verified


def build_audit(run: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = run / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    bank_path = run / "probe-bank" / "bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in bank.items() if key != "fingerprint"}
    if sha256_json(content) != bank["fingerprint"]:
        raise ValueError("Bank content fingerprint mismatch; refusing to audit it")

    corpus = load_training_corpus(split["runs"])
    specs = corpus.specs
    replay = restrict_replay(corpus, split["train"]).replay
    replay_prompts = {example.prompt_id for example in replay}

    scheduled = planned_exposure(config, split)
    verified_rounds = verify_executed_rounds(run, scheduled)
    if set(scheduled) != set(split["train"]):
        raise ValueError("Schedule does not reach the whole train partition; the exposure set "
                         "would then depend on the seed and needs recording per prompt")

    exposed_by_scene: dict[str, list[str]] = collections.defaultdict(list)
    for prompt_id in sorted(scheduled):
        exposed_by_scene[sha256_json(canonical_scene(specs[prompt_id]))].append(prompt_id)

    bank_ids = [pool["spec_id"] for pool in bank["pools"]]
    if len(set(bank_ids)) != len(bank_ids):
        raise ValueError("Bank holds more than one pool per spec")
    clusters = cluster(specs, bank_ids)

    rows = []
    for spec_id in sorted(bank_ids):
        scene = canonical_scene(specs[spec_id])
        digest = sha256_json(scene)
        matches = exposed_by_scene.get(digest, [])
        if not matches:
            continue
        rows.append({
            "spec_id": spec_id,
            "prompt": specs[spec_id].prompt,
            "canonical_scene": scene,
            "scene_sha256": digest,
            "training_matches": [{
                "spec_id": match,
                "prompt": specs[match].prompt,
                "scheduled_round": scheduled[match],
                "in_train_only_replay_pool": match in replay_prompts,
            } for match in matches],
        })

    def exposed(spec_ids: list[str]) -> int:
        return sum(1 for spec_id in spec_ids
                   if sha256_json(canonical_scene(specs[spec_id])) in exposed_by_scene)

    return {
        "version": VERSION,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_before_any_outcome_evaluation_artifact": False,
        "retrospective_notice": (
            "Built after the run had produced evaluation artifacts. Admissible for the "
            "gradient instrument, whose exclusion set is a function of the frozen split, bank "
            "and schedule only. Not admissible for choosing an outcome subset, which is why "
            "this file is not at RUN/audit-splits/scene_overlap.json and carries no outcome "
            "clusters. v4_decoupling_report.py refuses it on the flag above; do not flip the "
            "flag, rerun this against a run that has not started."),
        "consumer": "scripts/v4_gradient_sensitivity.py --audit",
        "canonical_scene_key_definition": CANONICAL_SCENE_KEY_DEFINITION,
        "exposure_definition": (
            "Every prompt in the frozen train partition. rounds x prompts_per_round equals the "
            "partition size, so the schedule reaches all of them exactly once and membership "
            "does not depend on the seed; only the round assignment does. Replay draws from a "
            "pool restricted to the same partition, so it adds no scene the generation "
            "schedule has not already exposed."),
        "provenance": {
            "run": str(run),
            "bank_sha256": sha256_file(bank_path),
            "bank_fingerprint": bank["fingerprint"],
            "split_sha256": sha256_file(split_path),
            "split_digest": split["digest"],
            "config_sha256": sha256_file(config_path),
            "config_path": str(config_path),
            "corpus_runs": list(split["runs"]),
            "schedule": {
                "rounds": int(config["training"]["rounds"]),
                "prompts_per_round": int(config["training"]["prompts_per_round"]),
                "seed": int(config["seed"]),
                "max_epochs": 1,
                "rounds_checked_against_executed_selection": verified_rounds,
            },
        },
        "within_probe_bank_clusters": clusters,
        "overlap": {"actual_probe_bank": rows},
        "summary": {
            "train_partition_n": len(split["train"]),
            "scheduled_train_prompt_n": len(scheduled),
            "train_only_replay_pool_prompt_n": len(replay_prompts),
            "train_only_replay_pool_examples": len(replay),
            "exposed_scene_n": len(exposed_by_scene),
            "probe_bank_n": len(bank_ids),
            "probe_bank_scene_n": len(clusters),
            "actual_probe_bank_overlap_n": len(rows),
            "probe_partition_overlap_n": exposed(split["probe"]),
            "outcome_overlap_n": exposed(split["outcome"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="Existing run directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write it; must not be under RUN/audit-splits/")
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must already exist")
    if "audit-splits" in args.output.resolve().parts:
        parser.error("This audit is retrospective and audit-splits/ is read by the report; "
                     "see the module docstring")
    audit = build_audit(args.outdir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, audit)
    summary = audit["summary"]
    print(f"{VERSION}: wrote {args.output}")
    print(f"  probe bank {summary['probe_bank_n']} prompts in "
          f"{summary['probe_bank_scene_n']} scenes; excluding "
          f"{summary['actual_probe_bank_overlap_n']} for training exposure")


if __name__ == "__main__":
    main()
