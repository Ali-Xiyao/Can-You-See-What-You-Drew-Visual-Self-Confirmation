"""Canonical-scene overlap between planned training exposure and a frozen probe bank.

Writes the scene audit that `v4_gradient_sensitivity.py --audit` consumes: which
probe-bank prompts describe a scene the model will also have been trained on, so
that the supplementary sensitivity analysis can drop them.

TWO MODES, BECAUSE THERE ARE TWO CONSUMERS AND TWO STANDARDS OF EVIDENCE
`v4_gradient_sensitivity.py --audit` wants a probe-bank exclusion set, which is
a function of the frozen split, the frozen bank and the frozen schedule alone.
All three predate any gradient measurement, so this can be computed at any time,
including in the middle of a run.

`v4_decoupling_report.py` reads `RUN/audit-splits/scene_overlap.json` implicitly
and uses it to choose a subset of the *outcome* population. That is a
sensitivity analysis if the subset was fixed before any outcome existed and a
choice if it was not, so the report refuses any audit whose
`created_before_any_outcome_evaluation_artifact` is not true.

Default mode serves the first consumer. It declares the flag `false`, refuses to
write anywhere under `audit-splits/`, and emits no outcome-side clusters and no
outcome spec_id subset at all -- structurally unable to supply one even if the
flag were defeated, because an outcome exclusion set chosen after seeing
outcomes should not exist in a readable form.

`--prospective` serves both. It is only allowed on a run directory that holds
nothing an outcome could have come from -- see `outcome_evaluation_artifacts`,
which is an allowlist -- and it is that check, not the flag, that makes the
claim true. Run it after `split` and `freeze-probe` and before the first round.
`runs/v4/decoupling-main-20260908` could not have one: it was already running.
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
PROSPECTIVE_KIND = "prospective_scene_overlap_sensitivity_freeze"
REPRESENTATIVE_KIND = "prospective_independent_scene_representative_sensitivity"
REPRESENTATIVE_RULE = (
    "For every frozen canonical scene, select its lexicographically first outcome spec; "
    "exclude the entire scene if any member lies outside the previously frozen "
    "scene-disjoint sensitivity. No outcome values are read.")
REPRESENTATIVE_ENCODING = ("UTF-8 JSON list, ensure_ascii=False, "
                           "compact comma/colon separators")

# Carried forward verbatim from the pilot's published audit so a reader comparing
# the two files sees the same instructions, not a reworded version of them. They
# are advice to whoever reads the report, not anything this script enforces.
RECOMMENDATIONS = [
    "Keep the prompt-level main analysis unchanged.",
    ("Add the sensitivity prespecified by this file on "
     "scene_disjoint_outcome_sensitivity.spec_ids."),
    ("Add a canonical-scene cluster bootstrap: resample whole scene clusters, keeping "
     "all prompt variants and all checkpoints/arms together; report the original "
     "prompt-weighted estimator plus the cluster count, and do not silently replace "
     "its population with scene-equal weighting."),
    ("The frozen probe bank has its own overlap and cluster inventory; report the "
     "sensitivity that excludes exposed scenes if the sample size permits, without "
     "replacing the bank."),
]

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


def overlap_rows(spec_ids: list[str], specs: dict[str, SceneSpec],
                 exposed_by_scene: dict[str, list[str]], scheduled: dict[str, int],
                 replay_prompts: set[str]) -> list[dict[str, Any]]:
    """Every spec in `spec_ids` whose scene key a training prompt also carries.

    `scheduled_rounds` is a list because a prompt can be in the exposure set
    without being on the generation schedule -- that is what the conservative
    section is for -- and an empty list says exactly that, where a null or a
    missing key would read as "not recorded".
    """

    rows = []
    for spec_id in sorted(spec_ids):
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
                "canonical_scene": canonical_scene(specs[match]),
                "scene_sha256": sha256_json(canonical_scene(specs[match])),
                "scheduled_rounds": [scheduled[match]] if match in scheduled else [],
                "in_train_only_replay_pool": match in replay_prompts,
            } for match in matches],
        })
    return rows


def outcome_evaluation_artifacts(run: Path) -> list[str]:
    """What in this run directory could already carry an outcome measurement.

    An allowlist, not a denylist. The claim the prospective flag makes is that
    nothing here could have been seen before the audit was fixed, and a
    denylist makes that claim about the artifacts someone thought of. Only the
    two stages that run before any arm draws anything are permitted, and every
    other name is reported rather than judged.
    """

    allowed = {"split.json", "probe-bank", "audit-splits", "run_manifest.json",
               "state.json", "supervisor.lock", "supervisor.log", "supervisor.err",
               "logs", "stage-completion"}
    found = [entry.name for entry in sorted(run.iterdir()) if entry.name not in allowed]
    stages = run / "stage-completion"
    if stages.is_dir():
        found += [f"stage-completion/{marker.name}" for marker in sorted(stages.glob("*.json"))
                  if marker.stem not in {"split", "freeze-probe"}]
    return found


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


def build_audit(run: Path, config_path: Path, *, prospective: bool = False) -> dict[str, Any]:
    if prospective:
        existing = outcome_evaluation_artifacts(run)
        if existing:
            raise ValueError(
                "A prospective audit claims nothing in the run could have been seen when it "
                f"was fixed, and {run} already holds {existing}. Audit the run before it "
                "starts, or build a retrospective one without --prospective.")
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

    rows = overlap_rows(bank_ids, specs, exposed_by_scene, scheduled, replay_prompts)

    def exposed(spec_ids: list[str]) -> int:
        return sum(1 for spec_id in spec_ids
                   if sha256_json(canonical_scene(specs[spec_id])) in exposed_by_scene)

    audit: dict[str, Any] = {
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
    if not prospective:
        return audit
    return with_outcome_side(audit, split, specs, exposed_by_scene, scheduled, replay_prompts)


def with_outcome_side(audit: dict[str, Any], split: dict[str, Any],
                      specs: dict[str, SceneSpec], exposed_by_scene: dict[str, list[str]],
                      scheduled: dict[str, int],
                      replay_prompts: set[str]) -> dict[str, Any]:
    """The half a retrospective audit is not allowed to have.

    Everything here picks a subset of the outcome population, which decides
    which prompts a sensitivity analysis is computed over. That is only a
    sensitivity analysis and not a choice if it was fixed before any outcome
    existed, which is what the caller has just checked and what the flag
    below asserts. `v4_decoupling_report.py` refuses the file otherwise.
    """

    outcome = list(split["outcome"])
    clusters = cluster(specs, outcome)
    conservative_exposed: dict[str, list[str]] = collections.defaultdict(list)
    for prompt_id in sorted(split["train"]):
        conservative_exposed[sha256_json(canonical_scene(specs[prompt_id]))].append(prompt_id)

    def disjoint(exposure: dict[str, list[str]]) -> list[str]:
        return sorted(spec_id for spec_id in outcome
                      if sha256_json(canonical_scene(specs[spec_id])) not in exposure)

    strict_ids = disjoint(exposed_by_scene)
    conservative_ids = disjoint(conservative_exposed)
    outcome_rows = overlap_rows(outcome, specs, exposed_by_scene, scheduled, replay_prompts)
    audit.update({
        "kind": PROSPECTIVE_KIND,
        "created_before_any_outcome_evaluation_artifact": True,
        "consumer": ("scripts/v4_decoupling_report.py (outcome sensitivity) and "
                     "scripts/v4_gradient_sensitivity.py --audit (probe exclusions)"),
        "within_outcome_clusters": clusters,
        "within_outcome_duplicate_scene_groups": [group for group in clusters
                                                  if len(group["spec_ids"]) > 1],
        "scene_disjoint_outcome_sensitivity": {
            "spec_ids": strict_ids,
            "spec_ids_sha256": sha256_json(strict_ids),
            "canonical_scene_clusters": cluster(specs, strict_ids),
            "status": "prespecified_before_any_outcome_evaluation_artifact",
        },
        "conservative_all_train_partition_sensitivity": {
            "spec_ids": conservative_ids,
            "spec_ids_sha256": sha256_json(conservative_ids),
            "overlap": overlap_rows(outcome, specs, conservative_exposed,
                                    scheduled, replay_prompts),
            "equals_scheduled_exposure": conservative_ids == strict_ids,
            "note": ("Exposure taken as the whole train partition rather than the generation "
                     "schedule. The two coincide whenever rounds x prompts_per_round covers "
                     "the partition, and `equals_scheduled_exposure` says whether they did."),
        },
        "recommendations": RECOMMENDATIONS,
    })
    audit["overlap"] = {
        "outcome": outcome_rows,
        "probe_partition": overlap_rows(list(split["probe"]), specs, exposed_by_scene,
                                        scheduled, replay_prompts),
        "actual_probe_bank": audit["overlap"]["actual_probe_bank"],
    }
    audit["summary"].update({
        "outcome_n": len(outcome),
        "outcome_unique_canonical_scenes": len(clusters),
        "outcome_repeated_scene_groups": len(audit["within_outcome_duplicate_scene_groups"]),
        "outcome_prompts_in_repeated_scene_groups": sum(
            len(group["spec_ids"]) for group in audit["within_outcome_duplicate_scene_groups"]),
        "scene_disjoint_outcome_n": len(strict_ids),
        "conservative_all_train_partition_outcome_overlap_n": len(outcome) - len(conservative_ids),
    })
    del audit["retrospective_notice"]
    return audit


def build_representatives(audit: dict[str, Any], parent_sha256: str) -> dict[str, Any]:
    """One outcome prompt per scene, so the sensitivity has independent rows.

    scripts/v4_decoupling_report.py consumes this and recomputes the rule from
    the parent audit before using it, refusing any file that disagrees -- so
    the only thing this adds is the artifact, not a choice. It was still
    missing: runs/v4/decoupling-pilot-20260906 has one, no script in the tree
    wrote it, and arm B would have gone without the analysis its comparison
    arm has.

    Written from the parent's bytes as they landed on disk, because the
    report matches `parent_scene_audit_sha256` against the file it read.
    """

    sensitivity = set(audit["scene_disjoint_outcome_sensitivity"]["spec_ids"])
    representatives = sorted(
        ({"scene_sha256": group["scene_sha256"], "spec_id": min(group["spec_ids"])}
         for group in audit["within_outcome_clusters"]
         if set(group["spec_ids"]) <= sensitivity),
        key=lambda row: row["spec_id"])
    spec_ids = [row["spec_id"] for row in representatives]
    # Counted off the clusters rather than read out of `summary`, so this also
    # runs against the pilot's audit, which is the same schema_version but
    # predates the outcome-side summary keys. That file is the only ground
    # truth there is for the rule, and a builder that cannot read it cannot be
    # checked against it.
    outcome_n = sum(len(group["spec_ids"]) for group in audit["within_outcome_clusters"])
    return {
        "schema_version": 1,
        "kind": REPRESENTATIVE_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_before_any_outcome_evaluation_artifact": True,
        "selection_rule": REPRESENTATIVE_RULE,
        "parent_scene_audit_sha256": parent_sha256,
        "spec_ids": spec_ids,
        "spec_ids_sha256": sha256_json(spec_ids),
        "spec_ids_hash_encoding": REPRESENTATIVE_ENCODING,
        "representatives": representatives,
        "n_independent_scene_representatives": len(representatives),
        "status": ("supplementary; conditional independence and sparse-binary CP "
                   f"assumptions; does not replace original{outcome_n} "
                   f"or sensitivity{len(sensitivity)}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="Existing run directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write it; RUN/audit-splits/ requires --prospective")
    parser.add_argument("--prospective", action="store_true",
                        help="Freeze the outcome-side subsets too. Only for a run that has "
                             "not produced anything yet; the run directory is checked.")
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must already exist")
    in_audit_splits = "audit-splits" in args.output.resolve().parts
    if in_audit_splits and not args.prospective:
        parser.error("audit-splits/ is the path v4_decoupling_report.py reads, and a "
                     "retrospective audit is not admissible there; see the module docstring")
    if args.prospective and not in_audit_splits:
        parser.error("A prospective audit is for the report, which only reads "
                     "RUN/audit-splits/scene_overlap.json; writing it elsewhere makes a "
                     "freeze nobody consumes")
    audit = build_audit(args.outdir, args.config, prospective=args.prospective)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, audit)
    summary = audit["summary"]
    print(f"{VERSION}: wrote {args.output}")
    child = None
    if args.prospective:
        # Not a separate command, because the pair is checked as a pair: the
        # child carries the parent's file digest and the report refuses a child
        # whose parent has moved. Two commands is two chances to freeze one.
        child = build_representatives(audit, sha256_file(args.output))
        child_path = args.output.parent / "scene_representatives.json"
        atomic_write_json(child_path, child)
        print(f"{VERSION}: wrote {child_path}")
    print(f"  probe bank {summary['probe_bank_n']} prompts in "
          f"{summary['probe_bank_scene_n']} scenes; excluding "
          f"{summary['actual_probe_bank_overlap_n']} for training exposure")
    if args.prospective:
        print(f"  {child['n_independent_scene_representatives']} independent scene "
              f"representatives, one per scene whose outcome prompts are all "
              f"scene-disjoint")
        print(f"  outcome {summary['outcome_n']} prompts in "
              f"{summary['outcome_unique_canonical_scenes']} scenes; "
              f"{summary['scene_disjoint_outcome_n']} scene-disjoint, "
              f"{summary['outcome_prompts_in_repeated_scene_groups']} sharing a scene "
              f"with another outcome prompt")


if __name__ == "__main__":
    main()
