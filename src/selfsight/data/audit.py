"""Reference-image verifier audit and split-integrity checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from selfsight.data.questions import build_primary_atom
from selfsight.data.verifier import verify_image
from selfsight.schemas import Atom, SceneSpec
from selfsight.utils.hashing import rgb_sha256
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def assert_zero_split_overlap(manifests: dict[str, str | Path]) -> dict[str, Any]:
    signatures: dict[str, set[str]] = {}
    templates: dict[str, set[str]] = {}
    prompt_texts: dict[str, set[str]] = {}
    for split, path in manifests.items():
        scenes = [SceneSpec.from_dict(record["scene"]) for record in read_jsonl(path)]
        signatures[split] = {scene.signature for scene in scenes}
        templates[split] = {scene.template_id for scene in scenes}
        prompt_texts[split] = {scene.prompt for scene in scenes}
        if len(signatures[split]) != len(scenes):
            raise AssertionError(f"Duplicate scene signature within {split}")
        if len(prompt_texts[split]) != len(scenes):
            raise AssertionError(f"Duplicate prompt within {split}")
    pairwise = {}
    names = sorted(manifests)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            overlap = {
                "signatures": len(signatures[left].intersection(signatures[right])),
                "templates": len(templates[left].intersection(templates[right])),
                "prompts": len(prompt_texts[left].intersection(prompt_texts[right])),
            }
            pairwise[key] = overlap
            if any(overlap.values()):
                raise AssertionError(f"Split leakage in {key}: {overlap}")
    return {"passed": True, "pairwise_overlap": pairwise}


def audit_reference_manifests(
    manifests: dict[str, str | Path],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    split_audit = assert_zero_split_overlap(manifests)
    family_counts: dict[str, int] = defaultdict(int)
    family_correct: dict[str, int] = defaultdict(int)
    family_parsed: dict[str, int] = defaultdict(int)
    failures = []
    total = correct = parsed = 0
    for split, manifest_path in manifests.items():
        for record in read_jsonl(manifest_path):
            atom = Atom.from_dict(record["atom"])
            result = verify_image(record["reference_image"], [atom])
            answer = result.answers[atom.atom_id]
            total += 1
            parsed += answer is not None
            correct += answer == atom.answer
            family_counts[atom.family.value] += 1
            family_parsed[atom.family.value] += answer is not None
            family_correct[atom.family.value] += answer == atom.answer
            if answer != atom.answer and len(failures) < 100:
                failures.append(
                    {
                        "split": split,
                        "scene_id": record["scene"]["scene_id"],
                        "expected": atom.answer,
                        "actual": answer,
                        "detections": len(result.detections),
                    }
                )
    report = {
        "schema_version": 1,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "coverage": parsed / total if total else 0.0,
        "family_accuracy": {key: family_correct[key] / value for key, value in family_counts.items()},
        "family_coverage": {key: family_parsed[key] / value for key, value in family_counts.items()},
        "split_audit": split_audit,
        "failures": failures,
        "gate_reference_pass": total > 0 and correct / total >= 0.98 and parsed / total >= 0.95,
        "human_agreement_status": "pending_manual_stratified_audit",
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def audit_tier_b_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require every registered pixel intervention to parse, be correct, and flip its atom."""

    totals: dict[str, int] = defaultdict(int)
    passing: dict[str, int] = defaultdict(int)
    failures: list[dict[str, Any]] = []
    total = 0
    for record in read_jsonl(manifest_path):
        pair = record["pair"]
        category = str(pair["category"])
        source = SceneSpec.from_dict(pair["source"])
        changed = SceneSpec.from_dict(pair["counterfactual"])
        source_atom = build_primary_atom(source)
        changed_atom = build_primary_atom(changed)
        source_result = verify_image(record["source_image"], [source_atom])
        changed_result = verify_image(record["counterfactual_image"], [changed_atom])
        source_answer = source_result.answers[source_atom.atom_id]
        changed_answer = changed_result.answers[changed_atom.atom_id]
        expected_source = str(pair["source_answer"])
        expected_changed = str(pair["counterfactual_answer"])
        rgb_valid = (
            rgb_sha256(record["source_image"]) == record["source_rgb_sha256"]
            and rgb_sha256(record["counterfactual_image"]) == record["counterfactual_rgb_sha256"]
            and record["source_rgb_sha256"] != record["counterfactual_rgb_sha256"]
        )
        passed = bool(
            source_answer is not None
            and changed_answer is not None
            and source_answer == expected_source
            and changed_answer == expected_changed
            and source_answer != changed_answer
            and rgb_valid
        )
        total += 1
        totals[category] += 1
        passing[category] += int(passed)
        if not passed and len(failures) < 100:
            failures.append(
                {
                    "pair_id": pair["pair_id"],
                    "category": category,
                    "expected": [expected_source, expected_changed],
                    "actual": [source_answer, changed_answer],
                    "rgb_valid": rgb_valid,
                }
            )
    report = {
        "schema_version": 1,
        "total_pairs": total,
        "passing_pairs": sum(passing.values()),
        "accuracy": sum(passing.values()) / total if total else 0.0,
        "category_counts": dict(totals),
        "category_passing": dict(passing),
        "failures": failures,
        "gate_tier_b_reference_pass": total == 400 and sum(passing.values()) == total,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def audit_tier_d_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the fixed 600-image Tier-D subset, RGB hashes, atoms, and pair completeness."""

    rows = list(read_jsonl(manifest_path))
    identifiers = [str(row.get("tier_d_id")) for row in rows]
    source_counts: dict[str, int] = defaultdict(int)
    stratum_counts: dict[str, int] = defaultdict(int)
    pair_roles: dict[str, set[str]] = defaultdict(set)
    failures: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        source_tier = str(row.get("source_tier"))
        stratum = str(row.get("tier_d_stratum"))
        role = str(row.get("image_role"))
        source_counts[source_tier] += 1
        stratum_counts[f"{source_tier}:{stratum}"] += 1
        if source_tier == "tier_b":
            pair_roles[str(row.get("source_record_id"))].add(role)
        atom = Atom.from_dict(row["atom"])
        actual_hash = rgb_sha256(row["reference_image"])
        result = verify_image(row["reference_image"], [atom])
        answer = result.answers[atom.atom_id]
        passed = actual_hash == row["reference_rgb_sha256"] and answer == atom.answer
        correct += int(passed)
        if not passed and len(failures) < 100:
            failures.append(
                {
                    "tier_d_id": row.get("tier_d_id"),
                    "expected": atom.answer,
                    "actual": answer,
                    "rgb_valid": actual_hash == row["reference_rgb_sha256"],
                }
            )
    expected_strata = {
        **{f"tier_a:{family}": 50 for family in ("existence", "count", "color", "size", "spatial", "binding")},
        **{
            f"tier_b:{category}": 60
            for category in (
                "count_delete",
                "color_change",
                "relation_left_right",
                "relation_size",
                "binding_swap",
            )
        },
    }
    pairs_complete = len(pair_roles) == 150 and all(
        roles == {"source", "counterfactual"} for roles in pair_roles.values()
    )
    conditions = {
        "exactly_600_unique_images": len(rows) == 600 and len(set(identifiers)) == 600,
        "registered_tier_balance": dict(source_counts) == {"tier_a": 300, "tier_b": 300},
        "registered_strata": dict(stratum_counts) == expected_strata,
        "tier_b_pairs_complete": pairs_complete,
        "all_rgb_and_atoms_valid": correct == len(rows),
    }
    report = {
        "schema_version": 1,
        "images": len(rows),
        "valid_images": correct,
        "source_counts": dict(source_counts),
        "stratum_counts": dict(stratum_counts),
        "tier_b_pairs": len(pair_roles),
        "conditions": conditions,
        "failures": failures,
        "gate_tier_d_manifest_pass": all(conditions.values()),
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
