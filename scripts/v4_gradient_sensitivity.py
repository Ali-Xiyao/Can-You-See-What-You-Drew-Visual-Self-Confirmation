"""CPU-only scene-disjoint sensitivity for the frozen main gradient instrument.

Reads RUN/gradient-probes/**/report.json and grams.npz, never runtime canaries.
Removes probe scenes marked as planned training exposure in the frozen scene
audit. Canonical scene clusters receive shared bootstrap multiplicities while
the point estimand stays the cosine of prompt-weighted mean gradients.

This supplementary analysis produces no warning declaration and cannot promote
the exploratory pilot to a confirmed finding. Main reports/bank/rules are read
only. Existing sensitivity outputs are archived before new data are folded in;
changed analysis rules, audit, or bank require a new output file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np

from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json

VERSION = "v4-gradient-scene-sensitivity-1"
PAIRS = ("gda_free", "gda_gold")


def cluster_weights(scenes: Sequence[str], *, resamples: int, seed: int) -> np.ndarray:
    """Draw G scenes G times, giving all prompts in a scene the same weight.

    Each original prompt has expected multiplicity one; the potentially varying
    total number of drawn prompts cancels from cosine normalization. We do not
    average scene means, which would change the original prompt-weighted target.
    """
    if not scenes or resamples < 100:
        raise ValueError("Need nonempty scene labels and >=100 resamples")
    unique = sorted(set(scenes))
    indices = [unique.index(scene) for scene in scenes]
    draws = np.random.default_rng(seed).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=resamples)
    return draws[:, indices].astype(np.float64)


def weighted_cosines(matrices: dict[str, np.ndarray], counts: np.ndarray) -> np.ndarray:
    numerator = ((counts @ matrices["lr"]) * counts).sum(axis=1)
    left = ((counts @ matrices["ll"]) * counts).sum(axis=1)
    right = ((counts @ matrices["rr"]) * counts).sum(axis=1)
    denominator = np.sqrt(np.clip(left, 0, None) * np.clip(right, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator > 0, numerator / denominator, np.nan)
    return np.clip(result, -1, 1)


def subset_gram(matrices: dict[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    return {part: matrix[np.ix_(indices, indices)] for part, matrix in matrices.items()}


def paired_interval(current: dict[str, np.ndarray], scenes: Sequence[str], *,
                    resamples: int, seed: int, baseline: dict[str, np.ndarray] | None = None,
                    confidence: float = .95, minimum_clusters: int = 4) -> dict[str, Any]:
    """Point and interval on one paired set, optionally current minus base."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be in (0, 1)")
    n = len(scenes)
    clusters = len(set(scenes))
    summary: dict[str, Any] = {
        "n_prompts": n, "n_clusters": clusters, "resamples": resamples, "seed": seed,
        "confidence": confidence, "method": "paired_scene_cluster_bootstrap",
        "point_estimand": "cosine(prompt_weighted_mean_gradient_left, prompt_weighted_mean_gradient_right)",
        "equal_scene_weights_for_point": False,
    }
    if n == 0:
        return summary | {"status": "no_retained_prompts", "point": None, "ci_low": None, "ci_high": None}
    weights = np.ones((1, n), dtype=np.float64)
    point = weighted_cosines(current, weights)[0]
    if baseline is not None:
        point -= weighted_cosines(baseline, weights)[0]
    summary["point"] = float(point) if np.isfinite(point) else None
    if not np.isfinite(point):
        return summary | {"status": "degenerate_mean_gradient", "ci_low": None, "ci_high": None}
    if clusters < minimum_clusters:
        return summary | {"status": "insufficient_clusters", "ci_low": None, "ci_high": None}
    counts = cluster_weights(scenes, resamples=resamples, seed=seed)
    draws = weighted_cosines(current, counts)
    if baseline is not None:
        draws -= weighted_cosines(baseline, counts)
    finite = draws[np.isfinite(draws)]
    summary["finite_resamples"] = int(finite.size)
    if finite.size < resamples / 2:
        return summary | {"status": "degenerate_bootstrap", "ci_low": None, "ci_high": None}
    alpha = (1 - confidence) / 2
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return summary | {"status": "ok", "ci_low": float(low), "ci_high": float(high)}


def read_partition(run: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any],
                                                        dict[str, str], set[str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    bank_path = run / "probe-bank" / "bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in bank.items() if key != "fingerprint"}
    if sha256_json(content) != bank["fingerprint"]:
        raise ValueError("Main bank content fingerprint mismatch")
    if (audit["provenance"]["bank_sha256"] != sha256_file(bank_path)
            or audit["provenance"]["bank_fingerprint"] != bank["fingerprint"]):
        raise ValueError("Scene audit does not describe this frozen main bank")
    spec_ids = [pool["spec_id"] for pool in bank["pools"]]
    if len(set(spec_ids)) != len(spec_ids):
        raise ValueError("Main bank must contain at most one pool per spec")
    scene_by_spec: dict[str, str] = {}
    for group in audit["within_probe_bank_clusters"]:
        if sha256_json(group["canonical_scene"]) != group["scene_sha256"]:
            raise ValueError("Audit canonical scene hash mismatch")
        for spec_id in group["spec_ids"]:
            if spec_id in scene_by_spec:
                raise ValueError("Spec assigned to multiple audit scene clusters")
            scene_by_spec[spec_id] = group["scene_sha256"]
    if set(scene_by_spec) != set(spec_ids):
        raise ValueError("Audit scene clusters do not cover exactly the main probe bank")
    excluded = set()
    for row in audit["overlap"]["actual_probe_bank"]:
        if row["spec_id"] in excluded or scene_by_spec.get(row["spec_id"]) != row["scene_sha256"]:
            raise ValueError("Invalid duplicate or mismatched excluded scene")
        excluded.add(row["spec_id"])
    if len(excluded) != audit["summary"]["actual_probe_bank_overlap_n"]:
        raise ValueError("Audit exclusion summary and rows differ")
    return audit, bank, scene_by_spec, excluded


def read_main_probe(path: Path, bank: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["bank_fingerprint"] != bank["fingerprint"]:
        raise ValueError(f"Report is not from the frozen main bank: {path}")
    arm, step = str(report["checkpoint"]["arm"]), int(report["checkpoint"]["step"])
    if arm not in {"base", "naive", "rfo_gold"} or step < 0 or (arm == "base" and step != 0):
        raise ValueError(f"Unrecognized main checkpoint identity: {path}")
    gram_path = path.parent / "grams.npz"
    if sha256_file(gram_path) != report["grams_sha256"]:
        raise ValueError(f"Gram checksum mismatch: {gram_path}")
    with np.load(gram_path, allow_pickle=False) as saved:
        prompt_ids = tuple(str(item) for item in saved["prompt_ids"])
        if prompt_ids != tuple(report["prompt_ids"]) or len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("Report and Gram prompt order differ or contain duplicates")
        if int(report["n_retained"]) != len(prompt_ids):
            raise ValueError("Reported gradient denominator differs from matrix rows")
        bank_prompt_ids = {pool["prompt_id"] for pool in bank["pools"]}
        if not set(prompt_ids) <= bank_prompt_ids:
            raise ValueError("Gradient matrix contains prompts outside the main bank")
        grams = {}
        for pair in PAIRS:
            matrices = {part: np.asarray(saved[f"{pair}_{part}"], dtype=np.float64)
                        for part in ("ll", "rr", "lr")}
            if any(matrix.shape != (len(prompt_ids), len(prompt_ids)) or not np.isfinite(matrix).all()
                   for matrix in matrices.values()):
                raise ValueError("Invalid or non-finite Gram matrix")
            for part in ("ll", "rr"):
                if not np.allclose(matrices[part], matrices[part].T, rtol=1e-9, atol=1e-12):
                    raise ValueError("Within-criterion Gram matrix is not symmetric")
            grams[pair] = matrices
    return {"arm": arm, "step": step, "checkpoint": report["checkpoint"],
            "prompt_ids": prompt_ids, "grams": grams,
            "source": {"report": {"path": str(path.resolve()), "sha256": sha256_file(path)},
                       "gram": {"path": str(gram_path.resolve()), "sha256": sha256_file(gram_path)}}}


def checkpoint_summary(row: dict[str, Any], baseline: dict[str, Any] | None, *,
                       prompt_to_spec: dict[str, str], scene_by_spec: dict[str, str],
                       excluded: set[str], rules: dict[str, Any]) -> dict[str, Any]:
    ids = tuple(p for p in row["prompt_ids"] if prompt_to_spec[p] not in excluded)
    indices = [row["prompt_ids"].index(p) for p in ids]
    scenes = [scene_by_spec[prompt_to_spec[p]] for p in ids]
    interval_options = {k: rules[k] for k in ("resamples", "seed", "confidence", "minimum_clusters")}
    result: dict[str, Any] = {"checkpoint": row["checkpoint"], "source": row["source"],
        "n_original_retained": len(row["prompt_ids"]), "n_scene_disjoint_retained": len(ids),
        "n_scene_clusters": len(set(scenes)), "prompt_ids": list(ids),
        "missing_scene_disjoint_spec_ids": sorted(set(prompt_to_spec.values()) - excluded
                                                    - {prompt_to_spec[p] for p in ids}),
        "removed_spec_ids": sorted({prompt_to_spec[p] for p in row["prompt_ids"]} & excluded)}
    for name in PAIRS:
        estimate = paired_interval(subset_gram(row["grams"][name], indices), scenes, **interval_options)
        result[name] = estimate | {"cosine": estimate["point"]}
    if baseline is None:
        result["delta_from_base"] = {"status": "no_base_report"}
        return result
    common = tuple(p for p in ids if p in baseline["prompt_ids"])
    current_index = [row["prompt_ids"].index(p) for p in common]
    base_index = [baseline["prompt_ids"].index(p) for p in common]
    common_scenes = [scene_by_spec[prompt_to_spec[p]] for p in common]
    result["delta_from_base"] = {}
    for name in PAIRS:
        estimate = paired_interval(subset_gram(row["grams"][name], current_index), common_scenes,
            baseline=subset_gram(baseline["grams"][name], base_index), **interval_options)
        result["delta_from_base"][name] = estimate | {
            "point_difference": estimate["point"], "reference_step": 0,
            "n_common": len(common), "prompt_ids": list(common)}
    return result


def build_report(run: Path, *, audit_path: Path | None = None,
                 resamples: int = 2000, seed: int = 20260906,
                 confidence: float = .95, minimum_clusters: int = 4) -> dict[str, Any]:
    if resamples < 100 or minimum_clusters < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid supplementary bootstrap settings")
    audit_path = audit_path or run / "audit-splits" / "scene_overlap.json"
    audit, bank, scenes, excluded = read_partition(run, audit_path)
    rules = {"version": VERSION, "resamples": resamples, "seed": seed, "confidence": confidence,
        "minimum_clusters": minimum_clusters, "exclusion_field": "overlap.actual_probe_bank",
        "clustering_field": "within_probe_bank_clusters", "point_weights": "one per original prompt",
        "bootstrap": "draw G canonical scenes G times; propagate scene multiplicity to every member prompt",
        "paired_difference": "current-minus-base on the exact prompt intersection using shared scene multiplicities",
        "warning_declarations": False}
    frozen_inputs = {"audit_sha256": sha256_file(audit_path),
        "bank_sha256": sha256_file(run / "probe-bank" / "bank.json"),
        "bank_fingerprint": bank["fingerprint"], "script_sha256": sha256_file(Path(__file__))}
    rows = [read_main_probe(path, bank)
            for path in sorted((run / "gradient-probes").glob("**/report.json"))]
    identities = [(row["arm"], row["step"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate main checkpoint arm/step reports")
    bases = [row for row in rows if row["step"] == 0]
    shared = [row for row in bases if row["arm"] == "base"]
    if len(shared) > 1:
        raise ValueError("Multiple shared base reports")
    prompt_to_spec = {pool["prompt_id"]: pool["spec_id"] for pool in bank["pools"]}
    analyses: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item["arm"], item["step"])):
        own_base = [base for base in bases if base["arm"] == row["arm"]]
        baseline = own_base[0] if own_base else (shared[0] if shared else None)
        analyses.setdefault(row["arm"], []).append(checkpoint_summary(row, baseline,
            prompt_to_spec=prompt_to_spec, scene_by_spec=scenes, excluded=excluded, rules=rules))
    status = "no_data" if not rows else ("base_only" if all(row["step"] == 0 for row in rows) else "supplementary")
    return {"schema_version": VERSION, "status": status,
        "message": "No completed main gradient report; runtime canaries are excluded" if not rows else
                   "Supplementary scene-disjoint analysis; no confirmed warning is declared",
        "rules": rules, "frozen_inputs": frozen_inputs,
        "analysis_fingerprint": sha256_json({"rules": rules, "frozen_inputs": frozen_inputs}),
        "n_bank": len(bank["pools"]), "excluded_spec_ids": sorted(excluded),
        "n_target_retained": len(bank["pools"]) - len(excluded),
        "retained_scene_clusters": len({scene for spec, scene in scenes.items() if spec not in excluded}),
        "canonical_scene_definition": audit["canonical_scene_key_definition"],
        "input_provenance": [row["source"] for row in rows], "arms": analyses,
        "limitations": ["Supplementary diagnostic; original outcomes, reports, thresholds and bank are unchanged",
                        "Intervals cover clustered prompt sampling, not independent training runs",
                        "Multiple looks are not adjusted and no warning or divergence is confirmed",
                        "Missing prompts are paired on the reported intersection, not imputed"]}


def write_versioned(path: Path, report: dict[str, Any]) -> str:
    """Reject rule changes; archive every prior data snapshot before replacement."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["analysis_fingerprint"] != report["analysis_fingerprint"]:
            raise ValueError("Sensitivity rules/audit/bank/implementation changed; use a new output file")
        content = {key: value for key, value in previous.items() if key != "created_utc"}
        if content == report:
            return "unchanged"
        history = path.parent / "gradient-sensitivity-history"
        history.mkdir(exist_ok=True)
        digest = sha256_file(path)
        archive = history / f"{path.stem}.{digest}.json"
        if archive.exists() and sha256_file(archive) != digest:
            raise ValueError("Existing sensitivity archive has different content")
        if not archive.exists():
            shutil.copyfile(path, archive)
    value = report | {"created_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(path, value)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="Existing pilot run directory")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must already exist")
    report = build_report(args.outdir, audit_path=args.audit, resamples=args.resamples, seed=args.seed)
    output = args.output or args.outdir / "gradient_sensitivity.json"
    action = write_versioned(output, report)
    print(f"{report['status']}: {report['message']}; {action} {output}")
    print(f"Frozen target {report['n_target_retained']}/{report['n_bank']} prompts, "
          f"{report['retained_scene_clusters']} canonical scenes; excluded {report['excluded_spec_ids']}")


if __name__ == "__main__":
    main()
