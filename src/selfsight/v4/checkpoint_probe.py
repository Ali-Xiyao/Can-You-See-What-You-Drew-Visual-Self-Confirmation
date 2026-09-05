"""A frozen, held-out gradient instrument measured repeatedly during training.

The primary statistic remains cosine(mean per-prompt gradient, mean per-prompt
gradient), with an exact paired prompt bootstrap. Mean per-prompt cosine is a
separate descriptive statistic. A small fixed bank limits scratch space to
3 * n * dimension * 4 bytes; only Gram matrices and scalar records survive.
This is a static-image instrument, not the evolving training-image gradient.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from selfsight.data.questions import score_answer
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import AtomicQuestion, CandidateRecord, ObservationResult, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json
from selfsight.v3.paired import GramMatrices, paired_bootstrap_cosine, paired_bootstrap_difference
from selfsight.v4.probe import build_pools, spec_questions

VERSION = "v4-static-checkpoint-probe-1"
CRITERIA = ("naive", "rfo", "gold")
PAIRS = (("naive", "rfo", "gda_free"), ("naive", "gold", "gda_gold"))
PROMPTED_PREAMBLE = '''You were asked to draw a picture from this description:
"{prompt}"

Here is the picture you drew. Answer about what is actually in the picture.

{question}'''


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def questions_for(pool: dict[str, Any]) -> tuple[AtomicQuestion, ...]:
    return tuple(AtomicQuestion.from_dict(item) for item in pool["questions"])


def validate_observation(observation: ObservationResult, pool: dict[str, Any],
                         candidate: dict[str, Any], *, identity: tuple[str, str]) -> None:
    """Check the image, observer, complete question order and raw normalization."""
    if (observation.observer_id, observation.observer_revision) != identity:
        raise ValueError("Observation model/revision differs from the frozen identity")
    if observation.rgb_sha256 != candidate["rgb_sha256"]:
        raise ValueError("Observation image digest differs from the frozen candidate")
    questions = questions_for(pool)
    if [a.question_id for a in observation.answers] != [q.question_id for q in questions]:
        raise ValueError("Observation question IDs/order differ from the frozen questions")
    for answer, question in zip(observation.answers, questions):
        if answer.error:
            continue  # Errors are recorded and drop the whole paired pool below.
        normalized = score_answer(answer.raw_answer, question)
        if (normalized.normalized, normalized.abstain) != (
                answer.normalized_answer, answer.abstain):
            raise ValueError(f"Stale answer normalization: {answer.question_id}")


def choose_pools(rows: list[dict[str, Any]], split: dict[str, Any], *, maximum: int,
                 seed: int, minimum: int = 4) -> list[dict[str, Any]]:
    """At most one pool per spec, chosen without inspecting observer answers."""
    if not 4 <= minimum <= maximum:
        raise ValueError("Require 4 <= minimum <= maximum")
    parts = {name: set(split[name]) for name in ("train", "outcome", "probe")}
    if any(parts[a] & parts[b] for a, b in (("train", "outcome"),
                                           ("train", "probe"), ("outcome", "probe"))):
        raise ValueError("Training, outcome and probe specs overlap")
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["prompt_id"]):
        if row["spec_id"] in parts["probe"] and {c["correct"] for c in row["candidates"]} == {True, False}:
            unique.setdefault(row["spec_id"], row)
    ordered = sorted(unique.values(), key=lambda item: sha256_json(
        [VERSION, seed, item["spec_id"]]))[:maximum]
    if len(ordered) < minimum:
        raise ValueError(f"Only {len(ordered)} held-out balanced specs; require {minimum}")
    return ordered


def freeze_bank(*, source: Path, split_path: Path, config: dict[str, Any],
                destination: Path, maximum: int = 16, minimum: int = 4,
                rfo_config: Path = Path("configs/observers/qwen2vl_2b.yaml")) -> dict[str, Any]:
    """Copy validated existing frozen RFO answers; never silently re-question images."""
    import yaml

    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected = choose_pools(read_jsonl(source / "pools.jsonl"), split,
                            maximum=maximum, minimum=minimum, seed=int(config["seed"]))
    source_runs = json.loads((source / "runs.json").read_text(encoding="utf-8"))["runs"]
    rebuilt = {p.prompt_id: p for p in build_pools(tuple(source_runs))}
    rfo = yaml.safe_load(rfo_config.read_text(encoding="utf-8"))
    if rfo.get("trainable", False):
        raise ValueError("RFO observer must remain frozen")
    identity = (str(rfo["observer_id"]), str(rfo["revision"]))
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for item in read_jsonl(source / "observations.rfo.jsonl"):
        key = (item["prompt_id"], item["candidate_id"])
        if key in observations:
            raise ValueError(f"Duplicate RFO answer record: {key}")
        observations[key] = item["observation"]
    pools = []
    for original in selected:
        pool = json.loads(json.dumps(original))
        current = rebuilt[pool["prompt_id"]]
        if pool["questions"] != [as_serializable(q) for q in spec_questions(current.spec)]:
            raise ValueError("Source uses a different question version; acquire fresh RFO answers")
        if pool["prompt"] != current.spec.prompt:
            raise ValueError("Source prompt changed")
        if pool["candidates"] != [as_serializable(c) for c in current.candidates]:
            raise ValueError("Source candidates/gold differ from current adjudicated source")
        for candidate in pool["candidates"]:
            candidate["rgb_sha256"] = rgb_sha256(candidate["image_path"])
            candidate["rfo_observation"] = observations[(pool["prompt_id"], candidate["candidate_id"])]
            observation = ObservationResult.from_dict(candidate["rfo_observation"])
            validate_observation(observation, pool, candidate, identity=identity)
            if any(a.abstain or a.error or a.normalized_answer is None for a in observation.answers):
                raise ValueError("Frozen RFO has missing answers; do not select a bank by answer quality")
        pools.append(pool)
    payload = {
        "schema_version": VERSION, "config_digest": sha256_json(config),
        "split": split, "split_sha256": sha256_file(split_path),
        "source": str(source.resolve()),
        "source_hashes": {name: sha256_file(source / name) for name in
                          ("pools.jsonl", "observations.rfo.jsonl", "runs.json")},
        "rfo": {"observer_id": identity[0], "revision": identity[1]},
        "requested_prompts": maximum, "actual_prompts": len(pools),
        "selection_rule": "split.probe; balanced; lexicographically first pool per spec; hash order",
        "pools": pools,
    }
    payload["fingerprint"] = sha256_json(payload)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "bank.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError("Refusing to replace an existing bank with different provenance")
    atomic_write_json(path, payload)
    return payload


def load_bank(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    bank = json.loads((directory / "bank.json").read_text(encoding="utf-8"))
    expected = bank.pop("fingerprint")
    if sha256_json(bank) != expected:
        raise ValueError("Frozen bank fingerprint mismatch")
    bank["fingerprint"] = expected
    if bank["schema_version"] != VERSION or bank["config_digest"] != sha256_json(config):
        raise ValueError("Frozen bank version/config mismatch")
    for pool in bank["pools"]:
        for candidate in pool["candidates"]:
            if rgb_sha256(candidate["image_path"]) != candidate["rgb_sha256"]:
                raise ValueError("Frozen bank image changed")
    return bank


def select_pool(pool: dict[str, Any], naive: dict[str, ObservationResult]) -> dict[str, Any]:
    """Complete-case pairing: one missing atomic answer drops the whole pool."""
    observations = {"naive": naive, "rfo": {
        c["candidate_id"]: ObservationResult.from_dict(c["rfo_observation"])
        for c in pool["candidates"]}}
    issues = [{"arm": arm, "candidate_id": cid, "question_id": a.question_id,
               "error": a.error, "abstain": a.abstain}
              for arm, answers in observations.items() for cid, result in answers.items()
              for a in result.answers if a.error or a.abstain or a.normalized_answer is None]
    row: dict[str, Any] = {"prompt_id": pool["prompt_id"], "spec_id": pool["spec_id"],
                           "dropped": bool(issues), "missing_answers": issues}
    if issues:
        return row
    candidates = [CandidateRecord(
        candidate_id=c["candidate_id"], prompt_id=pool["prompt_id"], scene_id=pool["spec_id"],
        sampling_seed=c["sampling_seed"], image_path=c["image_path"], rgb_sha256=c["rgb_sha256"],
        generator_id="frozen-bank", generator_revision="frozen-bank", checkpoint_id="frozen-bank")
        for c in pool["candidates"]]
    selections, scores = {}, {}
    for arm in ("naive", "rfo"):
        decision = select_candidate(prompt_id=pool["prompt_id"], arm=arm, candidates=candidates,
            observations=observations[arm], questions=questions_for(pool), selector_id=VERSION,
            observer_revision=VERSION)
        if decision.abstain:
            raise RuntimeError("Complete-case selector unexpectedly abstained")
        selections[arm] = decision.selected_candidate_id
        scores[arm] = decision.scores
    selections["gold"] = max(pool["candidates"], key=lambda c: (
        c["correct"], -c["sampling_seed"], c["candidate_id"]))["candidate_id"]
    return row | {"selected": selections, "scores": scores}


def gram_from_arrays(left: Any, right: Any, *, prompt_ids: tuple[str, ...],
                     left_name: str, right_name: str, chunk_elements: int = 1 << 20) -> GramMatrices:
    """Exact float64 accumulation over float32 vectors, bounded parameter chunks."""
    if left.shape != right.shape or left.shape[0] != len(prompt_ids) or len(prompt_ids) < 2:
        raise ValueError("Paired gradient shapes/prompt order mismatch")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Duplicate prompts invalidate paired resampling")
    n, dimension = left.shape
    ll, rr, lr = (np.zeros((n, n), dtype=np.float64) for _ in range(3))
    width = max(1, chunk_elements // n)
    for start in range(0, dimension, width):
        lblock = np.asarray(left[:, start:start + width], dtype=np.float64)
        rblock = np.asarray(right[:, start:start + width], dtype=np.float64)
        if not np.isfinite(lblock).all() or not np.isfinite(rblock).all():
            raise FloatingPointError("Non-finite gradient vector")
        ll += lblock @ lblock.T
        rr += rblock @ rblock.T
        lr += lblock @ rblock.T
    return GramMatrices(left_name, right_name, prompt_ids, ll, rr, lr)


def summarize_gram(gram: GramMatrices, *, resamples: int, seed: int) -> dict[str, Any]:
    norms = np.sqrt(np.clip(np.diag(gram.g_ll), 0, None) * np.clip(np.diag(gram.g_rr), 0, None))
    finite = norms > 0
    per_prompt = np.full(gram.n, np.nan)
    per_prompt[finite] = np.clip(np.diag(gram.g_lr)[finite] / norms[finite], -1, 1)
    report: dict[str, Any] = {
        "primary_definition": "cosine(mean_prompt_gradient_left, mean_prompt_gradient_right)",
        "mean_per_prompt_cosine": float(per_prompt[finite].mean()) if finite.any() else None,
        "finite_per_prompt_cosines": int(finite.sum()),
        "per_prompt_cosines": [float(v) if np.isfinite(v) else None for v in per_prompt],
        "rms_norm_left": float(np.sqrt(np.diag(gram.g_ll).mean())),
        "rms_norm_right": float(np.sqrt(np.diag(gram.g_rr).mean())),
    }
    try:
        report.update(paired_bootstrap_cosine(gram, resamples=resamples, seed=seed))
        if not np.isfinite(report["cosine"]):
            report["cosine"] = None
            report["status"] = "degenerate_mean_gradient"
        else:
            report["status"] = "ok"
    except FloatingPointError as error:
        report.update({"cosine": None, "status": "degenerate_bootstrap", "reason": str(error),
                       "n_prompts": gram.n})
    return report


def checkpoint_identity(checkpoint: Path | None, config: dict[str, Any]) -> dict[str, Any]:
    if checkpoint is None:
        return {"type": "seeded_base", "seed": int(config["seed"]), "step": 0, "round_index": -1}
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_digest"] != sha256_json(config):
        raise ValueError("Checkpoint and probe configuration differ")
    for name, digest in manifest["files"].items():
        if sha256_file(checkpoint / name) != digest:
            raise ValueError(f"Checkpoint file changed: {name}")
    return {"type": "checkpoint", "path": str(checkpoint.resolve()),
            "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "adapter_sha256": manifest["files"]["adapter.pt"],
            "step": manifest["step"], "round_index": manifest["round_index"]}


def bind_run(destination: Path, payload: dict[str, Any]) -> str:
    """Resume only the same bank, checkpoint, estimator, implementation and config."""
    fingerprint = sha256_json(payload)
    value = payload | {"fingerprint": fingerprint}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "run.meta.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Probe resume fingerprint mismatch; use a new output directory")
    elif any(destination.iterdir()):
        raise ValueError("Nonempty probe directory lacks an identity manifest")
    else:
        atomic_write_json(path, value)
    return fingerprint


def close_scratch(arrays: dict[str, Any], *, root: Path, remove: bool) -> None:
    """Close Windows memmaps before deletion; validate each exact path inside root."""
    resolved_root = root.resolve()
    for array in arrays.values():
        path = Path(array.filename).resolve()
        if not path.is_relative_to(resolved_root) or path.parent.name != "gradient-tmp":
            raise ValueError("Scratch path escaped the probe output directory")
        array.flush()
        array._mmap.close()
        if remove:
            path.unlink()


def reference_differences(grams: dict[str, GramMatrices], reference: Path, *,
                          bank_fingerprint: str, resamples: int, seed: int) -> dict[str, Any]:
    prior = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    if prior["bank_fingerprint"] != bank_fingerprint or prior["checkpoint"]["step"] != 0:
        raise ValueError("Reference must be step zero on the identical frozen bank")
    with np.load(reference / "grams.npz", allow_pickle=False) as saved:
        if sha256_file(reference / "grams.npz") != prior["grams_sha256"]:
            raise ValueError("Reference Gram digest mismatch")
        previous_ids = tuple(str(p) for p in saved["prompt_ids"])
        output = {}
        for name, current in grams.items():
            common = tuple(p for p in current.prompt_ids if p in previous_ids)
            if len(common) < 4:
                output[name] = {"status": "insufficient_common_prompts", "n_common": len(common)}
                continue
            now_index = [current.prompt_ids.index(p) for p in common]
            old_index = [previous_ids.index(p) for p in common]
            now_gram = GramMatrices(current.left, current.right, common, *[
                matrix[np.ix_(now_index, now_index)]
                for matrix in (current.g_ll, current.g_rr, current.g_lr)])
            old_gram = GramMatrices(current.left, current.right, common, *[
                saved[f"{name}_{part}"][np.ix_(old_index, old_index)] for part in ("ll", "rr", "lr")])
            try:
                difference = paired_bootstrap_difference(now_gram, old_gram,
                                                         resamples=resamples, seed=seed)
                output[name] = difference | {
                    "point_difference": difference["difference"], "n_common": len(common),
                    "reference_step": 0, "method": "paired_bootstrap_difference",
                    "status": "ok", "prompt_ids": list(common)}
            except FloatingPointError as error:
                output[name] = {"status": "degenerate_bootstrap", "n_common": len(common),
                                "reason": str(error)}
    return output


def run_probe(*, config: dict[str, Any], bank_dir: Path, destination: Path,
              checkpoint: Path | None, device: str, arm: str = "base",
              resamples: int = 2000, reference: Path | None = None,
              targets_path: Path = Path("runs/readiness/showo2-1p5b/a4-lora-targets-r1.json"),
              scratch_limit_gib: float = 8.0) -> dict[str, Any]:
    """Load one model state, observe static images, compute exactly paired gradients.

    Partial observations and completed gradient prompts are resumable. Scratch
    vectors are float32 (no quantization/projection) and all precision, latent
    seeds and parameter names are recorded. Successful reports retain only Gram
    matrices, so repeated checkpoints do not multiply vector-storage cost.
    """
    from datetime import datetime, timezone
    import shutil

    bank = load_bank(bank_dir, config)
    identity = checkpoint_identity(checkpoint, config) | {"arm": arm}
    if resamples < 100 or scratch_limit_gib <= 0:
        raise ValueError("Need >=100 bootstrap resamples and positive scratch budget")
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if targets.get("forbidden_modules_selected"):
        raise ValueError("LoRA target audit includes forbidden modules")
    repo = Path(__file__).resolve().parents[3]
    sources = ("src/selfsight/v4/checkpoint_probe.py", "scripts/v4_checkpoint_probe.py",
               "src/selfsight/backbones/showo2.py", "src/selfsight/training/gradients.py",
               "src/selfsight/v3/paired.py", "src/selfsight/rfo/selection.py",
               "src/selfsight/data/questions.py")
    reference_hash = sha256_file(reference / "report.json") if reference else None
    fingerprint = bind_run(destination, {
        "schema_version": VERSION, "bank_fingerprint": bank["fingerprint"],
        "checkpoint": identity, "config_digest": sha256_json(config),
        "targets_sha256": sha256_file(targets_path),
        "backbone_config_sha256": sha256_file(config["model"]["backbone_config"]),
        "implementation_sha256": {name: sha256_file(repo / name) for name in sources},
        "resamples": resamples, "seed": int(config["seed"]),
        "reference_report_sha256": reference_hash, "device": device,
        "complete_case_policy": "any missing atomic answer drops the whole paired pool",
    })
    report_path = destination / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError("Completed probe fingerprint mismatch")
        if sha256_file(destination / "grams.npz") != report["grams_sha256"]:
            raise ValueError("Completed Gram digest mismatch")
        return report

    from selfsight.backbones.showo2 import Showo2Adapter, Showo2GenerationBatch
    from selfsight.training.checkpoint import load_checkpoint
    from selfsight.v4.train import seed_training, trainable_snapshot, parameter_digest

    started = datetime.now(timezone.utc).isoformat()
    seed_training(int(config["seed"]))
    backbone = Showo2Adapter(backbone_config=config["model"]["backbone_config"],
                            device=device, dtype=config["hardware"].get("precision", "bf16"), lazy=False)
    if backbone.model_id != config["model"]["trainable_id"]:
        raise ValueError("Live model does not match requested training backbone")
    lora = config["training"]["lora"]
    seed_training(int(config["seed"]))  # Match train/generate immediately before randomized LoRA A.
    audit = backbone.attach_lora(target_modules=tuple(targets["target_modules"]),
        rank=int(lora["rank"]), alpha=int(lora["alpha"]), dropout=float(lora["dropout"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]))
    if checkpoint is not None:
        load_checkpoint(checkpoint, model=backbone.model, optimizer=None, scheduler=None,
                        expected_config_digest=sha256_json(config))
    adapter_digest = parameter_digest(trainable_snapshot(backbone.model))
    dimension = int(audit["trainable_parameters"])
    model_identity = (backbone.model_id, backbone.revision)
    observations_path = destination / "observations.naive.jsonl"
    observations: dict[tuple[str, str], ObservationResult] = {}
    pools_by_id = {p["prompt_id"]: p for p in bank["pools"]}
    for row in read_jsonl(observations_path):
        key = (row["prompt_id"], row["candidate_id"])
        if key in observations:
            raise ValueError("Duplicate resumed observation")
        pool = pools_by_id[key[0]]
        candidate = next(c for c in pool["candidates"] if c["candidate_id"] == key[1])
        result = ObservationResult.from_dict(row["observation"])
        validate_observation(result, pool, candidate, identity=model_identity)
        observations[key] = result
    total = sum(len(p["candidates"]) for p in bank["pools"])
    with observations_path.open("a", encoding="utf-8") as handle:
        for pool in bank["pools"]:
            for candidate in pool["candidates"]:
                key = (pool["prompt_id"], candidate["candidate_id"])
                if key in observations:
                    continue
                seed_training(int(sha256_json([bank["fingerprint"], *key, "observe"])[:8], 16))
                questions = tuple(replace(q, text=PROMPTED_PREAMBLE.format(
                    prompt=pool["prompt"], question=q.text)) for q in questions_for(pool))
                result = backbone.observe_atoms(candidate["image_path"], questions)
                validate_observation(result, pool, candidate, identity=model_identity)
                observations[key] = result
                handle.write(json.dumps({"prompt_id": key[0], "candidate_id": key[1],
                    "observation": as_serializable(result)}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"observe {len(observations)}/{total}", flush=True)
    selections = [select_pool(pool, {c["candidate_id"]: observations[(pool["prompt_id"], c["candidate_id"])]
                  for c in pool["candidates"]}) for pool in bank["pools"]]
    atomic_write_json(destination / "selection.json", {"fingerprint": fingerprint, "rows": selections})
    kept = [row for row in selections if not row["dropped"]]
    if len(kept) < 4:
        raise RuntimeError(f"Only {len(kept)} complete paired prompts; need at least four")
    n = len(kept)
    needed = len(CRITERIA) * n * dimension * 4
    if needed > scratch_limit_gib * 1024 ** 3:
        raise RuntimeError(f"Gradient scratch needs {needed / 1024**3:.2f} GiB, budget {scratch_limit_gib}")
    scratch = destination / "gradient-tmp"
    scratch.mkdir(exist_ok=True)
    progress_path = destination / "gradient-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {
        "fingerprint": fingerprint, "dimension": dimension, "selection_digest": sha256_json(kept), "rows": []}
    if (progress["fingerprint"] != fingerprint or progress["dimension"] != dimension
            or progress["selection_digest"] != sha256_json(kept)):
        raise ValueError("Gradient resume metadata mismatch")
    if [row["prompt_id"] for row in progress["rows"]] != [row["prompt_id"] for row in kept[:len(progress["rows"])]]:
        raise ValueError("Gradient resume prompt order mismatch")
    existing_bytes = sum((scratch / f"{c}.f32").stat().st_size for c in CRITERIA
                         if (scratch / f"{c}.f32").exists())
    if shutil.disk_usage(destination).free < max(0, needed - existing_bytes) + 256 * 1024**2:
        raise RuntimeError("Insufficient disk space for bounded temporary gradient vectors")
    arrays = {}
    success = False
    try:
        for criterion in CRITERIA:
            path = scratch / f"{criterion}.f32"
            if path.exists() and path.stat().st_size != n * dimension * 4:
                raise ValueError("Gradient scratch dimension changed")
            if progress["rows"] and not path.exists():
                raise ValueError("Completed gradient rows lost their scratch vectors")
            arrays[criterion] = np.memmap(path, dtype="float32", mode="r+" if path.exists() else "w+",
                                          shape=(n, dimension))
        backbone.model.train()
        for index in range(len(progress["rows"]), n):
            row = kept[index]
            pool = pools_by_id[row["prompt_id"]]
            latent_seed = int(sha256_json([VERSION, int(config["seed"]), pool["prompt_id"], "latent"])[:8], 16) % 2**31
            already: dict[str, str] = {}
            losses = {}
            for criterion in CRITERIA:
                cid = row["selected"][criterion]
                if cid in already:
                    arrays[criterion][index] = arrays[already[cid]][index]
                    losses[criterion] = losses[already[cid]]
                    continue
                candidate = next(c for c in pool["candidates"] if c["candidate_id"] == cid)
                result = backbone.compute_lora_gradient(Showo2GenerationBatch(
                    prompts=(pool["prompt"],), images=(candidate["image_path"],),
                    sample_ids=(pool["prompt_id"],), latent_seed=latent_seed), criterion)
                vector = result.vector.detach().float().cpu().numpy().reshape(-1)
                if vector.size != dimension or not np.isfinite(vector).all():
                    raise FloatingPointError("Invalid per-prompt gradient")
                arrays[criterion][index] = vector
                losses[criterion] = float(result.loss)
                already[cid] = criterion
                del vector, result
            for array in arrays.values():
                array.flush()
            progress["rows"].append({"prompt_id": pool["prompt_id"], "latent_seed": latent_seed,
                                      "selected": row["selected"], "losses": losses,
                                      "gradient_evaluations": len(already)})
            atomic_write_json(progress_path, progress)
            print(f"gradients {index + 1}/{n}; {len(already)} distinct selected images", flush=True)
        prompt_ids = tuple(row["prompt_id"] for row in kept)
        grams, summaries, saved = {}, {}, {"prompt_ids": np.asarray(prompt_ids)}
        for left, right, name in PAIRS:
            gram = gram_from_arrays(arrays[left], arrays[right], prompt_ids=prompt_ids,
                                    left_name=left, right_name=right)
            grams[name] = gram
            summaries[name] = summarize_gram(gram, resamples=resamples, seed=int(config["seed"]))
            for label, value in (("ll", gram.g_ll), ("rr", gram.g_rr), ("lr", gram.g_lr)):
                saved[f"{name}_{label}"] = value
        np.savez_compressed(destination / "grams.npz", **saved)
        per_prompt = []
        for i, row in enumerate(progress["rows"]):
            per_prompt.append(row | {name: {
                "cosine": summaries[name]["per_prompt_cosines"][i],
                "norm_left": float(np.sqrt(gram.g_ll[i, i])),
                "norm_right": float(np.sqrt(gram.g_rr[i, i]))} for name, gram in grams.items()})
        atomic_write_json(destination / "per_prompt.json", {"fingerprint": fingerprint, "rows": per_prompt})
        report = {
            "schema_version": VERSION, "fingerprint": fingerprint,
            "bank_fingerprint": bank["fingerprint"], "checkpoint": identity,
            "n_bank": len(bank["pools"]), "n_retained": n, "n_dropped": len(bank["pools"]) - n,
            "prompt_ids": list(prompt_ids), "dropped_pools": [r for r in selections if r["dropped"]],
            "dimension": dimension, "scratch_bytes": needed, "stored_vector_dtype": "float32",
            "gram_accumulation_dtype": "float64", "model_id": backbone.model_id,
            "adapter_parameter_digest": adapter_digest,
            "model_revision": backbone.revision, "lora": lora,
            "parameter_names": [name for name, p in backbone.model.named_parameters() if p.requires_grad],
            "gradient_evaluations": sum(r["gradient_evaluations"] for r in progress["rows"]),
            "instrument": "fixed held-out existing images; reobserve/reselect with current checkpoint",
            "limitations": ["Small balanced static bank is an exploratory instrument, not evidence of a warning alone",
                            "Gold uses adjudicated binary image correctness; RFO remains the pinned frozen observer",
                            "Paired bootstrap reflects prompts, not independent training-run uncertainty"],
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "grams_sha256": sha256_file(destination / "grams.npz"), **summaries,
        }
        if reference:
            report["delta_from_reference"] = reference_differences(grams, reference,
                bank_fingerprint=bank["fingerprint"], resamples=resamples, seed=int(config["seed"]))
        atomic_write_json(report_path, report)
        success = True
        return report
    finally:
        close_scratch(arrays, root=destination, remove=success)
