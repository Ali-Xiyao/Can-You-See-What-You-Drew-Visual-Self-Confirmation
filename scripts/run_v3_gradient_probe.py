"""Gate B (L2): per-prompt paired gradients under three selection criteria.

The v2.x instrument compared a cross-criterion cosine against a DISJOINT
split-half floor. Those are not on the same scale: cross-criterion gradients
share prompts and most candidates so their noise cancels pairwise (measured
identical cosine 1.000), while split-half gradients are independent and doubled
(measured 0.08-0.29). v3.0 therefore stores ONE GRADIENT PER PROMPT and puts a
paired bootstrap CI directly on the cosine. Split-half is still computed, but
only as a reported per-sample SNR diagnostic; it no longer gates.

Criteria, all scored on the identical candidate pool:
  naive - the trainable model observing its own RGB
  rfo   - the frozen HETEROGENEOUS detector, which never selects training samples
  gold  - the deterministic generated-image verifier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.backbones.showo2 import Showo2Adapter, Showo2GenerationBatch
from selfsight.data.questions import build_question
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import (
    Atom,
    BlindObservationRequest,
    CandidateRecord,
    QuestionFormat,
    SceneSpec,
    as_serializable,
)
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl
from selfsight.v3.paired import (
    PerPromptGradientStore,
    gate_b_instrument_report,
    gram_matrices,
    paired_bootstrap_cosine,
    sample_noise_diagnostic,
)

CRITERIA = ("naive", "rfo", "gold")


def _stable_seed(*parts: object) -> int:
    return int(sha256_json(list(parts))[:8], 16)


def load_scene_index(records_paths):
    """scene_id -> (SceneSpec, primary question) from the source manifests."""

    index = {}
    for path in records_paths:
        for row in read_jsonl(Path(path).resolve()):
            scene = SceneSpec.from_dict(dict(row["scene"]))
            atom = Atom.from_dict(dict(row["atom"]))
            index[scene.scene_id] = (
                scene,
                (build_question(atom, question_format=QuestionFormat.OPEN),),
            )
    return index


def load_pools(shard_dirs, families, scene_index):
    """Balanced pools joined to their candidate records, gold scores and scenes."""

    pools = []
    for shard in shard_dirs:
        root = Path(shard).resolve()
        packets = {}
        for path in sorted((root / "packets").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            packets[payload["scene_id"]] = payload
        for row in read_jsonl(root / "pools.jsonl"):
            if not row["balanced"] or row["family"] not in families:
                continue
            prompt_id = row["prompt_id"]
            if prompt_id not in scene_index:
                raise SystemExit(f"No source record for pooled prompt {prompt_id}")
            packet = packets[prompt_id]
            wanted = set(row["candidate_ids"])
            records = [
                CandidateRecord.from_dict(item)
                for item in packet["candidates"]
                if item["candidate_id"] in wanted
            ]
            if len(records) != len(wanted):
                raise SystemExit(f"Pool {prompt_id} lost candidates during join")
            scene, questions = scene_index[prompt_id]
            pools.append(
                {
                    "prompt_id": prompt_id,
                    "family": row["family"],
                    "scene": scene,
                    "questions": questions,
                    "candidates": sorted(records, key=lambda c: c.candidate_id),
                    "gold": {
                        item["candidate_id"]: float(item["gold_score"])
                        for item in packet["scored"]
                        if item["candidate_id"] in wanted
                    },
                }
            )
    pools.sort(key=lambda item: item["prompt_id"])
    return pools


def gold_selection(pool):
    """Deterministic verifier choice; ties broken by seed then id, as everywhere else."""

    return max(
        pool["candidates"],
        key=lambda c: (pool["gold"][c.candidate_id], -c.sampling_seed, c.candidate_id),
    ).candidate_id


def observer_command(observer_cfg, device, ready_report):
    return [
        str(Path("envs/observer/python.exe").resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        str(observer_cfg["backend"]),
        "--model-id",
        str(observer_cfg["observer_id"]),
        "--revision",
        str(observer_cfg["revision"]),
        "--device",
        str(device),
        "--ready-report",
        str(ready_report),
    ]


def run_selection(pools, adapter, detector, observer_cfg, output):
    """One pass over every pool, producing a selected candidate per criterion."""

    decisions, rows = {}, []
    for index, pool in enumerate(pools):
        prompt_id = pool["prompt_id"]
        questions = pool["questions"]
        naive_obs, rfo_obs = {}, {}
        for candidate in pool["candidates"]:
            naive_obs[candidate.candidate_id] = adapter.observe_atoms(
                candidate.image_path, questions
            )
            rfo_obs[candidate.candidate_id] = detector.observe(
                BlindObservationRequest(
                    request_id=f"{prompt_id}:{candidate.candidate_id}",
                    image_path=candidate.image_path,
                    rgb_sha256=candidate.rgb_sha256,
                    questions=questions,
                )
            )
        naive = select_candidate(
            prompt_id=prompt_id,
            arm="naive",
            candidates=pool["candidates"],
            observations=naive_obs,
            questions=questions,
            selector_id=adapter.model_id,
            observer_revision=adapter.revision,
        )
        rfo = select_candidate(
            prompt_id=prompt_id,
            arm="rfo",
            candidates=pool["candidates"],
            observations=rfo_obs,
            questions=questions,
            selector_id=str(observer_cfg["observer_id"]),
            observer_revision=str(observer_cfg["revision"]),
        )
        gold = gold_selection(pool)
        if naive.abstain or rfo.abstain:
            # Abstention removes the prompt from EVERY criterion, never just one.
            rows.append({"prompt_id": prompt_id, "dropped": "selector_abstained"})
            continue
        decisions[prompt_id] = {
            "naive": naive.selected_candidate_id,
            "rfo": rfo.selected_candidate_id,
            "gold": gold,
        }
        rows.append(
            {
                "prompt_id": prompt_id,
                "family": pool["family"],
                "dropped": None,
                "selected": decisions[prompt_id],
                "gold_scores": pool["gold"],
                "naive_decision": as_serializable(naive),
                "rfo_decision": as_serializable(rfo),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"[gate-b] selection {index + 1}/{len(pools)}", flush=True)
    atomic_write_jsonl(output / "selection.jsonl", rows)
    return decisions


def gradient_for(adapter, pool, candidate_id, tag, seed):
    """One prompt, one selected candidate -> one LoRA gradient vector."""

    record = next(c for c in pool["candidates"] if c.candidate_id == candidate_id)
    batch = Showo2GenerationBatch(
        prompts=(pool["scene"].prompt,),
        images=(record.image_path,),
        sample_ids=(pool["prompt_id"],),
        latent_seed=_stable_seed("v3-gradient-loss", seed, pool["prompt_id"]),
    )
    return adapter.compute_lora_gradient_accumulated([batch], tag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3_main_seed.yaml")
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--pool-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--identical-subset", type=int, default=16)
    parser.add_argument(
        "--lora-target-config",
        default="runs/readiness/showo2-1p5b-hq/a4-lora-targets-r1.json",
        help="Hash-bound LoRA target selection from the A4 canary",
    )
    parser.add_argument("--generator-device", default=None)
    parser.add_argument("--observer-device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    observer_cfg = yaml.safe_load(
        Path(config["observers"]["detector_config"]).read_text(encoding="utf-8")
    )
    probe_cfg = config["gradient_probe"]
    families = set(args.family or ["existence", "spatial"])

    scene_index = load_scene_index(args.records)
    pools = load_pools(args.pool_dir, families, scene_index)
    if args.limit:
        pools = pools[: args.limit]
    if not pools:
        raise SystemExit("No balanced pools matched the requested families")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_host_manifest(output / "host_manifest.json")

    gen_device = args.generator_device or config["hardware"]["generator_device"]
    obs_device = args.observer_device or config["hardware"]["observer_device"]
    seed = int(config["seed"])
    print(
        f"[gate-b] {len(pools)} balanced pools "
        f"({sorted(families)}) | gen={gen_device} obs={obs_device}",
        flush=True,
    )

    adapter = Showo2Adapter(
        backbone_config=Path(config["model"]["backbone_config"]).resolve(),
        device=gen_device,
        lazy=False,
    )
    lora = config["training"]["lora"]
    targets = json.loads(Path(args.lora_target_config).read_text(encoding="utf-8"))
    if targets["model_id"] != adapter.model_id or targets["revision"] != adapter.revision:
        raise SystemExit("LoRA target selection was made for a different backbone")
    forbidden = set(lora["forbidden_trainable"])
    if any(part in forbidden for name in targets["target_modules"] for part in name.split(".")):
        raise SystemExit("LoRA target selection includes a forbidden module")
    lora_summary = adapter.attach_lora(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=tuple(targets["target_modules"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
    )

    command = observer_command(observer_cfg, obs_device, output / "detector_ready.json")
    with ObserverServiceClient(command, output / "detector_wire.jsonl") as detector:
        decisions = run_selection(pools, adapter, detector, observer_cfg, output)

    kept = [pool for pool in pools if pool["prompt_id"] in decisions]
    if len(kept) < 8:
        raise SystemExit(f"Only {len(kept)} pools survived selection; not diagnostic")
    print(f"[gate-b] {len(kept)} pools survived selection", flush=True)

    # ---- per-prompt gradients, identical prompt order for every criterion ----
    dimension = int(lora_summary["trainable_parameters"])
    stores = {
        name: PerPromptGradientStore(
            output / f"gradients-{name}.dat",
            criterion=name,
            dimension=dimension,
            capacity=len(kept),
            dtype=str(probe_cfg["store_dtype"]),
        )
        for name in CRITERIA
    }
    subset = max(min(args.identical_subset, len(kept)), 2)
    identical = PerPromptGradientStore(
        output / "gradients-naive-identical.dat",
        criterion="naive_identical",
        dimension=dimension,
        capacity=subset,
        dtype=str(probe_cfg["store_dtype"]),
    )
    # gram_matrices requires identical prompt sets, so mirror naive over the subset.
    naive_subset = PerPromptGradientStore(
        output / "gradients-naive-subset.dat",
        criterion="naive",
        dimension=dimension,
        capacity=subset,
        dtype=str(probe_cfg["store_dtype"]),
    )
    for index, pool in enumerate(kept):
        prompt_id = pool["prompt_id"]
        for name in CRITERIA:
            result = gradient_for(
                adapter, pool, decisions[prompt_id][name], f"{name}:{prompt_id}", seed
            )
            stores[name].add(prompt_id, result.vector)
            if name == "naive" and index < subset:
                naive_subset.add(prompt_id, result.vector)
        if index < subset:
            repeat = gradient_for(
                adapter, pool, decisions[prompt_id]["naive"], f"identical:{prompt_id}", seed
            )
            identical.add(prompt_id, repeat.vector)
        if (index + 1) % 10 == 0:
            print(f"[gate-b] gradients {index + 1}/{len(kept)}", flush=True)
    for store in (*stores.values(), identical, naive_subset):
        store.finalize()

    chunk = int(probe_cfg["gram_chunk_elements"])
    resamples = int(probe_cfg["paired_bootstrap_resamples"])
    boot_seed = int(probe_cfg["paired_bootstrap_seed"])

    measurements = {}
    for left, right in (("naive", "rfo"), ("naive", "gold"), ("rfo", "gold")):
        gram = gram_matrices(stores[left], stores[right], chunk_elements=chunk)
        measurements[f"{left}_vs_{right}"] = paired_bootstrap_cosine(
            gram, resamples=resamples, seed=boot_seed
        )

    identical_gram = gram_matrices(naive_subset, identical, chunk_elements=chunk)
    identical_cosine = paired_bootstrap_cosine(
        identical_gram, resamples=resamples, seed=boot_seed
    )
    # g_ll of any naive-left gram is the naive Gram; the diagnostic only reads that.
    noise = sample_noise_diagnostic(
        gram_matrices(stores["naive"], stores["rfo"], chunk_elements=chunk),
        splits=int(probe_cfg["sample_noise_splits"]),
        seed=boot_seed,
    )

    gate = gate_b_instrument_report(
        list(measurements.values()),
        max_ci_width=float(probe_cfg["max_paired_ci_width"]),
        identical_cosine_min=float(probe_cfg["identical_cosine_min"]),
        identical_cosine=float(identical_cosine["cosine"]),
    )
    report = {
        "schema_version": 1,
        "benchmark_version": "3.0",
        "stage": "v3_gate_b_gradient_instrument",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "detector": {
            "observer_id": observer_cfg["observer_id"],
            "revision": observer_cfg["revision"],
            "participates_in_training": False,
        },
        "families": sorted(families),
        "pools_loaded": len(pools),
        "pools_used": len(kept),
        "trainable_parameters": dimension,
        "measurements": measurements,
        "identical_selection": identical_cosine,
        "sample_noise_diagnostic_not_gating": noise,
        "gate_b": gate,
        "lora": lora_summary,
        "model_downloads_used": False,
        "a800_used": False,
    }
    atomic_write_json(output / "gate_b.json", report)

    print(f"\n[gate-b] identical cosine   = {identical_cosine['cosine']:.6f}")
    for name, value in measurements.items():
        print(
            f"[gate-b] {name:<16} = {value['cosine']:+.4f}  "
            f"CI [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]  width {value['ci_width']:.4f}"
        )
    print(f"[gate-b] split-half (diagnostic only) = {noise['mean']:.4f}")
    print(f"[gate-b] PASSED = {gate['passed']}")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
