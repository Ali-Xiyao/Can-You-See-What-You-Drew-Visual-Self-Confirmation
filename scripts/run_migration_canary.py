"""Produce the fixed 32-prompt Windows/A800 migration canary manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from selfsight.config import load_config, write_config_snapshot
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.data.verifier import verify_image
from selfsight.pilot.real_loop import _stable_seed
from selfsight.schemas import SceneSpec
from selfsight.showo_adapter import ShowoAdapter
from selfsight.utils.cuda import cuda_device_index, reset_cuda_peak_memory_stats
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_config_snapshot(config, output / "resolved_config.json")
    write_host_manifest(output / "host_manifest.json")
    adapter = ShowoAdapter(
        device=str(config.values["hardware"]["generator_device"]),
        trainable=False,
        generation_timesteps=int(config.values["model"]["generation_timesteps"]),
        guidance_scale=float(config.values["model"]["guidance_scale"]),
        temperature=float(config.values["model"]["temperature"]),
    )
    reset_cuda_peak_memory_stats(adapter.device)
    rows = []
    started = perf_counter()
    records = stable_stratified_sample(
        list(read_jsonl(args.probe_manifest)),
        32,
        stratum=lambda record: str(record["atom"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=20260827,
    )
    for record in records:
        scene = SceneSpec.from_dict(record["scene"])
        atom = build_primary_atom(scene)
        question = build_question(atom)
        seed = _stable_seed(20260827, "migration-canary", scene.scene_id)
        candidate = adapter.generate_images(
            [scene.prompt], [seed], output / "images", "migration-canary"
        )[0]
        candidate = replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
        observation = adapter.observe_atoms(candidate.image_path, (question,)).answers[0]
        verifier = verify_image(candidate.image_path, [atom])
        pixel_answer = verifier.answers[atom.atom_id]
        rows.append(
            {
                "scene_id": scene.scene_id,
                "family": scene.family.value,
                "sampling_seed": seed,
                "image_sha256": candidate.rgb_sha256,
                "expected_answer": question.expected_answer,
                "observer_answer": observation.normalized_answer,
                "observer_abstain": observation.abstain,
                "verifier_answer": pixel_answer,
                "verifier_coverage": verifier.coverage,
                "observer_correct": observation.normalized_answer == question.expected_answer,
                "verifier_correct": pixel_answer == question.expected_answer,
            }
        )
    if len(rows) != 32:
        raise RuntimeError(f"Migration canary requires exactly 32 rows, got {len(rows)}")
    elapsed = perf_counter() - started
    summary = {
        "schema_version": 1,
        "profile": config.values["profile"],
        "config_digest": config.digest,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "samples": len(rows),
        "observer_accuracy": sum(row["observer_correct"] for row in rows) / len(rows),
        "verifier_accuracy": sum(row["verifier_correct"] for row in rows) / len(rows),
        "verifier_coverage": sum(row["verifier_answer"] is not None for row in rows) / len(rows),
        "elapsed_seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "peak_gpu_bytes": int(
            torch.cuda.max_memory_allocated(cuda_device_index(adapter.device))
        ),
    }
    atomic_write_jsonl(output / "canary_rows.jsonl", rows)
    atomic_write_json(output / "canary_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
