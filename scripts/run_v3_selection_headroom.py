"""Pre-check: how much can any selector buy on this backbone, before training?

Gate C (Phase 5, 200-320 A800 GPU-hours) asks whether a perfect selector beats
Naive on external correctness after training. The mechanism it relies on is
measurable at step 0 without an optimizer: `Oracle@K - Naive@K`, the fraction of
pools where a correct candidate exists and the naive criterion misses it.

Runs in two stages, so the free half can be read immediately:

  stage 1 (no GPU)   Oracle@K, natural rate and informative rate, read straight
                     out of the bank probe's saved gold scores
  stage 2 (--score-cycle, one GPU)  Naive@K, by scoring log p(prompt | image) on
                     the already-rendered PNGs -- observation only, no generation

Uses only already-materialized weights and packets. Never downloads anything and
never writes into a frozen run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl
from selfsight.v3.headroom import attach_selections, headroom_report, load_natural_pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3_main_seed.yaml")
    parser.add_argument("--backbone-config", default=None)
    parser.add_argument(
        "--packets",
        action="append",
        required=True,
        help="Bank-probe packets directory; repeat for each shard",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--records",
        default=None,
        help="Scene manifest supplying prompt text; required with --score-cycle",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--score-cycle",
        action="store_true",
        help="Stage 2: load the backbone and score the Naive cycle criterion",
    )
    return parser.parse_args()


def _prompts_by_scene(records_path: Path) -> dict[str, str]:
    prompts = {}
    for row in read_jsonl(records_path):
        scene = row["scene"]
        prompts[str(scene["scene_id"])] = str(scene["prompt"])
    return prompts


def _score_cycle_arm(rows, packet_dirs, prompts, config, backbone_config, device, candidate_k, output):
    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.schemas import CandidateRecord
    from selfsight.v3.selectors import cycle_selection

    adapter = Showo2Adapter(backbone_config=backbone_config, device=device, lazy=False)
    if adapter.model_id != config["model"]["trainable_id"]:
        raise SystemExit(
            f"Backbone mismatch: adapter has {adapter.model_id}, "
            f"config expects {config['model']['trainable_id']}"
        )
    if adapter.revision != config["model"]["revision"]:
        raise SystemExit(
            f"Revision mismatch: adapter has {adapter.revision}, "
            f"config expects {config['model']['revision']}"
        )

    packets = {}
    for directory in packet_dirs:
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            packets[str(payload["scene_id"])] = payload

    print(f"[headroom] scoring the cycle criterion on {len(rows)} prompts", flush=True)
    selections: dict[str, str | None] = {}
    score_rows = []
    for index, row in enumerate(rows):
        prompt = prompts.get(row.prompt_id)
        if not prompt:
            raise SystemExit(f"No prompt text for {row.prompt_id} in --records")
        candidates = [
            CandidateRecord.from_dict(item)
            for item in packets[row.prompt_id]["candidates"][:candidate_k]
        ]
        decision = cycle_selection(
            prompt_id=row.prompt_id,
            candidates=candidates,
            prompt=prompt,
            scorer=adapter,
            selector_id=f"{adapter.model_id}/cycle",
            observer_revision=adapter.revision,
        )
        selections[row.prompt_id] = decision.selected_candidate_id
        score_rows.append(
            {
                "prompt_id": row.prompt_id,
                "family": row.family,
                "scores": decision.scores,
                "selected_candidate_id": decision.selected_candidate_id,
                "abstain": decision.abstain,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"[headroom]   {index + 1}/{len(rows)}", flush=True)
    atomic_write_jsonl(output / "cycle_scores.jsonl", score_rows)
    return attach_selections(rows, selections)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    candidate_k = int(config["candidate_bank"]["candidate_k"])

    packet_dirs = [Path(value).resolve() for value in args.packets]
    rows = load_natural_pools(packet_dirs, candidate_k=candidate_k)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No packets found")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_host_manifest(output / "host_manifest.json")

    if args.score_cycle:
        if not args.records:
            raise SystemExit("--score-cycle requires --records for the prompt text")
        rows = _score_cycle_arm(
            rows,
            packet_dirs,
            _prompts_by_scene(Path(args.records).resolve()),
            config,
            Path(args.backbone_config or config["model"]["backbone_config"]).resolve(),
            args.device or config["hardware"]["generator_device"],
            candidate_k,
            output,
        )

    report = headroom_report(rows, candidate_k=candidate_k)
    report["packet_dirs"] = [str(path) for path in packet_dirs]
    report["cycle_scored"] = bool(args.score_cycle)
    report["model_downloads_used"] = False
    report["a800_used"] = False
    atomic_write_json(output / "selection_headroom.json", report)

    overall = report["overall"]
    print(
        f"[headroom] n={overall['usable_prompts']} "
        f"natural={overall['natural_rate']:.3f} "
        f"oracle@{candidate_k}={overall['oracle_at_k']:.3f} "
        f"ceiling={overall['selection_ceiling']:.3f} "
        f"informative={overall['informative_rate']:.3f}",
        flush=True,
    )
    if overall["naive_cycle_rate"] is None:
        print("[headroom] naive not scored; re-run with --score-cycle", flush=True)
    else:
        print(
            f"[headroom] naive_cycle={overall['naive_cycle_rate']:.3f} "
            f"headroom={overall['headroom_over_naive']:.3f} "
            f"verdict={report['verdict']}",
            flush=True,
        )
    for family, block in sorted(report["by_family"].items()):
        print(
            f"[headroom]   {family:<10} n={block['usable_prompts']:<4} "
            f"natural={block['natural_rate']:.3f} "
            f"oracle={block['oracle_at_k']:.3f} "
            f"ceiling={block['selection_ceiling']:.3f} "
            f"informative={block['informative_rate']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
