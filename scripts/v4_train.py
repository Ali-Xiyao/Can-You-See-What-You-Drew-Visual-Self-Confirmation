#!/usr/bin/env python
"""One arm, one seed, one card: the runner C1 and C2 launch in parallel.

Stages, in the order they run:

    split      freeze which prompts train and which are held out, once
    train      rounds of paired selection and SFT, resumable per round
    generate   draw the outcome set from one checkpoint, score cycle consistency
    score      fold that checkpoint's verified.jsonl into a metrics row
    report     D*, D_g when a gradient curve exists, and the lead between them

`generate` deliberately stops before detection. External correctness is measured
by `scripts/v4_run_pipeline.py detect` and `verify` over the manifest this stage
writes -- the same two detectors and the same adjudication ladder that produced
every number in sections 11 through 27. Running them from here would mean
loading a detector into a process that is already holding a 1.5B backbone plus
optimiser state, and would put a second copy of the verification path in the
repository for the external curve to drift away on.

The split is written once and then only read. `train` and `generate` both refuse
to run against a split file whose digest does not match the config they were
handed, because a re-split midway through a run is indistinguishable, in the
output, from a training set that quietly grew to include the outcome prompts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from selfsight.utils.hashing import sha256_json
from selfsight.v4.evaluate import (
    CheckpointMetrics,
    cycle_scores,
    divergence_report,
    evaluation_seed,
    external_correctness,
    read_metrics_csv,
    split_prompts,
    summarize_cycle,
    write_evaluation_manifest,
    write_metrics_csv,
)
from selfsight.v4.train import (
    ARMS,
    abandon_incomplete,
    build_schedule,
    completed_rounds,
    generate_and_select,
    load_training_corpus,
    pair_decisions,
    round_entries,
    train_arm,
    write_done,
)

MAIN_RUNS = ("runs/v4/main-2plus1", "runs/v4/main-1plus1plus1")
RFO_OBSERVER_CONFIG = "configs/observers/qwen2vl_2b.yaml"
LORA_TARGETS = "runs/readiness/showo2-1p5b/a4-lora-targets-r1.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def split_digest(config: dict[str, Any], runs: tuple[str, ...]) -> str:
    """What the split is a function of, and nothing else.

    Deliberately excludes the arm and the device: both arms on both cards must
    agree on which prompts are held out, or the two external curves are measured
    on different populations.
    """

    return sha256_json({
        "runs": sorted(runs),
        "seed": int(config["seed"]),
        "outcome": int(config["data"]["local_outcome"]),
        "probe": int(config["data"]["local_probe"]),
    })


# --------------------------------------------------------------------------


def stage_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    corpus = load_training_corpus(args.runs)
    split = split_prompts(
        corpus.prompt_ids,
        outcome=int(config["data"]["local_outcome"]),
        probe=int(config["data"]["local_probe"]),
        seed=int(config["seed"]),
    )
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; re-splitting mid-run silently moves prompts "
                         f"between training and evaluation. Pass --overwrite only on a "
                         f"run with no checkpoints.")
    payload = {
        "created": now(),
        "digest": split_digest(config, tuple(args.runs)),
        "runs": list(args.runs),
        "seed": int(config["seed"]),
        "train": list(split.train),
        "outcome": list(split.outcome),
        "probe": list(split.probe),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(split.train)} train / {len(split.outcome)} outcome / {len(split.probe)} probe")
    print(f"replay pool: {len(corpus.replay)} (image, question) pairs")
    print(f"wrote {path}")


def read_split(out_dir: Path, config: dict[str, Any], runs: tuple[str, ...]) -> dict[str, Any]:
    path = out_dir / "split.json"
    if not path.exists():
        raise SystemExit(f"Run the split stage first: {path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = split_digest(config, runs)
    if payload["digest"] != expected:
        raise SystemExit(
            f"Split was frozen under a different config: {payload['digest']} != {expected}. "
            f"Continuing would train on prompts this run holds out."
        )
    return payload


# --------------------------------------------------------------------------


def stage_train(args: argparse.Namespace) -> None:
    """Rounds of paired selection and SFT. Resumable at round granularity."""

    import torch

    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.observers.transformers_vlm import create_transformers_observer
    from selfsight.training.checkpoint import load_checkpoint, save_checkpoint

    config = load_config(args.config)
    out = Path(args.outdir)
    split = read_split(out, config, tuple(args.runs))
    corpus = load_training_corpus(args.runs)
    training = config["training"]

    schedule = build_schedule(
        split["train"],
        rounds=int(training["rounds"]),
        prompts_per_round=int(training["prompts_per_round"]),
        candidate_k=int(training["candidate_k"]),
        seed=int(config["seed"]),
        max_epochs=args.max_epochs,
    )

    done = completed_rounds(out)
    print(f"{len(done)} rounds already complete: {done}")

    backbone = Showo2Adapter(device=args.device, lazy=False)
    targets = json.loads(Path(LORA_TARGETS).read_text(encoding="utf-8"))
    lora = training["lora"]
    backbone.attach_lora(
        target_modules=targets["targets"],
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )

    observer_config = yaml.safe_load(Path(RFO_OBSERVER_CONFIG).read_text(encoding="utf-8"))
    if observer_config.get("trainable", False):
        raise SystemExit("The RFO arm's observer must be frozen")
    observer = create_transformers_observer(
        args.backend, str(observer_config["observer_id"]),
        str(observer_config["revision"]), args.device)

    digest = sha256_json(config)
    parameters = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizers = {
        arm: torch.optim.AdamW(parameters, lr=float(training["learning_rate"]),
                               weight_decay=float(training["weight_decay"]))
        for arm in ARMS
    }
    total_steps = int(training["rounds"]) * int(training["optimizer_steps_per_round"])
    warmup = max(1, round(total_steps * float(training["warmup_ratio"])))
    schedulers = {
        arm: torch.optim.lr_scheduler.LambdaLR(
            optimizers[arm], lr_lambda=lambda step: min(1.0, float(step + 1) / warmup))
        for arm in ARMS
    }

    for round_index in range(int(training["rounds"])):
        if round_index in done:
            continue
        round_dir = out / "rounds" / f"round-{round_index:03d}"
        abandon_incomplete(round_dir)
        round_dir.mkdir(parents=True)
        entries = round_entries(schedule, round_index)
        print(f"=== {now()} round {round_index}: {len(entries)} prompts ===")

        drawn: dict[str, Any] = {}
        for arm in ARMS:
            checkpoint = out / "checkpoints" / arm / f"round-{round_index:03d}"
            previous = out / "checkpoints" / arm / f"round-{round_index - 1:03d}"
            if previous.exists():
                load_checkpoint(previous, model=backbone.model, optimizer=optimizers[arm],
                                scheduler=schedulers[arm], expected_config_digest=digest)
            candidates, decisions = generate_and_select(
                arm=arm,
                backbone=backbone,
                observer=observer if arm == "rfo_self" else None,
                corpus=corpus,
                entries=entries,
                output_dir=round_dir / "candidates" / arm,
                checkpoint_id=f"{arm}-r{round_index:03d}",
            )
            drawn[arm] = (candidates, decisions)

        paired = pair_decisions(entries, {arm: drawn[arm][1] for arm in ARMS})
        kept = len(next(iter(paired.values())))
        print(f"    {kept}/{len(entries)} prompts survived pairing")

        reports = []
        for arm in ARMS:
            checkpoint = out / "checkpoints" / arm / f"round-{round_index:03d}"
            previous = out / "checkpoints" / arm / f"round-{round_index - 1:03d}"
            if previous.exists():
                load_checkpoint(previous, model=backbone.model, optimizer=optimizers[arm],
                                scheduler=schedulers[arm], expected_config_digest=digest)
            report = train_arm(
                arm=arm,
                backbone=backbone,
                optimizer=optimizers[arm],
                scheduler=schedulers[arm],
                decisions=paired[arm],
                candidates=drawn[arm][0],
                corpus=corpus,
                training=training,
                seed=int(config["seed"]),
                round_index=round_index,
            )
            save_checkpoint(
                checkpoint,
                model=backbone.model, optimizer=optimizers[arm], scheduler=schedulers[arm],
                config_digest=digest, config_values=config,
                step=(round_index + 1) * int(training["optimizer_steps_per_round"]),
                round_index=round_index,
                metadata=report,
            )
            reports.append(report)
            print(f"    {arm}: t2i {report['mean_t2i_loss']:.4f} "
                  f"grad {report['mean_gradient_norm_before_clip']:.3f}")

        write_done(round_dir, {
            "round": round_index, "finished": now(),
            "prompts": len(entries), "paired": kept, "arms": reports,
        })
    print(f"=== {now()} training complete ===")


# --------------------------------------------------------------------------


def stage_generate(args: argparse.Namespace) -> None:
    """Draw the outcome set from one checkpoint and score the internal curve.

    Stops before detection on purpose: see the module docstring.
    """

    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.training.checkpoint import load_checkpoint

    config = load_config(args.config)
    out = Path(args.outdir)
    split = read_split(out, config, tuple(args.runs))
    corpus = load_training_corpus(args.runs)

    checkpoint = out / "checkpoints" / args.arm / f"round-{args.round:03d}"
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise SystemExit(f"No checkpoint at {checkpoint}")

    backbone = Showo2Adapter(device=args.device, lazy=False)
    targets = json.loads(Path(LORA_TARGETS).read_text(encoding="utf-8"))
    lora = config["training"]["lora"]
    backbone.attach_lora(target_modules=targets["targets"], rank=int(lora["rank"]),
                         alpha=int(lora["alpha"]), dropout=float(lora["dropout"]),
                         gradient_checkpointing=False)
    load_checkpoint(checkpoint, model=backbone.model, optimizer=None, scheduler=None,
                    expected_config_digest=sha256_json(config))

    step = (args.round + 1) * int(config["training"]["optimizer_steps_per_round"])
    eval_dir = out / "evaluations" / args.arm / f"step-{step:05d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = list(split["outcome"])
    seeds = {prompt_id: evaluation_seed(seed=int(config["seed"]), arm=args.arm,
                                        step=step, prompt_id=prompt_id)
             for prompt_id in prompt_ids}
    drawn = backbone.generate_images(
        [corpus.specs[prompt_id].prompt for prompt_id in prompt_ids],
        [seeds[prompt_id] for prompt_id in prompt_ids],
        eval_dir / "images",
        f"{args.arm}-s{step:05d}",
        skip_existing=True,
    )
    images = {prompt_id: record.image_path for prompt_id, record in zip(prompt_ids, drawn)}

    write_evaluation_manifest(eval_dir, specs=corpus.specs, prompt_ids=prompt_ids,
                              images=images, seeds=seeds)

    scores = cycle_scores(backbone, specs=corpus.specs, images=images)
    mean, sem, count = summarize_cycle(scores)
    (eval_dir / "cycle.json").write_text(json.dumps({
        "arm": args.arm, "round": args.round, "step": step,
        "mean": mean, "sem": sem, "n": count, "scores": scores,
    }, indent=2), encoding="utf-8")
    print(f"{args.arm} step {step}: cycle {mean:.4f} +/- {sem:.4f} over {count} images")
    print(f"next: v4_run_pipeline.py detect --manifest {eval_dir / 'manifest.jsonl'} "
          f"--detector qwen3vl --device cuda:0")


def stage_score(args: argparse.Namespace) -> None:
    """Fold one verified checkpoint into the metrics table."""

    out = Path(args.outdir)
    rows = []
    metrics_path = out / "checkpoint_metrics.csv"
    if metrics_path.exists():
        rows = [row for row in read_metrics_csv(metrics_path)]

    added = 0
    for cycle_path in sorted(out.glob("evaluations/*/step-*/cycle.json")):
        eval_dir = cycle_path.parent
        cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
        rate, n, unadjudicated = external_correctness(eval_dir / "verified.jsonl")
        row = CheckpointMetrics(
            arm=cycle["arm"], round_index=int(cycle["round"]), step=int(cycle["step"]),
            internal_cycle=cycle["mean"], internal_sem=cycle["sem"],
            internal_n=int(cycle["n"]),
            external_correct=rate, external_n=n, external_unadjudicated=unadjudicated,
        )
        rows = [existing for existing in rows
                if (existing.arm, existing.step) != (row.arm, row.step)]
        rows.append(row)
        added += 1
    write_metrics_csv(metrics_path, rows)
    print(f"{added} checkpoints folded in; {len(rows)} rows at {metrics_path}")
    for row in sorted(rows, key=lambda item: (item.arm, item.step)):
        external = "--" if row.external_correct is None else f"{row.external_correct:.3f}"
        print(f"  {row.arm:9s} step {row.step:6d}  internal {row.internal_cycle:8.4f}  "
              f"external {external}")


def stage_report(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    rows = read_metrics_csv(out / "checkpoint_metrics.csv")
    payload = {}
    for arm in ARMS:
        if not any(row.arm == arm for row in rows):
            continue
        try:
            report = divergence_report(rows, arm)
        except ValueError as exc:
            payload[arm] = {"error": str(exc)}
            print(f"{arm}: {exc}")
            continue
        payload[arm] = report.to_dict()
        star = report.divergence.d_star
        print(f"{arm}: D* = {star if star is not None else 'not estimable'} "
              f"over {report.checkpoints} checkpoints "
              f"(floor {report.min_internal_slope:.3e})")
        if star is None:
            print(f"    {report.divergence.reason}")
    path = out / "divergence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--outdir", required=True, type=Path)
        target.add_argument("--config", default="configs/local_3090_showo2.yaml")
        target.add_argument("--runs", nargs="+", default=list(MAIN_RUNS))

    s = sub.add_parser("split", help="freeze the train/outcome/probe partition")
    common(s)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=stage_split)

    t = sub.add_parser("train", help="paired rounds, resumable")
    common(t)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--backend", default="qwen2vl")
    t.add_argument("--max-epochs", type=int, default=1,
                   help="passes over the prompt bank; >1 mixes memorisation into the curve")
    t.set_defaults(func=stage_train)

    g = sub.add_parser("generate", help="outcome images + internal curve for one checkpoint")
    common(g)
    g.add_argument("--arm", choices=list(ARMS), required=True)
    g.add_argument("--round", type=int, required=True)
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=stage_generate)

    c = sub.add_parser("score", help="fold verified checkpoints into the metrics table")
    c.add_argument("--outdir", required=True, type=Path)
    c.set_defaults(func=stage_score)

    r = sub.add_parser("report", help="D*, D_g and the lead")
    r.add_argument("--outdir", required=True, type=Path)
    r.set_defaults(func=stage_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
