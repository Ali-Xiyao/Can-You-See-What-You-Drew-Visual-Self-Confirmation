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
import subprocess
import time
from dataclasses import asdict
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
    self_selection_scores,
    split_prompts,
    summarize_cycle,
    summarize_selection,
    write_evaluation_manifest,
    write_metrics_csv,
)
from selfsight.v4.train import (
    ARMS,
    RFO_SELF,
    build_schedule,
    completed_rounds,
    generate_candidates,
    initialize_base_checkpoint,
    load_training_corpus,
    pair_decisions,
    parameter_digest,
    pending_rounds,
    prepare_round,
    previous_checkpoint,
    read_verdicts,
    restrict_replay,
    round_entries,
    select_by_observation,
    select_gold,
    seed_training,
    trainable_snapshot,
    train_arm,
    write_candidate_manifest,
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


def run_stage(command: list[str], log_path: Path) -> None:
    """Shell out, and keep the log even when it fails.

    The gold arm's selector runs detectors that live in another environment, so
    there is a process boundary here whether or not it is convenient. Capturing
    each call's output to its own file is what makes a round that died at the
    second detector distinguishable from one that died at the first.
    """

    # CreateProcess will not resolve a relative path written with forward
    # slashes, so "envs/observer/python.exe" -- the default this script ships
    # with -- fails with WinError 2 while "envs\observer\python.exe" and the
    # absolute form both work. Resolving the interpreter here covers detect,
    # crop and verify at once, and keeps the CLI defaults readable.
    interpreter = Path(command[0])
    if interpreter.exists():
        command = [str(interpreter.resolve()), *command[1:]]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nCOMMAND {subprocess.list2cmdline(command)}\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                       text=True, check=True)


def _same_cuda_device(left: str, right: str) -> bool:
    """True when two device strings name the same CUDA card.

    "cuda" and "cuda:0" are one card spelled two ways and torch.device does not
    call them equal, so the comparison is written out rather than trusted to ==.
    """
    import torch

    a, b = torch.device(left), torch.device(right)
    if a.type != "cuda" or b.type != "cuda":
        return False
    return (a.index or 0) == (b.index or 0)


def adjudicate(
    directory: Path,
    *,
    observer_python: str,
    core_python: str,
    device: str,
) -> Path:
    """Run the corpus's own ladder over a manifest and return verified.jsonl.

    Exactly the stages the corpus went through, in the order it went through
    them: both detectors, then the crop level for disputed objects, then verify.
    Skipping the crop level would make the gold arm's "correct" stricter than the
    corpus's, and the positive control would then be measured against an
    external curve built on a more forgiving rule than the one that trained it.
    """

    verified = directory / "verified.jsonl"
    if verified.exists():
        return verified
    for detector in ("qwen3vl", "internvl"):
        done = directory / f"detections.{detector}.jsonl"
        expected = sum(1 for _ in (directory / "manifest.jsonl").open(encoding="utf-8"))
        have = sum(1 for _ in done.open(encoding="utf-8")) if done.exists() else 0
        if have >= expected:
            continue
        # The ladder's card is shared with jobs that are not this project's, and
        # loading an 8B observer means asking for one 15 GiB block. That can lose
        # a race it would win a minute later: the first failure here reported
        # 22.76 GiB free in the same message that refused 15.17, because the free
        # figure is computed when the error is raised rather than when the
        # allocation was attempted.
        #
        # Retrying is cheap and cannot double-count -- detect appends, and rows
        # already written are skipped by the `have >= expected` test above, so a
        # second pass resumes rather than repeats. Losing a round of generation
        # to a transient is what is expensive: that is an hour, and it is what
        # the orchestrator's own retry would cost.
        for tries_left in (2, 1, 0):
            try:
                run_stage([observer_python, "scripts/v4_run_pipeline.py", "detect",
                           "--manifest", str(directory / "manifest.jsonl"),
                           "--detector", detector, "--device", device],
                          directory / f"detect.{detector}.log")
                break
            except subprocess.CalledProcessError:
                if not tries_left:
                    raise
                print(f"    detect {detector} failed, {tries_left} tries left, "
                      f"waiting 180s for the card", flush=True)
                time.sleep(180)
    run_stage([observer_python, "scripts/v4_run_pipeline.py", "crop",
               "--run", str(directory), "--device", device],
              directory / "crop.log")
    run_stage([core_python, "scripts/v4_run_pipeline.py", "verify",
               "--run", str(directory)],
              directory / "verify.log")
    return verified


def stage_train(args: argparse.Namespace) -> None:
    """Rounds of paired selection and SFT. Resumable at round granularity."""

    import torch

    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.observers.transformers_vlm import create_transformers_observer
    from selfsight.training.checkpoint import (
        capture_base_state, restore_arm_state, save_checkpoint)

    config = load_config(args.config)
    out = Path(args.outdir)
    split = read_split(out, config, tuple(args.runs))
    corpus = load_training_corpus(args.runs)
    corpus = restrict_replay(corpus, split["train"])
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
    pending = pending_rounds(int(training["rounds"]), done, getattr(args, "max_rounds", None))
    print(f"{len(done)} rounds already complete: {done}")
    if not pending:
        print("All scheduled rounds are already complete")
        return
    for arm in ARMS:
        previous_checkpoint(out, arm, pending[0])

    # This stage needs the ladder on a different card and there is no way round
    # it. The backbone stays resident across the whole round, so an adjudicator
    # sharing the card has to load beside 13.5 GB and dies part-way through its
    # shards -- on Windows as 0xC0000005, which reads as a crash rather than as
    # the capacity problem it is.
    #
    # Parking the backbone on the CPU for the ladder's duration was tried and
    # does not work: with LoRA attached and accelerate's hooks on the blocks,
    # moving the dispatched model segfaults the training process. It segfaults
    # after the round's images are drawn, so the cost of finding out is an hour
    # (STATUS 37).
    #
    # Hence: refuse at the top, before an hour of generation, rather than fail
    # deep in the round. Whoever hands this stage one card gets told in seconds.
    if _same_cuda_device(args.device, args.ladder_device):
        raise SystemExit(
            f"train needs the ladder on another card: --device {args.device} and "
            f"--ladder-device {args.ladder_device} are the same one. The backbone is "
            f"resident for the whole round and the adjudicator cannot load beside it.")
    seed_training(int(config["seed"]))
    backbone = Showo2Adapter(device=args.device, lazy=False)
    targets = json.loads(Path(LORA_TARGETS).read_text(encoding="utf-8"))
    lora = training["lora"]
    # The readiness artefact records which forbidden modules the selection let
    # through. It is empty, and the run should stop rather than train through it
    # if that ever changes: `forbidden_trainable` names the paths that would let
    # gradients into the understanding tower, and a run that quietly trained
    # them would answer a different question than the one being asked.
    leaked = sorted(targets.get("forbidden_modules_selected", []))
    if leaked:
        raise SystemExit(f"{LORA_TARGETS} selects forbidden modules: {leaked}")
    seed_training(int(config["seed"]))
    backbone.attach_lora(
        target_modules=targets["target_modules"],
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )

    # Pass 1 is Naive against RFO-Gold and neither arm consults an observer:
    # naive asks itself, gold reads the adjudication ladder. Loading one anyway
    # costs a model load and its memory on the same card the detectors need,
    # for a component that pass 1 never calls. The frozen-ness check still runs
    # whenever it *is* built, which is the part that matters.
    observer = None
    if RFO_SELF in ARMS:
        observer_config = yaml.safe_load(
            Path(RFO_OBSERVER_CONFIG).read_text(encoding="utf-8"))
        if observer_config.get("trainable", False):
            raise SystemExit("The RFO arm's observer must be frozen")
        observer = create_transformers_observer(
            args.backend, str(observer_config["observer_id"]),
            str(observer_config["revision"]), args.ladder_device)

    digest = sha256_json(config)
    # Both arms run through this one backbone, so "the weights the other arm
    # has not touched yet" has to be kept somewhere. Captured after attach_lora
    # and before any training, which is the only moment it is available.
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
    base_checkpoint = out / "checkpoints" / "base" / "round--01"
    initialize_base_checkpoint(base_checkpoint, model=backbone.model, optimizer=optimizers[ARMS[0]],
                               scheduler=schedulers[ARMS[0]], config=config)
    base_state = capture_base_state(backbone.model)

    for round_index in pending:
        round_dir = prepare_round(out, round_index)
        entries = round_entries(schedule, round_index)
        print(f"=== {now()} round {round_index}: {len(entries)} prompts ===")

        # One pool per arm, drawn by that arm's own checkpoint. The arms share
        # the schedule and the latent seeds, not the weights -- after round 0
        # they are different models and must draw their own candidates, or the
        # comparison would be "which selector picks better from the naive arm's
        # images" rather than "which selector trains a better model".
        pools: dict[str, Any] = {}
        for arm in ARMS:
            restore_arm_state(
                previous_checkpoint(out, arm, round_index),
                model=backbone.model, optimizer=optimizers[arm],
                scheduler=schedulers[arm], expected_config_digest=digest, base=base_state)
            pools[arm] = generate_candidates(
                backbone=backbone,
                corpus=corpus,
                entries=entries,
                output_dir=round_dir / "candidates" / arm,
                checkpoint_id=f"{arm}-r{round_index:03d}",
            )

        decisions: dict[str, Any] = {}
        for arm in ARMS:
            if arm == "rfo_gold":
                ladder_dir = round_dir / "ladder" / arm
                write_candidate_manifest(ladder_dir, corpus=corpus, pools=pools[arm])
                verified = adjudicate(ladder_dir, observer_python=args.observer_python,
                                      core_python=args.core_python,
                                      device=args.ladder_device)
                verdicts, unadjudicated = read_verdicts(verified, pools[arm])
                decisions[arm] = select_gold(pools=pools[arm], verdicts=verdicts)
                abstained = sum(1 for d in decisions[arm] if d.selected_candidate_id is None)
                print(f"    gold: {abstained}/{len(decisions[arm])} pools had nothing "
                      f"correct, {len(unadjudicated)} images unadjudicated")
            else:
                restore_arm_state(
                    previous_checkpoint(out, arm, round_index),
                    model=backbone.model, optimizer=optimizers[arm],
                    scheduler=schedulers[arm], expected_config_digest=digest,
                    base=base_state)
                decisions[arm] = select_by_observation(
                    arm=arm, backbone=backbone,
                    observer=observer if arm == RFO_SELF else None,
                    corpus=corpus, pools=pools[arm],
                    observation_path=round_dir / "observations" / f"{arm}.jsonl")

        paired = pair_decisions(entries, decisions)
        kept = len(next(iter(paired.values())))
        (round_dir / "selection.json").write_text(json.dumps({
            "round": round_index,
            "decisions": {arm: [asdict(decision) for decision in values]
                          for arm, values in decisions.items()},
            "paired_prompt_ids": [decision.prompt_id for decision in paired[ARMS[0]]],
        }, indent=2), encoding="utf-8")
        print(f"    {kept}/{len(entries)} prompts survived pairing")

        reports = []
        for arm in ARMS:
            checkpoint = out / "checkpoints" / arm / f"round-{round_index:03d}"
            # The one that was actually wrong: with no previous checkpoint the
            # old code did nothing, so round 0's second arm trained on top of
            # the first arm's update and every later round inherited it through
            # its own checkpoint.
            source = restore_arm_state(
                previous_checkpoint(out, arm, round_index),
                model=backbone.model, optimizer=optimizers[arm],
                scheduler=schedulers[arm], expected_config_digest=digest, base=base_state)
            print(f"    {arm}: training from {source}")
            report = train_arm(
                arm=arm,
                backbone=backbone,
                optimizer=optimizers[arm],
                scheduler=schedulers[arm],
                decisions=paired[arm],
                candidates=[c for pool in pools[arm].values() for c in pool],
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
            print(f"    {arm}: t2i {report['mean_t2i_loss']} "
                  f"grad {report['mean_gradient_norm_before_clip']:.3f} "
                  f"parameter delta {report['parameter_delta_l2']:.6g}")

        write_done(round_dir, {
            "round": round_index, "finished": now(),
            "prompts": len(entries), "paired": kept, "arms": reports,
            "initialization_seed": int(config["seed"]), "train_replay_examples": len(corpus.replay),
        })
    print(f"=== {now()} invocation complete: rounds {pending}; "
          f"{len(done) + len(pending)}/{training['rounds']} total ===")


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

    if args.round < -1:
        raise SystemExit("--round must be -1 (untrained baseline) or a nonnegative checkpoint")
    checkpoint = (out / "checkpoints" / "base" / "round--01" if args.round == -1 else
                  out / "checkpoints" / args.arm / f"round-{args.round:03d}")
    if args.round >= 0 and not checkpoint.is_dir():
        raise SystemExit(f"No checkpoint at {checkpoint}")

    seed_training(int(config["seed"]))
    backbone = Showo2Adapter(device=args.device, lazy=False)
    targets = json.loads(Path(LORA_TARGETS).read_text(encoding="utf-8"))
    lora = config["training"]["lora"]
    seed_training(int(config["seed"]))
    backbone.attach_lora(target_modules=targets["target_modules"], rank=int(lora["rank"]),
                         alpha=int(lora["alpha"]), dropout=float(lora["dropout"]),
                         gradient_checkpointing=False)
    if checkpoint.exists():
        load_checkpoint(checkpoint, model=backbone.model, optimizer=None, scheduler=None,
                        expected_config_digest=sha256_json(config))
    model_digest = parameter_digest(trainable_snapshot(backbone.model))

    step = (args.round + 1) * int(config["training"]["optimizer_steps_per_round"])
    eval_dir = out / "evaluations" / args.arm / f"step-{step:05d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = list(split["outcome"])
    seed_step = 0 if config.get("evaluation", {}).get("fixed_latents", False) else step
    seeds = {prompt_id: evaluation_seed(seed=int(config["seed"]), arm=args.arm,
                                        step=seed_step, prompt_id=prompt_id)
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

    selection_rows = self_selection_scores(
        backbone, specs=corpus.specs, images=images,
        output_path=eval_dir / "s_select.jsonl", metadata={
            "arm": args.arm, "round": args.round, "step": step,
            "config_digest": sha256_json(config), "parameter_digest": model_digest,
            "initialization_seed": int(config["seed"]),
            "evaluation_seed_step": seed_step,
            "score_policy": "fixed_question_denominator_v1"})
    selection_summary = summarize_selection(selection_rows)
    (eval_dir / "s_select.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
    scores = cycle_scores(backbone, specs=corpus.specs, images=images)
    mean, sem, count = summarize_cycle(scores)
    (eval_dir / "cycle.json").write_text(json.dumps({
        "arm": args.arm, "round": args.round, "step": step,
        "mean": mean, "sem": sem, "n": count, "scores": scores,
        "parameter_digest": model_digest, "initialization_seed": int(config["seed"]),
    }, indent=2), encoding="utf-8")
    print(f"{args.arm} step {step}: cycle {mean} +/- {sem} over {count} images; "
          f"s_select {selection_summary['mean']}, coverage {selection_summary['coverage']}")
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
        selection_path = eval_dir / "s_select.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        manifest_path = eval_dir / "manifest.jsonl"
        rate, n, unadjudicated = external_correctness(
            eval_dir / "verified.jsonl",
            manifest_path=manifest_path if manifest_path.exists() else None)
        row = CheckpointMetrics(
            arm=cycle["arm"], round_index=int(cycle["round"]), step=int(cycle["step"]),
            internal_cycle=cycle["mean"], internal_sem=cycle["sem"],
            internal_n=int(cycle["n"]),
            external_correct=rate, external_n=n, external_unadjudicated=unadjudicated,
            external_coverage_policy=("manifest_image_verdict_v1" if manifest_path.exists()
                                      else "legacy_row_only"),
            s_select=selection.get("mean"), s_select_sem=selection.get("sem"),
            s_select_n=int(selection.get("n", 0)),
            s_select_available=int(selection.get("available", 0)),
            s_select_total=int(selection.get("total", 0)),
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
              f"s_select {row.s_select}  external {external}")


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


def resolve_devices(args: argparse.Namespace) -> None:
    """`--ladder-device` falls back to `--device`. split/score/report have neither."""

    if not hasattr(args, "device"):
        return
    if getattr(args, "ladder_device", None) is None:
        args.ladder_device = args.device
    if args.ladder_device != args.device:
        print(f"backbone on {args.device}, ladder on {args.ladder_device}")


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
    t.add_argument("--device", default="cuda:0",
                   help="the card the backbone draws and trains on")
    t.add_argument("--ladder-device", default=None,
                   help="the card the detectors and the frozen observer run on; "
                        "defaults to --device. Splitting the two is what lets "
                        "generation and adjudication overlap instead of queueing, "
                        "and on this machine it is the difference between a wall "
                        "clock that is the slower of the two and one that is their "
                        "sum. Generation is compute-bound and tolerates the gen3 x4 "
                        "card; per-image detection does not (STATUS 32).")
    t.add_argument("--backend", default="qwen2vl")
    t.add_argument("--observer-python", default="envs/observer/python.exe",
                   help="the environment the detectors live in")
    t.add_argument("--core-python", default="envs/core/python.exe")
    t.add_argument("--max-epochs", type=int, default=1,
                   help="passes over the prompt bank; >1 mixes memorisation into the curve")
    t.add_argument("--max-rounds", type=int, default=None,
                   help="run at most this many unfinished rounds now; frozen total schedule is unchanged")
    t.set_defaults(func=stage_train)

    g = sub.add_parser("generate", help="outcome images + internal curve for one checkpoint")
    common(g)
    g.add_argument("--arm", choices=list(ARMS), required=True)
    g.add_argument("--round", type=int, required=True, help="-1 is the untrained step-0 baseline")
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(func=stage_generate)

    c = sub.add_parser("score", help="fold verified checkpoints into the metrics table")
    c.add_argument("--outdir", required=True, type=Path)
    c.set_defaults(func=stage_score)

    r = sub.add_parser("report", help="D*, D_g and the lead")
    r.add_argument("--outdir", required=True, type=Path)
    r.set_defaults(func=stage_report)

    args = parser.parse_args()
    resolve_devices(args)
    args.func(args)


if __name__ == "__main__":
    main()
