"""The `image_only` pass endpoint 2 needs and the main run never recorded.

Deviation 11.3. Endpoint 2 is a *blind* discrimination gap: how far the
backbone's self-score separates the pictures that turned out externally right
from the ones that turned out wrong, when it is not being told what it was
asked to draw. The run records `s_select` for every arm in the prompted
condition only -- `self_selection_scores` calls `observe_naive`, which wraps
every question in `PROMPTED_PREAMBLE`, and it does so for the RFO arm too,
because the RFO arm's blindness lives in the *detector*, not in the
self-report. So the primary endpoint cannot be computed from what is on disk,
and this script measures it offline instead.

    envs/showo2/python.exe scripts/v4_e3_blind_observe.py \
        --run runs/v4/decoupling-main-20260908 \
        --config configs/v4_decoupling_main_20260908.yaml --device cuda:0

Exactly one thing differs from the prompted pass: the questions go to the
backbone bare. Same checkpoint, same adapter, same image, same questions in
the same order with the same `choice_order_seed`, same scorer. The questions
are read back out of `s_select.jsonl` rather than regenerated, so "the same
questions" is a fact about the bytes rather than a claim about two code paths
agreeing.

Candidate 0 only, per deviation 11.1: `internal_curve_scope` is
`first_draw_only`, so candidate 0 is the only draw with a self-report to be
blind about, and it is the draw whose external verdict endpoint 2 pairs with.

Costed at 1.3 min per arm-checkpoint from measured latencies
(`review-packets/e3-data-availability-20260909/`): 0.6 h for the main run,
2.8 h across the five replicates. If it does not finish, deviation 11.3 is
explicit about the consequence -- endpoint 2 reads *not done*, and the
prompted gap is never substituted for it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfsight.schemas import AtomicQuestion
from selfsight.utils.hashing import sha256_json
from selfsight.v4.observe import PROMPTED_PREAMBLE

# scripts/v4_train.py's constant, restated. Importing v4_train would drag in
# the training stack and its side effects for a read-only pass.
LORA_TARGETS = ROOT / "runs/readiness/showo2-1p5b/a4-lora-targets-r1.json"

CONDITION = "image_only"

# The first sentence of the preamble, enough to recognise it and short enough
# not to be a second copy of the string. `tests/test_v4_probe.py` already pins
# the copies that exist; this is a guard, not a fourth copy, and the assert
# below is what stops it becoming a stale one.
PREAMBLE_MARKER = "You were asked to draw a picture"
assert PREAMBLE_MARKER in PROMPTED_PREAMBLE, "the marker no longer matches the preamble"


def checkpoint_for(run: Path, arm: str, step: int, steps_per_round: int) -> Path:
    """Step 0 is the untrained base; step (i+1)*k is arm `arm`'s round i.

    Both arms share `checkpoints/base/round--01` at step 0, which is not a
    mistake to be tidied away: the two arms are the same model until the first
    optimizer step, and the blind pass measures each of them there anyway
    because the *images* differ from step 0 onwards.
    """

    if step == 0:
        return run / "checkpoints" / "base" / "round--01"
    if step % steps_per_round:
        raise ValueError(f"step {step} is not a multiple of {steps_per_round}")
    return run / "checkpoints" / arm / f"round-{step // steps_per_round - 1:03d}"


def prompted_rows(run: Path, arm: str, step: int) -> list[dict]:
    """The prompted pass's own rows: images, questions and metadata."""

    path = run / "evaluations" / arm / f"step-{step:05d}" / "s_select.jsonl"
    if not path.exists():
        raise SystemExit(f"No prompted pass to mirror at {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def bare_questions(row: dict) -> tuple[AtomicQuestion, ...]:
    """The saved questions, checked to be the ones without the preamble.

    A `s_select.jsonl` written by some future pass that saved the wrapped text
    would make this script measure the prompted condition under the blind
    condition's name -- the single worst thing it could do, and the reason the
    check is here rather than in a comment.
    """

    questions = tuple(AtomicQuestion(**question) for question in row["questions"])
    leaked = [question.question_id for question in questions
              if PREAMBLE_MARKER in question.text]
    if leaked:
        raise SystemExit(f"{row['prompt_id']}: saved question text already carries the "
                         f"prompted preamble ({leaked[0]}); this pass would not be blind")
    return questions


def steps_present(run: Path, arm: str) -> list[int]:
    directory = run / "evaluations" / arm
    if not directory.is_dir():
        return []
    return sorted(int(child.name.split("-")[1]) for child in directory.glob("step-*")
                  if (child / "s_select.jsonl").exists())


def observe_step(backbone, run: Path, arm: str, step: int, out_dir: Path,
                 *, model_id: str, adapter_digest: str) -> dict:
    """One arm-checkpoint, resumable, one line flushed per image."""

    from selfsight.v4.evaluate import fixed_atomic_score

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step-{step:05d}.jsonl"
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["prompt_id"])
    rows = prompted_rows(run, arm, step)
    started = time.time()
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["prompt_id"] in done:
                continue
            questions = bare_questions(row)
            observation = backbone.observe_atoms(row["image_path"], questions)
            if observation.rgb_sha256 != row["rgb_sha256"]:
                raise SystemExit(f"{row['prompt_id']}: the image on disk is not the one the "
                                 f"prompted pass scored")
            record = {
                "prompt_id": row["prompt_id"],
                # Deviation 11.1. Constant rather than read off the row because
                # s_select.jsonl has no candidate field: it is first-draw-only
                # by construction, and writing the 0 makes the reader able to
                # check what the writer was assuming.
                "candidate_index": 0,
                "condition": CONDITION,
                "image_path": row["image_path"],
                "rgb_sha256": row["rgb_sha256"],
                "question_digest": row["question_digest"],
                "arm": arm, "step": step,
                "checkpoint_model_id": model_id,
                "adapter_digest": adapter_digest,
                "prompted_s_select": row["s_select"],
                "observation": observation.to_dict(),
                **fixed_atomic_score(observation, questions),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1
    elapsed = time.time() - started
    print(f"  {arm} step {step:>5}: {written} new, {len(done)} resumed, {elapsed / 60:.1f} min",
          flush=True)
    return {"arm": arm, "step": step, "written": written, "resumed": len(done),
            "images": len(rows), "seconds": elapsed, "adapter_digest": adapter_digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["naive", "rfo_gold"])
    parser.add_argument("--steps", type=int, nargs="*",
                        help="Default: every step with a prompted pass in every arm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", type=Path,
                        help="Default: <run>/analysis/blind_observe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve everything and touch no GPU. Prints the plan, the "
                             "checkpoint each step maps to, and what is already done.")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    steps_per_round = int(config["training"]["optimizer_steps_per_round"])
    out_root = args.outdir or args.run / "analysis" / "blind_observe"

    per_arm = {arm: steps_present(args.run, arm) for arm in args.arms}
    if args.steps:
        steps = sorted(args.steps)
    else:
        # Deviation 13.1 point 6 wants the fit to run on checkpoints both arms
        # finished. Taking the union here and letting the analysis discover the
        # hole later would spend GPU time on a checkpoint the fit cannot use.
        common = set.intersection(*(set(values) for values in per_arm.values()))
        steps = sorted(common)
    if not steps:
        raise SystemExit(f"No step has a prompted pass in every arm: {per_arm}")

    print(f"run          {args.run}")
    print(f"config       {args.config} (digest {sha256_json(config)[:12]})")
    print(f"arms         {args.arms}")
    print(f"steps        {steps}")
    for arm in args.arms:
        missing = sorted(set(steps) - set(per_arm[arm]))
        if missing:
            print(f"  {arm}: no prompted pass at {missing}")
    print(f"outdir       {out_root}")
    print()

    plan = []
    for arm in args.arms:
        for step in steps:
            checkpoint = checkpoint_for(args.run, arm, step, steps_per_round)
            if not checkpoint.is_dir():
                raise SystemExit(f"No checkpoint for {arm} at step {step}: {checkpoint}")
            path = out_root / arm / f"step-{step:05d}.jsonl"
            existing = 0
            if path.exists():
                existing = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                               if line.strip())
            plan.append((arm, step, checkpoint, existing, len(prompted_rows(args.run, arm, step))))

    for arm, step, checkpoint, existing, total in plan:
        print(f"  {arm:<10} step {step:>5}  {checkpoint.relative_to(args.run)}  "
              f"{existing}/{total} done")
    if args.dry_run:
        print()
        print("dry run: nothing loaded, nothing written")
        return 0

    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.training.checkpoint import load_checkpoint
    from selfsight.v4.train import parameter_digest, seed_training, trainable_snapshot

    targets = json.loads(LORA_TARGETS.read_text(encoding="utf-8"))
    lora = config["training"]["lora"]
    digest = sha256_json(config)
    summary = []
    # One adapter load per arm-checkpoint: 26 for the main run, and the
    # availability audit costed the pass with that number in it. Step 0 is the
    # one place both arms share a checkpoint, so scoring them under a single
    # load would save exactly one of the 26 and complicate the resume logic
    # that makes this restartable.
    for arm, step, checkpoint, _, _ in plan:
        print(f"loading {checkpoint.relative_to(args.run)} on {args.device}", flush=True)
        # Both seeds are the training loop's, in the training loop's order:
        # once before the adapter is constructed and once immediately before
        # attach_lora, which randomises LoRA A. A blind pass whose adapter was
        # initialised differently would be measuring a different model.
        seed_training(int(config["seed"]))
        backbone = Showo2Adapter(device=args.device, lazy=False)
        seed_training(int(config["seed"]))
        backbone.attach_lora(target_modules=targets["target_modules"], rank=int(lora["rank"]),
                             alpha=int(lora["alpha"]), dropout=float(lora["dropout"]),
                             gradient_checkpointing=False)
        load_checkpoint(checkpoint, model=backbone.model, optimizer=None, scheduler=None,
                        expected_config_digest=digest)
        adapter_digest = parameter_digest(trainable_snapshot(backbone.model))
        summary.append(observe_step(backbone, args.run, arm, step, out_root / arm,
                                    model_id=backbone.model_id, adapter_digest=adapter_digest))
        del backbone

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps({
        "condition": CONDITION,
        "registered_by": "deviation 11.3",
        "run": str(args.run), "config": str(args.config), "config_digest": digest,
        "device": args.device, "arms": args.arms, "steps": steps,
        "candidate_index": 0, "candidate_rule": "deviation 11.1: first_draw_only",
        "questions": "read verbatim from s_select.jsonl; no preamble applied",
        "checkpoints": summary,
    }, indent=2) + "\n", encoding="utf-8")
    total = sum(entry["seconds"] for entry in summary)
    print()
    print(f"wrote {manifest} -- {len(summary)} arm-checkpoints in {total / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
