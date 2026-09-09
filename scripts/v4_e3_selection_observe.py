"""The blind/prompted selection pass endpoint 3 needs and the run never recorded.

Deviation 13.2. Endpoint 3 plots one point per (arm, checkpoint): x is the
context effect there -- how much better the backbone answers about the pixels
when it is *not* told what it was asked to draw, on the trials where the
description and the picture disagree -- and y is the selection gain, how much
better a loop does at keeping the right drawing when it scores candidates
blind. Neither axis exists in the run. The per-checkpoint evaluation records
one condition (prompted), one candidate (0) and one question set
(spec-derived); endpoint 3 needs both conditions, all four candidates, and the
detector's question set, which is the only one carrying a `gold_source` and
therefore the only one that can tell a conflict trial from an agreeing one.

    envs/showo2/python.exe scripts/v4_e3_selection_observe.py \
        --run runs/v4/decoupling-main-20260908 \
        --config configs/v4_decoupling_main_20260908.yaml --device cuda:0

This is `scripts/v4_run_pipeline.py`'s observe stage pointed at a checkpoint
instead of at a corpus run: same `build_questions`, same `grade`, same
`has_unnameable` skip, same row shape, both conditions in one file and told
apart by `condition`. The corpus runs it mirrors used the untrained backbone,
which is why that stage never had to load an adapter and this does.

Costed at 9.4 min per arm-checkpoint from measured latencies: 3.8 h for the
main run and 18.7 h across the five replicates
(`review-packets/e3-data-availability-20260909/` and its
`CORRECTION-checkpoint-count.md`). Deviation 13.2 point 7: if it does not
finish, endpoint 3 reads *not done*, and the corpus replay is not allowed to
stand in for per-checkpoint points.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.endpoint3 import MIN_CANDIDATES
from selfsight.utils.hashing import sha256_json
from selfsight.v4.checkpoint_reload import checkpoint_for, load_trained_backbone
from selfsight.v4.observe import PROMPTED_PREAMBLE
from selfsight.v4.questions import build_questions, grade, to_atomic
from selfsight.v4.spec import SceneSpec, has_unnameable

# bsv_selection.py's two condition names, which deviation 13.2 pins the whole
# analysis to. `selfsight.analysis.endpoint3` refuses any other label, and a
# test holds the two lists together.
CONDITIONS = ("blind", "prompted")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def steps_present(run: Path, arm: str) -> list[int]:
    """Checkpoints with an adjudicated outcome set: both inputs this pass needs."""

    directory = Path(run) / "evaluations" / arm
    if not directory.is_dir():
        return []
    return sorted(int(child.name.split("-")[1]) for child in directory.glob("step-*")
                  if (child / "verified.jsonl").exists()
                  and (child / "manifest.jsonl").exists())


def plan_step(run: Path, arm: str, step: int) -> tuple[list[dict], dict[str, int]]:
    """Every image that supplies trials, with its questions. Touches no GPU.

    `has_unnameable` is `v4_run_pipeline.stage_observe`'s rule and not a new
    one: an image containing something nobody could name has a verdict -- a
    miss -- but no answerable questions, because "how many pears" has no answer
    when a candidate is half a pear. Those images count towards external
    correctness and supply no trials, so they leave the selection pool rather
    than entering it with a made-up score.
    """

    eval_dir = Path(run) / "evaluations" / arm / f"step-{step:05d}"
    verified = {row["image_path"]: row for row in _rows(eval_dir / "verified.jsonl")}
    work: list[dict] = []
    counts = Counter()
    every: dict[str, list[bool]] = defaultdict(list)
    kept: dict[str, list[bool]] = defaultdict(list)
    for row in _rows(eval_dir / "manifest.jsonl"):
        counts["images"] += 1
        image = row["image_path"]
        if image in verified:
            every[row["spec_id"]].append(bool(verified[image]["image_correct"]))
        if image not in verified:
            # Deviation 12's skipped row seen from the other side: the detector
            # found nothing, so there is no verdict to select on.
            counts["unverified"] += 1
            continue
        settled = verified[image]["detections"]
        if has_unnameable(settled):
            counts["unnameable"] += 1
            continue
        questions = build_questions(SceneSpec.from_dict(row["spec"]), settled,
                                    seed=row["seed"])
        if not questions:
            counts["no_questions"] += 1
            continue
        work.append({"row": row, "verdict": verified[image], "questions": questions})
        kept[row["spec_id"]].append(bool(verified[image]["image_correct"]))
        counts["trials"] += len(questions)
    counts.update(_selectability(every, kept))
    return work, dict(counts)


def _selectability(every: dict[str, list[bool]], kept: dict[str, list[bool]]) -> dict:
    """How much of the selection task the unaskable images take away.

    An image that supports no question leaves the pool, and on the main run
    every such image is externally incorrect -- `build_counting` returns None
    when none of the requested categories was detected at all, which is also
    the definition of a miss. So the exclusion removes candidates that are all
    wrong, which raises the ceiling a selection loop is scored against and, in
    the worst case, drops a prompt below two candidates entirely.

    It cancels between the conditions by construction, because the question set
    depends on the image and not on the condition, and endpoint 3's y is a
    difference within one pool. What it does not cancel is the pool
    composition, so the size is counted per checkpoint rather than asserted to
    be small: on the main run's first three checkpoints it costs 0 to 3 prompts
    of 64 and moves the ceiling by at most +0.033.
    """

    surviving = {spec_id: values for spec_id, values in kept.items()
                 if len(values) >= MIN_CANDIDATES}
    return {"prompts": len(every),
            "pools_kept": len(surviving),
            "pools_kept_with_correct": sum(any(values) for values in surviving.values()),
            "prompts_with_correct": sum(any(values) for values in every.values())}


def resume_state(path: Path, expected: dict[str, int]) -> tuple[set, list[dict] | None]:
    """Which (image, condition) pairs are finished, and what to keep if not all are.

    A pair is several lines, so a process killed mid-pair leaves a partial one,
    and appending to it would give that candidate a score built from some of
    its questions twice. Completeness is therefore a count against the plan,
    not the mere presence of a row, and partial pairs are dropped and redone.
    The second element is `None` when nothing needs dropping.
    """

    if not path.exists():
        return set(), None
    rows = _rows(path)
    written = Counter((row["image_path"], row["condition"]) for row in rows)
    stray = sorted({image for image, _ in written} - set(expected))
    if stray:
        raise SystemExit(f"{path} holds {len(stray)} image(s) this checkpoint does not "
                         f"plan, e.g. {stray[0]}. Wrong run or wrong --outdir; refusing "
                         f"to mix two measurements in one file.")
    complete = {key for key, count in written.items() if count == expected[key[0]]}
    if len(complete) == len(written):
        return complete, None
    return complete, [row for row in rows
                      if (row["image_path"], row["condition"]) in complete]


def observe_step(backbone, run: Path, arm: str, step: int, out_dir: Path,
                 *, adapter_digest: str) -> dict:
    """One arm-checkpoint, both conditions, resumable per (image, condition)."""

    work, counts = plan_step(run, arm, step)
    expected = {item["row"]["image_path"]: len(item["questions"]) for item in work}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step-{step:05d}.jsonl"
    done, keep = resume_state(path, expected)
    dropped = 0
    if keep is not None:
        dropped = len(_rows(path)) - len(keep)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in keep), encoding="utf-8")

    started = time.time()
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for item in work:
            row, verdict, questions = item["row"], item["verdict"], item["questions"]
            image = row["image_path"]
            spec = SceneSpec.from_dict(row["spec"])
            atoms = [to_atomic(question) for question in questions]
            for condition in CONDITIONS:
                if (image, condition) in done:
                    continue
                asked = atoms if condition == "blind" else [
                    replace(atom, text=PROMPTED_PREAMBLE.format(
                        prompt=spec.prompt, question=atom.text))
                    for atom in atoms]
                observation = backbone.observe_atoms(image, asked)
                lines = []
                for question, answer in zip(questions, observation.answers):
                    correct = grade(answer.raw_answer, question)
                    lines.append(json.dumps({
                        "condition": condition,
                        "spec_id": spec.spec_id,
                        "image_path": image,
                        "candidate_index": row["candidate_index"],
                        "arm": arm, "step": step, "adapter_digest": adapter_digest,
                        "family": question.family.value,
                        "question": question.prompt_text,
                        "gold": question.gold,
                        "gold_source": question.gold_source,
                        # The builders' bookkeeping travels with the trial, as
                        # it does in stage_observe: the analysis splits the
                        # existence family on it.
                        "metadata": dict(question.metadata),
                        "option_a": question.option_a,
                        "option_b": question.option_b,
                        "raw_answer": answer.raw_answer,
                        "correct": correct,
                        "abstain": correct is None,
                        "image_correct": verdict["image_correct"],
                        "resolution": verdict["resolution"],
                    }, ensure_ascii=False) + "\n")
                # One write per (image, condition), so the unit that resume
                # counts is the unit that reaches the file.
                handle.write("".join(lines))
                handle.flush()
                written += len(lines)
    elapsed = time.time() - started
    summary = {"arm": arm, "step": step, "answers": written, "resumed_pairs": len(done),
               "dropped_partial_rows": dropped, "seconds": elapsed,
               "adapter_digest": adapter_digest, **counts}
    print(f"  {arm} step {step:>5}: {written} answers, {len(done)} pairs resumed, "
          f"{dropped} partial rows redone, {counts.get('unnameable', 0)} unnameable, "
          f"{counts.get('unverified', 0)} unverified, {elapsed / 60:.1f} min", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["naive", "rfo_gold"])
    parser.add_argument("--steps", type=int, nargs="*",
                        help="Default: every step adjudicated in every arm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", type=Path, help="Default: <run>/analysis/selection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and plan everything, touch no GPU")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    steps_per_round = int(config["training"]["optimizer_steps_per_round"])
    out_root = args.outdir or args.run / "analysis" / "selection"

    per_arm = {arm: steps_present(args.run, arm) for arm in args.arms}
    steps = sorted(args.steps) if args.steps else sorted(
        set.intersection(*(set(values) for values in per_arm.values())))
    if not steps:
        raise SystemExit(f"No step is adjudicated in every arm: {per_arm}")

    print(f"run          {args.run}")
    print(f"config       {args.config} (digest {sha256_json(config)[:12]})")
    print(f"arms         {args.arms}")
    print(f"steps        {steps}")
    print(f"outdir       {out_root}")
    print()

    plan = []
    for arm in args.arms:
        for step in steps:
            checkpoint = checkpoint_for(args.run, arm, step, steps_per_round)
            if not checkpoint.is_dir():
                raise SystemExit(f"No checkpoint for {arm} at step {step}: {checkpoint}")
            work, counts = plan_step(args.run, arm, step)
            expected = {item["row"]["image_path"]: len(item["questions"]) for item in work}
            done, _ = resume_state(out_root / arm / f"step-{step:05d}.jsonl", expected)
            plan.append((arm, step, checkpoint, len(done), len(work) * len(CONDITIONS),
                         counts))

    for arm, step, checkpoint, done, total, counts in plan:
        print(f"  {arm:<10} step {step:>5}  {checkpoint.relative_to(args.run)}  "
              f"{done}/{total} image-conditions done, {counts.get('trials', 0)} trials, "
              f"{counts.get('unnameable', 0)} unnameable, "
              f"{counts.get('no_questions', 0)} unaskable, "
              f"{counts.get('unverified', 0)} unverified")
    if args.dry_run:
        print()
        print("dry run: nothing loaded, nothing written")
        return 0

    summary = []
    for arm, step, checkpoint, _, _, _ in plan:
        print(f"loading {checkpoint.relative_to(args.run)} on {args.device}", flush=True)
        backbone, adapter_digest = load_trained_backbone(config, checkpoint,
                                                         device=args.device)
        summary.append(observe_step(backbone, args.run, arm, step, out_root / arm,
                                    adapter_digest=adapter_digest))
        del backbone

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps({
        "conditions": list(CONDITIONS),
        "registered_by": "deviation 13.2",
        "run": str(args.run), "config": str(args.config),
        "config_digest": sha256_json(config),
        "device": args.device, "arms": args.arms, "steps": steps,
        "questions": "selfsight.v4.questions.build_questions, the detector caliber",
        "candidates": "every candidate in the checkpoint's manifest",
        "checkpoints": summary,
    }, indent=2) + "\n", encoding="utf-8")
    total = sum(entry["seconds"] for entry in summary)
    print()
    print(f"wrote {manifest} -- {len(summary)} arm-checkpoints in {total / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
