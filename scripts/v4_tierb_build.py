"""Build and run Tier B: matched pairs that differ in one atomic fact.

    plan       pick the edits, write the edited PNGs, emit plan.jsonl
    check      re-detect the recoloured images and accept or reject each edit
    questions  emit the 2AFC trials, gold from the edit
    observe    ask the generator, one condition at a time
    analyze    accuracy per member, and the paired flip rate

Stages are separate processes because they need different environments and
different amounts of GPU, and because `check` is the one that can reject work:
running it inside `plan` would mean regenerating images to re-run a gate.

Sources are only images the main pipeline verified as matching their spec and
that are not awaiting human adjudication. Editing an image whose contents are
already in doubt would give a pair whose "before" label is a guess and whose
"after" label inherits the guess.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from selfsight.v4.questions import Family, ForcedChoice, grade, to_atomic
from selfsight.v4.spec import SceneSpec, canonical_noun, countable
from selfsight.v4.tasks import PLAUSIBLE_COLORS
from selfsight.v4.tierb import (
    MAX_DELETE_SHARE,
    MIN_DELETE_CENTRE,
    ACHROMATIC,
    DeletePlan,
    delete,
    delete_accepted,
    delete_confirmed,
    delete_question,
    deletion_mask,
    sham_box,
    COLOUR_HUES,
    MAX_MASK_SHARE,
    MIN_CENTRE_COVERAGE,
    MIN_MASK_SHARE,
    RecolourPlan,
    edit_accepted,
    edit_confirmed,
    flip,
    flip_question,
    recolour,
    recolour_question,
    target_colours,
)

PROMPTED_PREAMBLE = """You were asked to draw a picture from this description:
"{prompt}"

Here is the picture you drew. Answer about what is actually in the picture.

{question}"""
"""Identical to the main pipeline's, deliberately. The two-condition contrast is
only comparable across Tier A and Tier B if the wording is the same string."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _plausible(noun: str) -> set[str] | None:
    allowed = PLAUSIBLE_COLORS.get(noun) or PLAUSIBLE_COLORS.get(noun.rstrip("s"))
    return set(allowed) if allowed else None


def _sources(runs: list[Path]) -> list[dict[str, Any]]:
    """Verified-correct, not-pending images with their detections and spec."""
    out = []
    for run in runs:
        manifest = {r["image_path"]: r for r in read_jsonl(run / "manifest.jsonl")}
        for row in read_jsonl(run / "verified.jsonl"):
            if not row["image_correct"] or row["resolution"] == "pending_human":
                continue
            entry = manifest.get(row["image_path"])
            if entry is None:
                continue
            out.append({
                "run": str(run),
                "image_path": row["image_path"],
                "spec": entry["spec"],
                "seed": entry["seed"],
                "detections": countable(row["detections"]),
            })
    return out


# -------------------------------------------------------------------- plan


def stage_plan(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sources = _sources([Path(run) for run in args.run])
    print(f"{len(sources)} verified-correct source images")

    rows: list[dict[str, Any]] = []
    skipped: collections.Counter = collections.Counter()

    for source in sources:
        image_path = source["image_path"]
        stem = Path(image_path).stem
        detections = source["detections"]
        image = np.asarray(Image.open(image_path).convert("RGB"))

        if "flip" not in args.edits and "recolour" not in args.edits:
            continue

        # ---- flip: needs two categories at distinguishable x positions
        positions: dict[str, list[float]] = collections.defaultdict(list)
        for item in detections:
            centre = item.get("center")
            if centre:
                positions[canonical_noun(item["object"])].append(float(centre[0]))
        singles = {k: v[0] for k, v in positions.items() if len(v) == 1}
        if "flip" in args.edits and len(singles) >= 2:
            first, second = rng.sample(sorted(singles), 2)
            # The same 24px floor the natural-pool spatial builder uses. A near
            # tie is not a fact about the image, and mirroring it is not either.
            if abs(singles[first] - singles[second]) >= 24.0:
                edited_path = out_dir / "images" / f"{stem}.flip.png"
                Image.fromarray(flip(image)).save(edited_path)
                rows.append({
                    "pair_id": f"{stem}:flip",
                    "edit": "flip",
                    "original_path": image_path,
                    "edited_path": str(edited_path),
                    "spec": source["spec"],
                    "subject": first,
                    "other": second,
                    "subject_left": singles[first] < singles[second],
                    "needs_check": False,
                })
            else:
                skipped["flip_near_tie"] += 1
        else:
            skipped["flip_needs_two_singletons"] += 1

        # ---- recolour: one whole category per image
        # Every instance of the noun is recoloured together. The atom this
        # experiment is built on is (object, colour) at the category level --
        # that is what the spec states and what the gold rule compares -- so
        # "the mugs are blue" becoming "the mugs are red" is one fact changed.
        # Recolouring one of two mugs instead would leave a question with no
        # answer, which is the case the natural-pool binding builder skips.
        by_noun: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in detections:
            by_noun[canonical_noun(item["object"])].append(item)

        if "recolour" not in args.edits:
            continue

        candidates = []
        for noun, items in sorted(by_noun.items()):
            colours = {str(d.get("color", "")).strip().lower() for d in items}
            if len(colours) != 1:
                skipped["recolour_category_has_two_colours"] += 1
                continue
            colour = colours.pop()
            if colour not in COLOUR_HUES or any(not d.get("bbox") for d in items):
                continue
            others = [str(d.get("color", "")).strip().lower()
                      for d in detections
                      if canonical_noun(d["object"]) != noun]
            targets = target_colours(noun, colour, others, _plausible(noun))
            if targets:
                candidates.append((noun, items, colour, targets))
        if not candidates:
            skipped["recolour_no_candidate"] += 1
            continue
        # Candidates are tried in random order and the first that survives the
        # mask tests is used. Drawing one and giving up on the image if it fails
        # threw away a third of the supply: whether a mask can be built is a
        # property of that object against that background, not of the image.
        rng.shuffle(candidates)
        chosen = None
        for noun, items, colour, targets in candidates:
            target = rng.choice(targets)
            attempt, shares, centres, failed = image, [], [], None
            for item in items:
                attempt, share, centre = recolour(
                    attempt, item["bbox"], colour, target)
                shares.append(share)
                centres.append(centre)
                if not MIN_MASK_SHARE <= share <= MAX_MASK_SHARE:
                    failed = "recolour_mask_share"
                elif centre < MIN_CENTRE_COVERAGE:
                    failed = "recolour_mask_missed_the_object"
            # All or nothing: a category half recoloured is worse than none,
            # because the question about it then has two right answers.
            if failed:
                skipped[failed] += 1
                continue
            chosen = (noun, items, colour, target, attempt, shares, centres)
            break
        if chosen is None:
            continue
        noun, items, colour, target, edited, shares, centres = chosen
        edited_path = out_dir / "images" / f"{stem}.recolour.png"
        Image.fromarray(edited).save(edited_path)
        rows.append({
            "pair_id": f"{stem}:recolour",
            "edit": "recolour",
            "original_path": image_path,
            "edited_path": str(edited_path),
            "spec": source["spec"],
            "noun": noun,
            "bbox": [list(d["bbox"]) for d in items],
            "source_colour": colour,
            "target_colour": target,
            "colour_constrained": _plausible(noun) is not None,
            "count": len(items),
            "mask_share": [round(v, 4) for v in shares],
            "centre_coverage": [round(v, 4) for v in centres],
            "detections_before": detections,
            "needs_check": True,
        })

    if "delete" in args.edits:
        _plan_deletions(args, out_dir, sources, rng, rows, skipped)

    write_jsonl(out_dir / "plan.jsonl", rows)
    kinds = collections.Counter(r["edit"] for r in rows)
    print(f"planned {len(rows)} pairs: {dict(kinds)}")
    print(f"skipped: {dict(skipped)}")
    print(f"wrote {out_dir / 'plan.jsonl'}")


def _plan_deletions(args, out_dir, sources, rng, rows, skipped) -> None:
    """One deletion and one matched sham per image, where one will build.

    Only nouns the scene holds exactly one of. Deleting one of two apples
    changes a count, which is a different family and a weaker contrast -- the
    picture still holds an apple, so a model reciting the prompt is not
    contradicted by the object being gone. Deleting the only apple is the atom
    this arm is for.

    The sham is planned from the same image and given the same area, so the two
    arms differ in what the edit means and not in how much of the image went
    through the filler.
    """
    from selfsight.v4.inpaint import LamaInpainter

    filler = LamaInpainter()
    started = time.time()
    planned = 0
    for number, source in enumerate(sources):
        image_path = source["image_path"]
        stem = Path(image_path).stem
        detections = source["detections"]
        by_noun: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in detections:
            by_noun[canonical_noun(item["object"])].append(item)

        candidates = []
        for noun, items in sorted(by_noun.items()):
            if len(items) != 1 or not items[0].get("bbox"):
                continue
            colour = str(items[0].get("color", "")).strip().lower()
            if colour not in COLOUR_HUES and colour not in ACHROMATIC:
                continue
            candidates.append((noun, items[0], colour))
        if not candidates:
            skipped["delete_no_singleton_object"] += 1
            continue

        image = np.asarray(Image.open(image_path).convert("RGB"))
        rng.shuffle(candidates)
        chosen = None
        for noun, item, colour in candidates:
            others = [(o["bbox"], str(o.get("color", "")).strip().lower())
                      for o in detections if o is not item and o.get("bbox")]
            mask, centre = deletion_mask(image, item["bbox"], colour, others)
            if centre < MIN_DELETE_CENTRE:
                skipped["delete_neighbour_sits_in_the_target"] += 1
                continue
            if float(mask.mean()) > MAX_DELETE_SHARE:
                skipped["delete_hole_too_large_to_fill"] += 1
                continue
            edited, mask, centre = delete(image, item["bbox"], colour,
                                          filler, others)
            chosen = (noun, item, colour, edited, mask, centre)
            break
        if chosen is None:
            continue
        noun, item, colour, edited, mask, centre = chosen
        edited_path = out_dir / "images" / f"{stem}.delete.png"
        Image.fromarray(edited).save(edited_path)
        rows.append({
            "pair_id": f"{stem}:delete",
            "edit": "delete",
            "original_path": image_path,
            "edited_path": str(edited_path),
            "spec": source["spec"],
            "noun": noun,
            "colour": colour,
            "bbox": [list(item["bbox"])],
            "hole_share": round(float(mask.mean()), 4),
            "centre_coverage": round(centre, 4),
            "detections_before": detections,
            "needs_check": True,
        })
        planned += 1

        # ---- the sham: same filler, same area, no object touched
        window = sham_box(image.shape[:2],
                          [d["bbox"] for d in detections if d.get("bbox")],
                          int(mask.sum()),
                          random.Random(f"{args.seed}:{stem}:sham"))
        if window is None:
            skipped["sham_no_clear_background"] += 1
        else:
            sx0, sy0, sx1, sy1 = window
            sham_mask = np.zeros(image.shape[:2], dtype=bool)
            sham_mask[sy0:sy1, sx0:sx1] = True
            filled = np.where(sham_mask[..., None], filler(image, sham_mask),
                              image).astype(np.uint8)
            sham_path = out_dir / "images" / f"{stem}.sham.png"
            Image.fromarray(filled).save(sham_path)
            rows.append({
                "pair_id": f"{stem}:sham",
                "edit": "sham",
                "original_path": image_path,
                "edited_path": str(sham_path),
                "spec": source["spec"],
                "noun": noun,
                "colour": colour,
                "bbox": [list(item["bbox"])],
                "sham_box": [sx0, sy0, sx1, sy1],
                "hole_share": round(float(sham_mask.mean()), 4),
                "detections_before": detections,
                "needs_check": True,
            })
        if number % 20 == 0:
            print(f"delete {number + 1}/{len(sources)} planned={planned} "
                  f"{(time.time() - started) / max(1, number + 1):.1f}s/img",
                  flush=True)


# ------------------------------------------------------------------- check


def stage_check(args: argparse.Namespace) -> None:
    """Both detectors read the edited image; only exact single-change is kept.

    Flips are not checked. A mirror is exact -- there is no synthesis to verify
    and no way for it to have changed something else -- so sending them through
    a detector would spend GPU to re-derive a fact that is true by construction,
    and would let a detector error reject a correct pair.
    """
    from selfsight.v4.detectors import load

    out_dir = Path(args.outdir)
    plan = read_jsonl(out_dir / "plan.jsonl")
    todo = [row for row in plan if row["needs_check"]]
    out = out_dir / "checks.jsonl"
    # Keyed by detector as well as pair: both detectors append to this one file,
    # so resuming on pair alone let the second detector see the first one's rows
    # and skip every image, which then failed the gate as "not checked by both"
    # and rejected all 80 recolours at a yield of exactly zero.
    done: set[tuple[str, str]] = set()
    # `--overwrite` drops this detector's rows and keeps the other's. It used to
    # truncate the file, which silently destroyed a finished 206-image pass by
    # the first detector when the second one was started with the same flag.
    if out.exists() and args.overwrite:
        kept = [r for r in read_jsonl(out) if r["detector"] != args.detector]
        write_jsonl(out, kept)
        print(f"overwriting {args.detector}, keeping {len(kept)} rows from the other")
    if out.exists():
        done = {(r["pair_id"], r["detector"]) for r in read_jsonl(out)}
        print(f"resuming: {len(done)} pair-detector rows already checked")
    todo = [row for row in todo
            if (row["pair_id"], args.detector) not in done]
    print(f"{len(todo)} edited images to re-detect with {args.detector}")
    if not todo:
        return

    detector = load(args.detector, device=args.device)
    started = time.time()
    with out.open("a" if done else "w", encoding="utf-8") as handle:
        for index, row in enumerate(todo):
            record = {"pair_id": row["pair_id"], "detector": args.detector}
            try:
                record["detections"] = detector.detect(row["edited_path"])
            except (ValueError, OSError, RuntimeError) as exc:
                record["error"] = str(exc)[:300]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index + 1}/{len(todo)} "
                      f"{(time.time() - started) / max(1, index + 1):.1f}s/img",
                      flush=True)
    print(f"wrote {out}")


def stage_accept(args: argparse.Namespace) -> None:
    """Apply the gate. Both detectors must see the single intended change."""
    out_dir = Path(args.outdir)
    plan = {row["pair_id"]: row for row in read_jsonl(out_dir / "plan.jsonl")}
    checks: dict[str, dict[str, list]] = collections.defaultdict(dict)
    for row in read_jsonl(out_dir / "checks.jsonl"):
        if "detections" in row:
            checks[row["pair_id"]][row["detector"]] = row["detections"]

    accepted: list[dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    for pair_id, row in plan.items():
        if not row["needs_check"]:
            accepted.append(row)
            reasons["flip_no_check_needed"] += 1
            continue
        seen = checks.get(pair_id, {})
        if len(seen) < 2:
            reasons["not_checked_by_both"] += 1
            continue
        if row["edit"] in {"delete", "sham"}:
            plan_obj = DeletePlan(
                row["original_path"], row["noun"], row["colour"],
                tuple(row["bbox"][0]), row["edit"] == "sham")
            verdicts = [delete_accepted(plan_obj, row["detections_before"], after)
                        for after in seen.values()]
            strict = all(ok for ok, _ in verdicts)
            ladder = (all(delete_confirmed(plan_obj, after)
                          for after in seen.values())
                      and any(ok for ok, _ in verdicts))
            if strict or (args.gate == "ladder" and ladder):
                accepted.append(dict(row, gate="strict" if strict else "ladder"))
                reasons[row["edit"] + ("_accepted" if strict
                                       else "_accepted_on_the_edited_fact")] += 1
            else:
                reasons[row["edit"] + "_"
                        + next(why for ok, why in verdicts if not ok)] += 1
            continue
        plan_obj = RecolourPlan(
            row["original_path"], row["noun"], tuple(row["bbox"][0]),
            row["source_colour"], row["target_colour"], row["colour_constrained"],
        )
        verdicts = [edit_accepted(plan_obj, row["detections_before"], after)
                    for after in seen.values()]
        strict = all(ok for ok, _ in verdicts)
        # Both must see the edited fact; only one need see nothing else move.
        # See `edit_confirmed` for why the second half is not "both".
        ladder = (all(edit_confirmed(plan_obj, after) for after in seen.values())
                  and any(ok for ok, _ in verdicts))
        if strict or (args.gate == "ladder" and ladder):
            row = dict(row, gate="strict" if strict else "ladder")
            accepted.append(row)
            reasons["recolour_accepted" if strict
                    else "recolour_accepted_on_the_edited_fact"] += 1
        else:
            reasons["recolour_" + next(why for ok, why in verdicts if not ok)] += 1

    write_jsonl(out_dir / "accepted.jsonl", accepted)
    kinds = collections.Counter(r["edit"] for r in accepted)
    n_recolour_planned = sum(1 for r in plan.values()
                             if r["needs_check"] and r["edit"] == "recolour")
    strict_n = reasons["recolour_accepted"]
    total_n = strict_n + reasons["recolour_accepted_on_the_edited_fact"]
    print(f"accepted {len(accepted)}: {dict(kinds)}")
    if not n_recolour_planned:
        print(f"outcomes: {dict(reasons)}")
        return
    print(f"recolour yield {total_n / n_recolour_planned:.3f} "
          f"({total_n}/{n_recolour_planned}), of which {strict_n} pass the "
          f"strict whole-list gate")
    print(f"outcomes: {dict(reasons)}")


# --------------------------------------------------------------- questions


def _forced_choice(pair: dict[str, Any], built: dict[str, Any],
                   member: str) -> ForcedChoice:
    gold = built["gold_original"] if member == "original" else built["gold_edited"]
    return ForcedChoice(
        question_id=f"{pair['pair_id']}:{member}",
        spec_id=pair["spec"]["spec_id"],
        family={"binding": Family.BINDING, "spatial": Family.SPATIAL,
                "existence": Family.EXISTENCE}[built["family"]],
        prompt_text=built["question"],
        option_a=built["option_a"],
        option_b=built["option_b"],
        gold=gold,
        # The tag the analysis splits on. On a recoloured image the picture
        # contradicts the description by construction; on its original it agrees.
        # A flip pair is neither: the specs carry no relations, so both members
        # are equally consistent with the description and there is no deference
        # to measure -- calling them diagnostic would inflate the count with
        # trials that cannot separate the two hypotheses.
        gold_source=(
            # A sham changes no fact, so neither member contradicts the
            # description and neither is diagnostic: it is a control on the
            # artifact, not a trial.
            "no_spec_claim" if pair["edit"] in {"flip", "sham"}
            else ("image_differs_from_spec" if member == "edited"
                  else "spec_matches_image")
        ),
        metadata={"pair_id": pair["pair_id"], "edit": pair["edit"],
                  "member": member,
                  "colour_constrained": pair.get("colour_constrained")},
    )


def stage_questions(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    rows: list[dict[str, Any]] = []
    for pair in read_jsonl(out_dir / "accepted.jsonl"):
        # Which option is A is drawn per pair, not per member, so the two
        # members share one question string and differ only in gold.
        #
        # Seeded from the pair id rather than taken from a running generator, so
        # this stage is idempotent. It was not: accepting twelve more pairs
        # after a gate change shifted the stream and silently re-lettered 136 of
        # 254 already-answered trials, whose stored answers were then being
        # matched against options in the other order.
        gold_first = random.Random(f"{args.seed}:{pair['pair_id']}").random() < 0.5
        if pair["edit"] in {"delete", "sham"}:
            built = delete_question(pair["noun"], pair["colour"], gold_first)
            if pair["edit"] == "sham":
                # Nothing was removed, so the answer does not move. That is the
                # measurement: whatever the filler does to an image it does
                # here too, and here the right answer is the same on both sides.
                built = dict(built, gold_edited=built["gold_original"])
        elif pair["edit"] == "flip":
            built = flip_question(pair["subject"], pair["other"],
                                  pair["subject_left"], gold_first)
        else:
            built = recolour_question(pair["noun"], pair["source_colour"],
                                      pair["target_colour"], pair["count"],
                                      gold_first)
        for member, path in (("original", pair["original_path"]),
                             ("edited", pair["edited_path"])):
            question = _forced_choice(pair, built, member)
            rows.append({"image_path": path, "prompt": pair["spec"]["prompt"],
                         **question.to_dict()})
    write_jsonl(out_dir / "questions.jsonl", rows)
    print(f"{len(rows)} trials ({len(rows) // 2} pairs) -> "
          f"{out_dir / 'questions.jsonl'}")


# ----------------------------------------------------------------- observe


def stage_observe(args: argparse.Namespace) -> None:
    from selfsight.backbones.showo2 import Showo2Adapter

    out_dir = Path(args.outdir)
    questions = read_jsonl(out_dir / "questions.jsonl")
    prompted = args.condition == "prompted"
    out = out_dir / ("answers.prompted.jsonl" if prompted else "answers.jsonl")
    done: set[str] = set()
    if out.exists() and not args.overwrite:
        done = {r["question_id"] for r in read_jsonl(out)}
        print(f"resuming: {len(done)} already answered")
    todo = [row for row in questions if row["question_id"] not in done]
    print(f"{len(todo)} trials to answer, condition={args.condition}")
    if not todo:
        return

    by_image: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in todo:
        by_image[row["image_path"]].append(row)

    backbone = Showo2Adapter(device=args.device, lazy=False)
    started = time.time()
    with out.open("a" if done else "w", encoding="utf-8") as handle:
        for index, (image, rows) in enumerate(sorted(by_image.items())):
            choices = [
                ForcedChoice(
                    question_id=r["question_id"], spec_id=r["spec_id"],
                    family=Family(r["family"]), prompt_text=r["prompt_text"],
                    option_a=r["option_a"], option_b=r["option_b"],
                    gold=r["gold"], gold_source=r["gold_source"],
                    metadata=r["metadata"],
                )
                for r in rows
            ]
            atoms = [to_atomic(choice) for choice in choices]
            if prompted:
                atoms = [
                    replace(atom, text=PROMPTED_PREAMBLE.format(
                        prompt=rows[0]["prompt"], question=atom.text))
                    for atom in atoms
                ]
            observation = backbone.observe_atoms(image, atoms)
            for row, choice, answer in zip(rows, choices, observation.answers):
                correct = grade(answer.raw_answer, choice)
                handle.write(json.dumps({
                    "condition": args.condition,
                    "question_id": row["question_id"],
                    "image_path": image,
                    "family": row["family"],
                    "question": row["prompt_text"],
                    "gold": row["gold"],
                    "gold_source": row["gold_source"],
                    "metadata": row["metadata"],
                    "option_a": row["option_a"],
                    "option_b": row["option_b"],
                    "raw_answer": answer.raw_answer,
                    "correct": correct,
                    "abstain": correct is None,
                }, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index + 1}/{len(by_image)} images "
                      f"{(time.time() - started) / max(1, index + 1):.1f}s/img",
                      flush=True)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--run", required=True, action="append",
                   help="a main-pipeline run directory; repeat for both halves")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--edits", default="flip,recolour",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="which edits to plan. The deletion arm is built into a "
                        "separate outdir so the frozen recolour arm is not "
                        "replanned underneath its answers.")
    p.set_defaults(func=stage_plan)

    c = sub.add_parser("check")
    c.add_argument("--outdir", required=True)
    c.add_argument("--detector", choices=["qwen3vl", "internvl"], required=True)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=stage_check)

    a = sub.add_parser("accept")
    a.add_argument("--outdir", required=True)
    a.add_argument("--gate", choices=["strict", "ladder"], default="ladder",
                   help="strict: the whole detected list must move by exactly "
                        "one pair, for both detectors. ladder: both must see "
                        "the edited fact and one must see nothing else move, "
                        "which is the rule the main pipeline settled on.")
    a.set_defaults(func=stage_accept)

    q = sub.add_parser("questions")
    q.add_argument("--outdir", required=True)
    q.add_argument("--seed", type=int, default=20260901)
    q.set_defaults(func=stage_questions)

    o = sub.add_parser("observe")
    o.add_argument("--outdir", required=True)
    o.add_argument("--device", default="cuda:0")
    o.add_argument("--condition", choices=["image_only", "prompted"],
                   default="image_only")
    o.add_argument("--overwrite", action="store_true")
    o.set_defaults(func=stage_observe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
