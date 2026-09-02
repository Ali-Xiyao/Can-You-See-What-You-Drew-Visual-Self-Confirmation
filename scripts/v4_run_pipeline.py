"""Run the v4 measurement end to end: generate, verify, ask, score.

Four stages, each writing its own file, so a stage can be rerun without redoing
the one before it. The order matters and is not arbitrary:

    generate   spec  -> K candidate images
    detect     image -> object list, from one or two frozen external models
    crop       disputed object -> a second look with more pixels on it
    verify     object list + spec -> image_correct, via the escalation ladder
    observe    image -> the generator's own 2AFC answers -> graded

The questions are built *after* detection and from the detected objects, never
from the spec. This is the point of the whole design: the model is asked what it
drew, not what it was told to draw, so when it drew three pears having been
asked for two, "three" is the correct answer and reciting the prompt is wrong.
Building the questions from the spec would quietly turn the experiment into a
prompt-recall test and the most informative trials would disappear.

    # stage 1, on the generation env
    envs/showo2/python.exe scripts/v4_run_pipeline.py generate \
        --corpus data/v4/corpus.jsonl --outdir runs/v4/main --device cuda:0

    # stage 2, twice, on the observer env
    envs/observer/python.exe scripts/v4_run_pipeline.py detect \
        --manifest runs/v4/main/manifest.jsonl --detector qwen3vl --device cuda:0
    envs/observer/python.exe scripts/v4_run_pipeline.py detect \
        --manifest runs/v4/main/manifest.jsonl --detector internvl --device cuda:1

    # stage 3, on the observer env: ladder level 2, only the disputed objects
    envs/observer/python.exe scripts/v4_run_pipeline.py crop         --run runs/v4/main --device cuda:0

    # stage 4, no GPU
    python scripts/v4_run_pipeline.py verify --run runs/v4/main

    # stage 5, on the generation env: the generator answers about its own images
    envs/showo2/python.exe scripts/v4_run_pipeline.py observe \
        --run runs/v4/main --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from selfsight.v4.questions import build_questions, grade, to_atomic
from selfsight.v4.spec import SceneSpec, has_unnameable
from selfsight.v4.verifier import Resolution, ladder_summary, verify

BASE_SEED = 20260901


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- generate


def stage_generate(args: argparse.Namespace) -> None:
    from selfsight.backbones.showo2 import Showo2Adapter

    specs = [SceneSpec.from_dict(row) for row in read_jsonl(args.corpus)]
    specs = [s for i, s in enumerate(specs) if i % args.shards == args.shard]
    run = Path(args.outdir)
    images = run / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = run / (f"manifest.{args.shard}.jsonl" if args.shards > 1 else "manifest.jsonl")

    backbone = Showo2Adapter(device=args.device, lazy=False)
    started = time.time()
    rows: list[dict[str, Any]] = []
    with manifest.open("w", encoding="utf-8") as handle:
        for index, spec in enumerate(specs):
            # Seeds are derived from the spec index, not drawn: the same corpus
            # must give the same images on a rerun, and the K candidates for one
            # spec must differ from each other.
            seeds = [BASE_SEED + index * args.k + j for j in range(args.k)]
            records = backbone.generate_images(
                [spec.prompt] * args.k, seeds, str(images), checkpoint_id="v4-main"
            )
            for j, record in enumerate(records):
                row = {
                    "spec_id": spec.spec_id,
                    "candidate_index": j,
                    "seed": seeds[j],
                    "prompt": spec.prompt,
                    "image_path": str(record.image_path),
                    "spec": spec.to_dict(),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            handle.flush()
            if index % 10 == 0:
                rate = (time.time() - started) / max(1, index * args.k + args.k)
                print(f"{index + 1}/{len(specs)} specs, {rate:.1f}s/img", flush=True)
    print(f"wrote {len(rows)} rows to {manifest}")


# ------------------------------------------------------------------ detect


def stage_detect(args: argparse.Namespace) -> None:
    from selfsight.v4.detectors import load

    rows = read_jsonl(args.manifest)
    out = args.output or args.manifest.with_name(f"detections.{args.detector}.jsonl")
    done: set[str] = set()
    if out.exists() and not args.overwrite:
        done = {r["image_path"] for r in read_jsonl(out) if "detections" in r}
        print(f"resuming: {len(done)} already done")

    detector = load(args.detector, device=args.device)
    started = time.time()
    with out.open("a" if done else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["image_path"] in done:
                continue
            try:
                detections = detector.detect(row["image_path"])
                record = {"image_path": row["image_path"], "detections": detections,
                          "reply": getattr(detector, "last_reply", "")}
            except (ValueError, OSError) as exc:
                # Recorded as an error row with no `detections` key, so the
                # verify stage sees it as absent rather than as an empty scene.
                record = {"image_path": row["image_path"], "error": str(exc)[:300]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index + 1}/{len(rows)} "
                      f"{(time.time() - started) / max(1, index + 1):.1f}s/img", flush=True)
    print(f"wrote {out}")


# ------------------------------------------------------------------- crop


def crop_key(image_path: str, bbox: Any) -> str:
    """Cache key for one crop query: the image and the exact region asked about.

    Keying by image alone would pool every crop taken from it, and a hit found in
    one region would then settle a dispute about another. That is not what the
    ladder claims -- level 2 says "looked closely at this object and saw it", not
    "looked closely somewhere and saw it".
    """
    return "{}||{}".format(
        image_path, ",".join(f"{float(v):.1f}" for v in bbox)
    )


def stage_crop(args: argparse.Namespace) -> None:
    """Level 2 of the ladder: re-ask the primary about each disputed object.

    Run between `detect` and `verify`. Without it every disagreement falls
    straight through to a human, which both blows the adjudication budget and
    wastes the cheaper evidence: most disputes are one small or occluded object
    that more pixels settle.
    """
    from selfsight.v4.detectors import cached_detections, load
    from selfsight.v4.spec import image_correct
    from selfsight.v4.verifier import _bbox_of, _pad, disputed_keys

    run = Path(args.run)
    primary = cached_detections(run / f"detections.{args.primary}.jsonl")
    secondary = cached_detections(run / f"detections.{args.secondary}.jsonl")
    specs = {
        row["image_path"]: SceneSpec.from_dict(row["spec"])
        for row in read_jsonl(run / "manifest.jsonl")
    }

    jobs: list[tuple[str, tuple[float, ...], tuple[str, str | None]]] = []
    no_bbox = 0
    same_verdict = 0
    for image, first in primary.items():
        second = secondary.get(image)
        if second is None:
            continue
        spec = specs.get(image)
        # A dispute that cannot change image_correct does not need a second look.
        # Querying it anyway spent three quarters of this pass on images whose
        # verdict was never in doubt.
        if spec is not None and disputed_keys(first, second):
            if image_correct(spec, first) == image_correct(spec, second):
                same_verdict += 1
                continue
        for key in disputed_keys(first, second):
            bbox = _bbox_of(first, key) or _bbox_of(second, key)
            if bbox is None:
                # Neither detector located it, so there is no region to enlarge.
                # Counted, not silently skipped: these are the rows the ladder
                # cannot help and that must reach a human.
                no_bbox += 1
                continue
            jobs.append((image, _pad(bbox), key))

    out = run / "detections.crops.jsonl"
    done: set[str] = set()
    if out.exists() and not args.overwrite:
        done = {r["key"] for r in read_jsonl(out) if "detections" in r}
        print(f"resuming: {len(done)} crops already done")
    todo = [j for j in jobs if crop_key(j[0], j[1]) not in done]
    print(f"{len(jobs)} disputed objects, {len(todo)} to query, "
          f"{no_bbox} with no box from either detector, "
          f"{same_verdict} images skipped because the verdict agrees anyway")
    if not todo:
        return

    detector = load(args.primary, device=args.device)
    started = time.time()
    with out.open("a" if done else "w", encoding="utf-8") as handle:
        for index, (image, bbox, key) in enumerate(todo):
            record: dict[str, Any] = {
                "key": crop_key(image, bbox),
                "image_path": image,
                "bbox": list(bbox),
                "disputed": [key[0], key[1]],
            }
            try:
                record["detections"] = detector.detect_crop(image, bbox)
            except (ValueError, OSError, RuntimeError) as exc:
                # No `detections` key, so verify treats the crop as having
                # settled nothing and the row escalates rather than resolving
                # against the object by default.
                #
                # RuntimeError covers CUDA OOM, which is a property of one crop's
                # shape rather than of the run. Killing 400 queries because the
                # 50th was a sliver is the wrong trade; the row is recorded as
                # unsettled and the image goes to a human.
                record["error"] = str(exc)[:300]
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - diagnostics only
                    pass
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index + 1}/{len(todo)} "
                      f"{(time.time() - started) / max(1, index + 1):.1f}s/crop",
                      flush=True)
    print(f"wrote {out}")


def cached_crops(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Crop results keyed by `crop_key`, skipping the rows that errored."""
    if not path.exists():
        return {}
    return {
        row["key"]: list(row["detections"])
        for row in read_jsonl(path)
        if "detections" in row
    }


# ------------------------------------------------------------------ verify


class _Replay:
    """A `Detector` backed by a detections file instead of a GPU."""

    def __init__(self, detector_id: str, table: dict[str, list[dict[str, Any]]],
                 crops: dict[str, list[dict[str, Any]]] | None = None):
        self.detector_id = detector_id
        self._table = table
        self._crops = crops or {}

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return list(self._table[image_path])

    def detect_crop(self, image_path: str, bbox: Any) -> list[dict[str, Any]]:
        # Absent a cached crop pass, no dispute can be settled at level 2 and the
        # row escalates to PENDING_HUMAN. That is the honest outcome: pretending
        # the crop confirmed nothing would silently resolve disputes against the
        # object, which is a decision, not a default.
        return list(self._crops.get(crop_key(image_path, bbox), []))


def stage_verify(args: argparse.Namespace) -> None:
    from selfsight.v4.detectors import cached_detections

    run = Path(args.run)
    rows = read_jsonl(run / "manifest.jsonl")
    primary = cached_detections(run / f"detections.{args.primary}.jsonl")
    secondary_path = run / f"detections.{args.secondary}.jsonl"
    secondary = cached_detections(secondary_path) if secondary_path.exists() else {}
    crops = cached_crops(run / "detections.crops.jsonl")

    human_path = run / "human_labels.jsonl"
    human = cached_detections(human_path) if human_path.exists() else None

    results = []
    skipped = 0
    for row in rows:
        image = row["image_path"]
        if image not in primary:
            skipped += 1
            continue
        spec = SceneSpec.from_dict(row["spec"])
        second = (
            _Replay(args.secondary, secondary, crops) if image in secondary else None
        )
        result = verify(
            image, spec,
            _Replay(args.primary, primary, crops),
            second,
            human_labels=human,
        )
        record = result.to_dict()
        record.update({"candidate_index": row["candidate_index"], "seed": row["seed"]})
        results.append(record)

    write_jsonl(run / "verified.jsonl", results)

    from selfsight.v4.verifier import VerificationResult

    summary = ladder_summary([
        VerificationResult(
            image_path=r["image_path"], spec_id=r["spec_id"],
            detections=tuple(r["detections"]), image_correct=r["image_correct"],
            resolution=Resolution(r["resolution"]),
            verifier_agreement=r["verifier_agreement"],
        )
        for r in results
    ])
    summary["skipped_no_detection"] = skipped
    summary["selection_ceiling"] = _ceiling(results)
    (run / "verified.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _ceiling(results: list[dict[str, Any]]) -> dict[str, Any]:
    """p, C(p), and the balanced-pool rate the selection experiment needs.

    Every generated image is in here, including the ones holding something no
    one could name. Those are failures like any other mismatch: a candidate that
    came out unreadable is a candidate the selector cannot use, and excusing it
    from p would raise the measured generation rate by deleting the generator's
    worst output from its own score.
    """
    by_spec: dict[str, list[bool]] = {}
    for row in results:
        by_spec.setdefault(row["spec_id"], []).append(bool(row["image_correct"]))
    n = sum(len(v) for v in by_spec.values()) or 1
    correct = sum(sum(v) for v in by_spec.values())
    p = correct / n
    k = max((len(v) for v in by_spec.values()), default=4)
    balanced = sum(1 for v in by_spec.values() if 0 < sum(v) < len(v))
    return {
        "n_images": n,
        "n_specs": len(by_spec),
        "p": round(p, 4),
        "k": k,
        "C_p": round((1 - (1 - p) ** k) - p, 4),
        "balanced_pool_rate": round(balanced / max(1, len(by_spec)), 4),
        "balanced_pools": balanced,
    }


# ----------------------------------------------------------------- observe


PROMPTED_PREAMBLE = """You were asked to draw a picture from this description:
"{prompt}"

Here is the picture you drew. Answer about what is actually in the picture.

{question}"""
"""The prompted observation condition.

Without this the model never sees the description it drew from, and then the
"answers from the prompt rather than the pixels" hypothesis has nothing to
answer from: observe_atoms hands it an image and a question and no state carries
over from generation. The image-only condition measures how well the model reads
its own picture, and the gap between an on-spec and an off-spec image is real,
but it is a fact about difficulty until this condition is run beside it.

The wording puts the description first and then says outright to answer about
the picture, so a model that goes with the description is going against an
explicit instruction rather than filling a gap in an ambiguous one.
"""


def stage_observe(args: argparse.Namespace) -> None:
    from selfsight.backbones.showo2 import Showo2Adapter

    run = Path(args.run)
    verified = {r["image_path"]: r for r in read_jsonl(run / "verified.jsonl")}
    manifest = read_jsonl(run / "manifest.jsonl")

    prompted = args.condition == "prompted"
    out = run / ("answers.prompted.jsonl" if prompted else "answers.jsonl")
    done: set[str] = set()
    if out.exists() and not args.overwrite:
        done = {r["image_path"] for r in read_jsonl(out)}
        print(f"resuming: {len(done)} images already answered")
    todo = sum(1 for row in manifest
               if row["image_path"] in verified and row["image_path"] not in done)
    if not todo:
        print("nothing to do")
        return

    backbone = Showo2Adapter(device=args.device, lazy=False)
    started = time.time()
    written = 0
    skipped_unnameable = 0
    with out.open("a" if done else "w", encoding="utf-8") as handle:
        for index, row in enumerate(manifest):
            image = row["image_path"]
            if image not in verified or image in done:
                continue
            spec = SceneSpec.from_dict(row["spec"])
            settled = verified[image]["detections"]
            # An image containing something nobody could name has a verdict --
            # a miss -- but no answerable questions: "how many pears" has no
            # answer when one candidate is half a pear. It counts toward p and
            # supplies no trials. See v4.spec.UNNAMEABLE.
            if has_unnameable(settled):
                skipped_unnameable += 1
                continue
            questions = build_questions(spec, settled, seed=row["seed"])
            if not questions:
                continue
            atoms = [to_atomic(question) for question in questions]
            if prompted:
                atoms = [
                    replace(atom, text=PROMPTED_PREAMBLE.format(
                        prompt=spec.prompt, question=atom.text))
                    for atom in atoms
                ]
            observation = backbone.observe_atoms(image, atoms)
            for question, answer in zip(questions, observation.answers):
                raw = answer.raw_answer
                correct = grade(raw, question)
                handle.write(json.dumps({
                    "condition": args.condition,
                    "spec_id": spec.spec_id,
                    "image_path": image,
                    "candidate_index": row["candidate_index"],
                    "family": question.family.value,
                    "question": question.prompt_text,
                    "gold": question.gold,
                    "gold_source": question.gold_source,
                    # The builders' bookkeeping travels with the trial. Without
                    # it the analysis cannot split the existence family into the
                    # substitution trials, where a prompt-reciter is wrong, and
                    # the rest, where it is merely at chance.
                    "metadata": dict(question.metadata),
                    "option_a": question.option_a,
                    "option_b": question.option_b,
                    "raw_answer": raw,
                    "correct": correct,
                    "abstain": correct is None,
                    "image_correct": verified[image]["image_correct"],
                    "resolution": verified[image]["resolution"],
                }, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            if index % 20 == 0:
                print(f"{index + 1}/{len(manifest)} images, {written} answers, "
                      f"{(time.time() - started) / max(1, index + 1):.1f}s/img", flush=True)
    if skipped_unnameable:
        print(f"{skipped_unnameable} images supplied no trials: something in "
              f"them is not any object")
    print(f"wrote {written} answers to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--corpus", type=Path, required=True)
    g.add_argument("--outdir", required=True)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--k", type=int, default=4)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--shards", type=int, default=1)
    g.set_defaults(func=stage_generate)

    d = sub.add_parser("detect")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--detector", choices=["qwen3vl", "internvl"], required=True)
    d.add_argument("--device", default="cuda:0")
    d.add_argument("--output", type=Path)
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=stage_detect)

    v = sub.add_parser("verify")
    v.add_argument("--run", required=True)
    v.add_argument("--primary", default="qwen3vl")
    v.add_argument("--secondary", default="internvl")

    c = sub.add_parser("crop", help="ladder level 2: re-ask about disputed objects")
    c.add_argument("--run", required=True, type=Path)
    c.add_argument("--primary", default="qwen3vl")
    c.add_argument("--secondary", default="internvl")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=stage_crop)
    v.set_defaults(func=stage_verify)

    o = sub.add_parser("observe")
    o.add_argument("--run", required=True)
    o.add_argument("--device", default="cuda:0")
    # Resumable so the rows a human later re-labels can be re-answered on their
    # own: drop those images from answers.jsonl and run the stage again, rather
    # than paying for all 468 to recover 36.
    o.add_argument("--overwrite", action="store_true")
    o.add_argument("--condition", choices=["image_only", "prompted"],
                   default="image_only")
    o.set_defaults(func=stage_observe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
