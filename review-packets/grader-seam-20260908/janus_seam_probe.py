"""How much of Janus-Pro-1B's answer does the grader actually see?

The grader that produced every frozen number was written for a model that
replies with a bare letter. E4 puts the same questions to models that write a
sentence. This asks a real slice of the corpus and reports, on the same replies,
what the old rule scored and what the new one does -- so the size of the
instrument change is measured rather than asserted.

Understanding only, CPU only. Nothing here is evidence for a claim; it exists to
size a defect in the reader.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

WORKTREE = Path(r"H:\Xiyao_Wang\062_armB")
sys.path.insert(0, str(WORKTREE / "src"))

from selfsight.backbones.janus_pro import JanusProAdapter  # noqa: E402
from selfsight.v4.questions import build_questions, grade, to_atomic  # noqa: E402
from selfsight.v4.spec import SceneSpec, has_unnameable  # noqa: E402


def old_grade(reply: str, question) -> bool | None:
    """`grade` as it stood at 67bdfb5, for the side-by-side."""

    text = reply.strip().upper()
    picked = None
    for letter in ("A", "B"):
        if text.startswith(letter) or f" {letter}." in f" {text}":
            picked = letter
            break
    if picked is None:
        lowered = reply.strip().lower()
        matches = [
            letter
            for letter, option in (("A", question.option_a), ("B", question.option_b))
            if option.lower() in lowered
        ]
        if len(matches) == 1:
            picked = matches[0]
    if picked is None:
        return None
    return picked == question.gold


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/v4/main-2plus1")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    verified = {row["image_path"]: row for row in read_jsonl(run / "verified.jsonl")}
    manifest = read_jsonl(run / "manifest.jsonl")

    observer = JanusProAdapter(
        device=args.device,
        dtype="fp32" if args.device == "cpu" else "bf16",
        max_new_tokens=args.max_new_tokens,
        lazy=False,
    )
    print(f"loaded {observer.model_id} @ {observer.revision[:12]} on {args.device}", flush=True)

    rows: list[dict] = []
    images = 0
    for row in manifest:
        if images >= args.images:
            break
        image = row["image_path"]
        if image not in verified:
            continue
        settled = verified[image]["detections"]
        if has_unnameable(settled):
            continue
        spec = SceneSpec.from_dict(row["spec"])
        questions = build_questions(spec, settled, seed=row["seed"])
        if not questions:
            continue
        images += 1
        observation = observer.observe_atoms(image, [to_atomic(q) for q in questions])
        for question, answer in zip(questions, observation.answers):
            raw = answer.raw_answer
            rows.append({
                "spec_id": spec.spec_id,
                "family": question.family.value,
                "gold": question.gold,
                "gold_source": question.gold_source,
                "option_a": question.option_a,
                "option_b": question.option_b,
                "raw_answer": raw,
                "old": old_grade(raw, question),
                "new": grade(raw, question),
            })
        print(f"{images}/{args.images} images, {len(rows)} trials", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(
        {"run": args.run, "device": args.device, "max_new_tokens": args.max_new_tokens,
         "model": observer.identity, "trials": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    total = len(rows)
    old_abstain = sum(1 for r in rows if r["old"] is None)
    new_abstain = sum(1 for r in rows if r["new"] is None)
    moved = [r for r in rows if r["old"] is not r["new"]]
    print(f"\n{total} trials over {images} images")
    print(f"  abstention  old {old_abstain}/{total} ({old_abstain / total:.1%})"
          f"   new {new_abstain}/{total} ({new_abstain / total:.1%})")
    print(f"  grades that move: {len(moved)}")
    for kind, count in collections.Counter(
        f"{r['old']} -> {r['new']}" for r in moved
    ).most_common():
        print(f"      {kind:<20} {count}")
    print("\nreply shapes:")
    for reply, count in collections.Counter(r["raw_answer"][:70] for r in rows).most_common(15):
        print(f"  {count:>4}  {reply!r}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
