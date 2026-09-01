"""Build a review sheet for the images the two detectors could not settle.

This is level 3 of the ladder, and it is the reason the ladder is allowed to keep
every image instead of filtering. Filtering to the agreeing subset would drop the
occluded and ambiguous scenes, resampling the corpus toward easy ones -- so the
hard images stay in and a person settles them here.

The sheet shows what is in dispute and nothing else. It does not show the prompt,
the spec, or either detector's verdict on whether the image is correct, because a
human told "this was supposed to be two pears" will see two pears. The question
put to the reviewer is the same one put to the detectors: what objects are in
this picture. Their answer overrides both models.

    python scripts/v4_human_review.py --run runs/v4/main --out review.html

Fill in `review.template.jsonl` next to it, rename to `human_labels.jsonl` in the
run directory, and rerun the verify stage: those rows resolve as HUMAN.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>v4 human adjudication - {n} images</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 980px;
         padding: 24px; color: #111; }}
  h1 {{ font-size: 20px; }}
  .intro {{ background: #f6f6f4; padding: 14px 18px; border-radius: 8px;
            margin-bottom: 28px; }}
  .card {{ display: grid; grid-template-columns: 340px 1fr; gap: 22px;
           padding: 20px 0; border-top: 1px solid #ddd; align-items: start; }}
  img {{ width: 320px; height: auto; border-radius: 6px; background: #eee; }}
  .id {{ font: 12px ui-monospace, monospace; color: #666; }}
  .dispute {{ font-weight: 600; color: #b03030; }}
  table {{ border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
  td, th {{ padding: 3px 12px 3px 0; text-align: left; vertical-align: top; }}
  th {{ color: #666; font-weight: 500; }}
  code {{ background: #f2f2f0; padding: 1px 5px; border-radius: 3px; }}
</style>
<h1>Human adjudication &mdash; {n} images</h1>
<div class="intro">
  <p>The two detectors disagree about these images and the crop pass did not
  settle it. For each one: <strong>what objects are actually in the picture?</strong>
  Give a common noun and a colour for every object, counting each separately.</p>
  <p>Do not list the surface, background, shadows or reflections. The disputed
  object is named in red, but check the whole list &mdash; your answer replaces
  both models&rsquo;, it does not just arbitrate the one disagreement.</p>
  <p>Neither the prompt nor what the image was supposed to contain is shown, on
  purpose: knowing that two pears were requested makes two pears easier to see.</p>
</div>
{cards}
"""

CARD = """<div class="card">
  <div><img src="data:image/png;base64,{b64}" alt=""></div>
  <div>
    <div class="id">{index}. {name}</div>
    <p>in dispute: <span class="dispute">{disputed}</span></p>
    <table>
      <tr><th>{primary}</th><td>{primary_list}</td></tr>
      <tr><th>{secondary}</th><td>{secondary_list}</td></tr>
    </table>
  </div>
</div>
"""


def _fmt(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "<em>nothing</em>"
    counts: dict[str, int] = {}
    for item in detections:
        key = f"{item.get('color') or '?'} {item.get('object')}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(
        f"{n}&times; {html.escape(k)}" if n > 1 else html.escape(k)
        for k, n in sorted(counts.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--primary", default="qwen3vl")
    parser.add_argument("--secondary", default="internvl")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    from selfsight.v4.detectors import cached_detections

    verified = [
        json.loads(line)
        for line in (args.run / "verified.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pending = [r for r in verified if r["resolution"] == "pending_human"]
    if not pending:
        print("nothing pending: every image was settled by the detectors")
        return

    primary = cached_detections(args.run / f"detections.{args.primary}.jsonl")
    secondary_path = args.run / f"detections.{args.secondary}.jsonl"
    secondary = cached_detections(secondary_path) if secondary_path.exists() else {}

    cards = []
    template = []
    for index, row in enumerate(pending[: args.limit], start=1):
        image_path = Path(row["image_path"])
        try:
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError:
            continue
        disputed = ", ".join(
            f"{d.get('color') or '?'} {d.get('object')} ({d.get('reason', '')})"
            for d in row.get("disputed", ())
        ) or "n/a"
        cards.append(CARD.format(
            b64=b64,
            index=index,
            name=html.escape(image_path.name),
            disputed=html.escape(disputed),
            primary=html.escape(args.primary),
            secondary=html.escape(args.secondary),
            primary_list=_fmt(primary.get(row["image_path"], [])),
            secondary_list=_fmt(secondary.get(row["image_path"], [])),
        ))
        template.append({
            "image_path": row["image_path"],
            "index": index,
            "detections": [{"object": "", "color": ""}],
        })

    out = args.out or args.run / "review.html"
    out.write_text(PAGE.format(n=len(cards), cards="\n".join(cards)), encoding="utf-8")
    out.with_name("review.template.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in template),
        encoding="utf-8",
    )
    print(f"{len(pending)} pending, {len(cards)} in the sheet -> {out}")
    if len(pending) > args.limit:
        print(f"note: {len(pending) - args.limit} beyond --limit not included")
    print(f"adjudication rate: {len(pending) / max(1, len(verified)):.1%} "
          f"(locked criterion: report it, must be <= 10%)")


if __name__ == "__main__":
    main()
