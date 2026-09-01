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

    python scripts/v4_human_review.py --run runs/v4/main

The page is a form. Type each answer, press Download, and drop the file into the
run directory as `human_labels.jsonl`, then rerun verify: those rows resolve as
HUMAN. The earlier version emitted a `review.template.jsonl` to be edited by hand
in a text editor, which meant transcribing seventy JSON objects and getting the
image paths right; the form writes the same file from the same fields.

The answer boxes start empty and are never pre-filled from a detector. There are
buttons to copy either detector's list in, because retyping a five-object scene
to change one word is a waste, but agreeing with a model has to be something the
reviewer does rather than something they leave alone.
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
         padding: 24px 24px 110px; color: #111; }}
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
  input.answer {{ width: 100%; box-sizing: border-box; margin-top: 12px;
                  font: 15px ui-monospace, monospace; padding: 8px 10px;
                  border: 2px solid #ccc; border-radius: 6px; }}
  input.answer:focus {{ border-color: #3b6ea5; outline: none; }}
  .card.done input.answer {{ border-color: #3a8a4a; background: #f4fbf5; }}
  .parsed {{ font-size: 13px; color: #444; min-height: 20px; margin-top: 6px; }}
  .parsed.bad {{ color: #b03030; }}
  .fill button {{ font-size: 12px; margin: 6px 8px 0 0; padding: 3px 9px;
                  border: 1px solid #bbb; background: #fafafa; border-radius: 5px;
                  cursor: pointer; }}
  #bar {{ position: fixed; left: 0; right: 0; bottom: 0; background: #111;
          color: #fff; padding: 12px 24px; display: flex; gap: 18px;
          align-items: center; justify-content: center; }}
  #bar button {{ font-size: 15px; padding: 8px 16px; border-radius: 6px;
                 border: 0; cursor: pointer; }}
  #save {{ background: #3a8a4a; color: #fff; }}
  #save[disabled] {{ background: #555; cursor: not-allowed; }}
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
  <p><strong>How to write it.</strong> Comma-separated, one entry per kind, as
  <code>count colour noun</code>; the count may be left off when it is one. So
  <code>2 blue mug, white plate, green pear</code>. A picture with nothing in it
  is <code>none</code>. What you typed is parsed back underneath &mdash; read that
  line, not the box, to check it understood you.</p>
  <p>The buttons under each box copy a detector&rsquo;s list into the field so you
  can edit rather than retype. The box starts empty on purpose: if it came
  pre-filled, agreeing with a model would be the option that takes no effort.</p>
  <p>Answers are kept in this browser as you type, so you can close the page and
  come back. When the counter reads {n} of {n}, press <strong>Download</strong>
  and save the file into <code>{run}</code> as <code>human_labels.jsonl</code>.</p>
</div>
{cards}
<div id="bar">
  <span id="count"></span>
  <button id="save" disabled>Download human_labels.jsonl</button>
  <button id="clear">Clear all</button>
</div>
<script>
const KEY = "selfsight-review:{run_key}";
const cards = Array.from(document.querySelectorAll(".card"));

function parse(text) {{
  const s = text.trim();
  if (!s) return null;
  if (s.toLowerCase() === "none") return [];
  const out = [];
  for (const chunk of s.split(",")) {{
    const words = chunk.trim().split(/\\s+/).filter(Boolean);
    if (!words.length) continue;
    let count = 1;
    if (/^[0-9]+$/.test(words[0])) count = parseInt(words.shift(), 10);
    if (words.length < 2) return "needs a colour and a noun: " + chunk.trim();
    if (count < 1 || count > 12) return "odd count in: " + chunk.trim();
    const noun = words.pop();
    const colour = words.join(" ");
    for (let i = 0; i < count; i++) out.push({{object: noun, color: colour}});
  }}
  return out.length ? out : "nothing understood";
}}

function describe(list) {{
  if (!list.length) return "nothing in the picture";
  const seen = new Map();
  for (const d of list) {{
    const k = d.color + " " + d.object;
    seen.set(k, (seen.get(k) || 0) + 1);
  }}
  return Array.from(seen, ([k, n]) => (n > 1 ? n + " x " + k : k)).join(", ");
}}

function refresh() {{
  let done = 0;
  for (const card of cards) {{
    const input = card.querySelector("input.answer");
    const out = card.querySelector(".parsed");
    const result = parse(input.value);
    if (result === null) {{
      out.textContent = ""; out.className = "parsed";
      card.classList.remove("done");
    }} else if (typeof result === "string") {{
      out.textContent = result; out.className = "parsed bad";
      card.classList.remove("done");
    }} else {{
      out.textContent = "\\u2192 " + describe(result);
      out.className = "parsed"; card.classList.add("done"); done++;
    }}
  }}
  document.getElementById("count").textContent =
      done + " of " + cards.length + " done";
  document.getElementById("save").disabled = done !== cards.length;
  const state = {{}};
  for (const card of cards) {{
    state[card.dataset.path] = card.querySelector("input.answer").value;
  }}
  localStorage.setItem(KEY, JSON.stringify(state));
}}

const saved = JSON.parse(localStorage.getItem(KEY) || "{{}}");
for (const card of cards) {{
  const input = card.querySelector("input.answer");
  if (saved[card.dataset.path]) input.value = saved[card.dataset.path];
  input.addEventListener("input", refresh);
  card.querySelectorAll(".fill button").forEach(function (b) {{
    b.addEventListener("click", function () {{
      input.value = b.dataset.list; input.focus(); refresh();
    }});
  }});
}}
refresh();

document.getElementById("save").addEventListener("click", function () {{
  const lines = cards.map(function (card, i) {{
    return JSON.stringify({{
      image_path: card.dataset.path,
      index: i + 1,
      detections: parse(card.querySelector("input.answer").value),
    }});
  }});
  const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "application/jsonl"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_labels.jsonl";
  a.click();
}});

document.getElementById("clear").addEventListener("click", function () {{
  if (!confirm("Clear every answer on this page?")) return;
  localStorage.removeItem(KEY);
  for (const card of cards) card.querySelector("input.answer").value = "";
  refresh();
}});
</script>
"""

CARD = """<div class="card" data-path="{path}">
  <div><img src="data:image/png;base64,{b64}" alt=""></div>
  <div>
    <div class="id">{index}. {name}</div>
    <p>in dispute: <span class="dispute">{disputed}</span></p>
    <table>
      <tr><th>{primary}</th><td>{primary_list}</td></tr>
      <tr><th>{secondary}</th><td>{secondary_list}</td></tr>
    </table>
    <input class="answer" placeholder="2 blue mug, white plate" spellcheck="false">
    <div class="parsed"></div>
    <div class="fill">
      <button data-list="{primary_fill}">use {primary}</button>
      <button data-list="{secondary_fill}">use {secondary}</button>
    </div>
  </div>
</div>
"""


def _counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        key = f"{item.get('color') or '?'} {item.get('object')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fmt(detections: list[dict[str, Any]]) -> str:
    counts = _counts(detections)
    if not counts:
        return "<em>nothing</em>"
    return ", ".join(
        f"{n}&times; {html.escape(k)}" if n > 1 else html.escape(k)
        for k, n in sorted(counts.items())
    )


def _fill(detections: list[dict[str, Any]]) -> str:
    """The same list in the syntax the box accepts, for the copy buttons."""
    counts = _counts(detections)
    if not counts:
        return "none"
    return html.escape(", ".join(
        f"{n} {k}" if n > 1 else k for k, n in sorted(counts.items())
    ), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--primary", default="qwen3vl")
    parser.add_argument("--secondary", default="internvl")
    parser.add_argument("--limit", type=int, default=200)
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
        first = primary.get(row["image_path"], [])
        second = secondary.get(row["image_path"], [])
        cards.append(CARD.format(
            b64=b64,
            path=html.escape(row["image_path"], quote=True),
            index=index,
            name=html.escape(image_path.name),
            disputed=html.escape(disputed),
            primary=html.escape(args.primary),
            secondary=html.escape(args.secondary),
            primary_list=_fmt(first),
            secondary_list=_fmt(second),
            primary_fill=_fill(first),
            secondary_fill=_fill(second),
        ))

    out = args.out or args.run / "review.html"
    out.write_text(
        PAGE.format(
            n=len(cards),
            cards="\n".join(cards),
            run=html.escape(str(args.run)),
            run_key=html.escape(args.run.name, quote=True),
        ),
        encoding="utf-8",
    )
    print(f"{len(pending)} pending, {len(cards)} in the sheet -> {out}")
    if len(pending) > args.limit:
        print(f"note: {len(pending) - args.limit} beyond --limit not included")
    print(f"adjudication rate: {len(pending) / max(1, len(verified)):.1%} "
          f"(locked criterion: report it, must be <= 10%)")


if __name__ == "__main__":
    main()
