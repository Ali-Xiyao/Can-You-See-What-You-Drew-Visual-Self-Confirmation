"""Build the review sheets for the images the two detectors could not settle.

This is level 3 of the ladder, and it is the reason the ladder is allowed to keep
every image instead of filtering. Filtering to the agreeing subset would drop the
occluded and ambiguous scenes, resampling the corpus toward easy ones -- so the
hard images stay in and a person settles them here.

The sheet shows what is in dispute and nothing else. It does not show the prompt,
the spec, or either detector's verdict on whether the image is correct, because a
human told "this was supposed to be two pears" will see two pears.

    python scripts/v4_human_review.py --run runs/v4/main --chunk 10 --inline-images

**Three choices, not free text.** The reviewer picks one of:

    Q  the qwen3vl list is right
    I  the internvl list is right
    X  something in this picture is not any object -- two things fused into one
       body, or a shape that cannot be named -- so neither list is right and no
       list would be

The earlier version asked the reviewer to type the object list. Typing is the
right interface when the answer is a list nobody has written down yet; here two
lists are already on screen and one of them is almost always correct, so typing
mostly re-enters what is already there and invites transcription errors. `X` is
not an escape hatch from the two lists, it is the third real outcome, and it
carries a verdict of its own: see v4.spec.UNNAMEABLE.

The fourth case -- both lists wrong and the picture readable -- is rare but real,
so `other` still takes a typed list. It is deliberately the least prominent
control on the card: it should cost more than clicking, because a reviewer who
finds it easier to retype than to compare will retype.

Which list sits on the left is randomised per card. Both buttons name their
detector, so "pick qwen's" is still one click, but a reviewer working quickly
down forty cards cannot fall into always taking the top one.

**Pages are chunked and self-contained.** Every image is re-encoded as a small
JPEG and inlined, so a page opens anywhere -- in a chat client, on a phone, from
a copy on the desktop -- with no sibling `images/` directory to lose. One page of
all 42 came to 10.6 MB and would not open at all; `--chunk 10` puts it at about
400 KB a page. The re-encode is display-only and never touches the corpus.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from pathlib import Path
from typing import Any

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人工裁定 {part} — 第 {lo}–{hi} 张</title>
<style>
  body {{ font: 15px/1.6 system-ui, "Microsoft YaHei", sans-serif;
          margin: 0 auto; max-width: 900px; padding: 20px 20px 60px; color: #111; }}
  h1 {{ font-size: 19px; margin-bottom: 4px; }}
  .intro {{ background: #f6f6f4; padding: 14px 18px; border-radius: 8px;
            margin-bottom: 24px; font-size: 14px; }}
  .intro p {{ margin: 8px 0; }}
  .card {{ padding: 22px 0; border-top: 1px solid #ddd; }}
  .head {{ display: flex; align-items: baseline; gap: 12px; }}
  .num {{ font-size: 22px; font-weight: 700; }}
  .id {{ font: 11px ui-monospace, monospace; color: #888; }}
  .body {{ display: grid; grid-template-columns: 300px 1fr; gap: 20px;
           margin-top: 10px; align-items: start; }}
  img {{ width: 300px; height: auto; border-radius: 6px; background: #eee;
         display: block; }}
  .dispute {{ font-size: 13px; color: #b03030; margin-bottom: 10px; }}
  .opts {{ display: flex; flex-direction: column; gap: 8px; }}
  .opt {{ display: block; width: 100%; text-align: left; cursor: pointer;
          border: 2px solid #ccc; border-radius: 8px; background: #fff;
          padding: 9px 13px; font: inherit; }}
  .opt:hover {{ border-color: #888; }}
  .opt .tag {{ font: 12px ui-monospace, monospace; color: #666; display: block; }}
  .opt .list {{ font-size: 14px; }}
  .opt.on {{ border-color: #2f7d3b; background: #eef8f0; }}
  .opt.on .tag {{ color: #2f7d3b; font-weight: 700; }}
  .opt.x {{ border-style: dashed; }}
  .opt.x.on {{ border-color: #b06a10; background: #fdf4e6; }}
  .opt.x.on .tag {{ color: #b06a10; }}
  .other {{ margin-top: 6px; font-size: 12px; }}
  .other summary {{ color: #777; cursor: pointer; }}
  .other input {{ width: 100%; box-sizing: border-box; margin-top: 6px;
                  font: 14px ui-monospace, monospace; padding: 6px 8px;
                  border: 2px solid #ccc; border-radius: 6px; }}
  .parsed {{ font-size: 12px; color: #555; margin-top: 4px; }}
  .parsed.bad {{ color: #b03030; }}
  #out {{ position: sticky; bottom: 0; background: #111; color: #fff;
          padding: 12px 16px; margin-top: 24px; border-radius: 8px; }}
  #out textarea {{ width: 100%; box-sizing: border-box; height: 74px;
                   font: 13px ui-monospace, monospace; margin-top: 8px;
                   border-radius: 6px; border: 0; padding: 8px; }}
  #count {{ font-size: 14px; }}
</style>
<h1>人工裁定：第 {lo}–{hi} 张（共 {total} 张，本页 {n} 张）</h1>
<div class="intro">
  <p>两个检测器在这些图上分歧，裁切重问也没能解决。每张图三选一：</p>
  <p><strong>Q</strong> 千问的列表对　·　<strong>I</strong> InternVL 的列表对　·
     <strong>X</strong> 这张图里有东西不是任何东西（两个物体粘成一块、或形状根本认不出来）</p>
  <p><strong>X 不是「跳过」。</strong>它自己就是一个结论：那东西无论是什么，都不是
     prompt 要的东西，所以这张图算生成失败、计入 p；只是不能拿来出题——
     「你画了几个梨」在候选是半个梨时没有正确答案。整张图都读不了也选 X。</p>
  <p>图画得丑、但东西都认得出来 —— 不选 X，按哪个列表对选。</p>
  <p>不要把桌面、背景、阴影、倒影算作物体（盘子和碗算）。红字是两个模型具体吵在哪里，
     但请看整个列表 —— 你的选择替掉的是整份列表，不只是仲裁那一处。</p>
  <p>不显示 prompt，也不显示这张图本来该画什么——知道「要的是两个梨」会让人看出两个梨。</p>
  <p>两个列表都不对、但图能读 —— 展开卡片下方的「都不对」自己写，
     格式 <code>2 blue mug, white plate</code>。</p>
  <p>选完后把最下方框里的那一行发回聊天窗口即可。</p>
</div>
{cards}
<div id="out">
  <span id="count"></span>
  <textarea readonly id="codes"></textarea>
</div>
<script>
const PART = "{part}";
const cards = Array.from(document.querySelectorAll(".card"));

function parseList(text) {{
  const s = text.trim();
  if (!s) return null;
  if (s.toLowerCase() === "none") return [];
  const out = [];
  for (const chunk of s.split(",")) {{
    const words = chunk.trim().split(/\\s+/).filter(Boolean);
    if (!words.length) continue;
    let count = 1;
    if (/^[0-9]+$/.test(words[0])) count = parseInt(words.shift(), 10);
    if (words.length < 2) return "要颜色加名词：" + chunk.trim();
    if (count < 1 || count > 12) return "个数不对：" + chunk.trim();
    const noun = words.pop();
    for (let i = 0; i < count; i++)
      out.push({{object: noun, color: words.join(" ")}});
  }}
  return out.length ? out : "没读懂";
}}

function choose(card, code) {{
  card.dataset.choice = code;
  card.querySelectorAll(".opt").forEach(function (b) {{
    b.classList.toggle("on", b.dataset.code === code);
  }});
  refresh();
}}

function refresh() {{
  const bits = [];
  let done = 0;
  for (const card of cards) {{
    const n = card.dataset.num;
    const box = card.querySelector(".other input");
    const parsed = card.querySelector(".parsed");
    const typed = box.value.trim() ? parseList(box.value) : null;
    if (typed !== null && typeof typed !== "string") {{
      card.querySelectorAll(".opt").forEach(function (b) {{
        b.classList.remove("on");
      }});
      card.dataset.choice = "O";
      parsed.textContent = "→ " + box.value.trim();
      parsed.className = "parsed";
      // Quoted, because a typed list holds spaces and commas while the code
      // line is space-separated. Unquoted, `4=2 blue mug, white plate` cannot
      // be told from `4=2 blue mug` followed by a card called `white`.
      bits.push(n + '="' + box.value.trim().replace(/\\s+/g, " ") + '"');
      done++;
      continue;
    }}
    parsed.textContent = typeof typed === "string" ? typed : "";
    parsed.className = typeof typed === "string" ? "parsed bad" : "parsed";
    if (card.dataset.choice && card.dataset.choice !== "O") {{
      bits.push(n + card.dataset.choice);
      done++;
    }}
  }}
  document.getElementById("count").textContent =
      "已选 " + done + " / " + cards.length +
      (done === cards.length ? " — 全部选完，把下面这行发给我" : "");
  document.getElementById("codes").value = PART + ": " + bits.join(" ");
}}

for (const card of cards) {{
  card.querySelectorAll(".opt").forEach(function (b) {{
    b.addEventListener("click", function () {{
      card.querySelector(".other input").value = "";
      choose(card, b.dataset.code);
    }});
  }});
  card.querySelector(".other input").addEventListener("input", refresh);
}}
refresh();
</script>
"""

CARD = """<div class="card" data-num="{index}" data-path="{path}">
  <div class="head"><span class="num">{index}</span><span class="id">{name}</span></div>
  <div class="body">
    <div><img src="{src}" alt=""></div>
    <div>
      <div class="dispute">分歧：{disputed}</div>
      <div class="opts">
        {options}
        <button class="opt x" data-code="X">
          <span class="tag">X</span>
          <span class="list">有东西不是任何东西（粘连 / 认不出）</span>
        </button>
      </div>
      <details class="other">
        <summary>都不对，我自己写</summary>
        <input placeholder="2 blue mug, white plate" spellcheck="false">
        <div class="parsed"></div>
      </details>
    </div>
  </div>
</div>
"""

OPTION = """<button class="opt" data-code="{code}">
          <span class="tag">{code} · {label}</span>
          <span class="list">{list}</span>
        </button>"""


def _counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        key = f"{item.get('color') or '?'} {item.get('object')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fmt(detections: list[dict[str, Any]]) -> str:
    counts = _counts(detections)
    if not counts:
        return "<em>什么都没有</em>"
    return "，".join(
        f"{n}× {html.escape(k)}" if n > 1 else html.escape(k)
        for k, n in sorted(counts.items())
    )


def _thumb(path: Path, max_width: int, quality: int) -> str:
    """A display-only JPEG, inlined. Never written back over the corpus."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, help="defaults to the run directory")
    parser.add_argument("--stem", default="review")
    parser.add_argument("--primary", default="qwen3vl")
    parser.add_argument("--secondary", default="internvl")
    parser.add_argument("--primary-code", default="Q")
    parser.add_argument("--secondary-code", default="I")
    parser.add_argument("--primary-label", default="千问")
    parser.add_argument("--secondary-label", default="InternVL")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=0,
                        help="images per page; 0 puts them all on one. Ten "
                             "inlined JPEGs is about 400 KB, which opens "
                             "anywhere; all 42 as PNGs was 10.6 MB, which did "
                             "not open at all.")
    parser.add_argument("--inline-images", action="store_true",
                        help="re-encode each image small and embed it, so the "
                             "page is one self-contained file")
    parser.add_argument("--max-width", type=int, default=360)
    parser.add_argument("--quality", type=int, default=78)
    parser.add_argument("--seed", default="review",
                        help="seeds which detector's list is shown first")
    args = parser.parse_args()
    out_dir = (args.out_dir or args.run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    rows = pending[: args.limit]
    cards: list[str] = []
    kept: list[str] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        try:
            src = (_thumb(image_path, args.max_width, args.quality)
                   if args.inline_images
                   else html.escape(image_path.as_posix(), quote=True))
        except OSError:
            continue
        disputed = "，".join(
            f"{d.get('color') or '?'} {d.get('object')}"
            for d in row.get("disputed", ())
        ) or "n/a"
        options = [
            OPTION.format(code=args.primary_code, label=args.primary_label,
                          list=_fmt(primary.get(row["image_path"], []))),
            OPTION.format(code=args.secondary_code, label=args.secondary_label,
                          list=_fmt(secondary.get(row["image_path"], []))),
        ]
        # Which detector is on top is decided per card. Both buttons are
        # labelled, so picking one by name is still a single click; what this
        # removes is the reviewer who works down forty cards and takes whatever
        # is first.
        if random.Random(f"{args.seed}:{row['image_path']}").random() < 0.5:
            options.reverse()
        cards.append(CARD.format(
            src=src,
            path=html.escape(row["image_path"], quote=True),
            index=index,
            name=html.escape(image_path.name),
            disputed=html.escape(disputed),
            options="\n        ".join(options),
        ))
        kept.append(row["image_path"])

    size = args.chunk if args.chunk > 0 else len(cards)
    written = []
    for start in range(0, len(cards), size):
        group = cards[start:start + size]
        part = (f"{args.stem}{start // size + 1}" if args.chunk > 0
                else args.stem)
        out = out_dir / f"{part}.html"
        out.write_text(PAGE.format(
            cards="\n".join(group),
            n=len(group),
            lo=start + 1,
            hi=start + len(group),
            total=len(cards),
            part=part,
        ), encoding="utf-8")
        written.append(out)
        print(f"{out.name}  {len(group)} images  {out.stat().st_size // 1024} KB")

    # The pages carry card numbers, not paths. This is what turns "3X 4Q" back
    # into image paths, and it is written next to the pages so the mapping
    # cannot drift from the sheet the reviewer actually saw.
    index_path = out_dir / f"{args.stem}.index.json"
    index_path.write_text(json.dumps({
        "run": str(args.run),
        "chunk": size,
        "codes": {args.primary_code: args.primary,
                  args.secondary_code: args.secondary,
                  "X": "unnameable",
                  "O": "typed by the reviewer"},
        "pages": [p.name for p in written],
        "images": kept,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{index_path.name}  maps card numbers back to image paths")
    print(f"adjudication rate: {len(pending) / len(verified):.1%} "
          f"(locked criterion: report it, must be <= 10%)")


if __name__ == "__main__":
    main()
