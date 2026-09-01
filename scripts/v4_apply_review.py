"""Turn the reviewer's answer codes back into `human_labels.jsonl`.

The sheets built by `v4_human_review.py` carry card numbers, not paths, and the
reviewer sends back one line per page:

    A1: 1Q 2I 3X 4="2 blue mug, white plate" 5Q ...

    Q  take the qwen3vl list for that image
    I  take the internvl list
    X  something in the picture is not any object -- written as the single
       detection `unnameable`, which scores the image a miss and bars it from
       supplying trials (v4.spec.UNNAMEABLE)
    =  a typed list, when both detectors were wrong and the picture is readable

    python scripts/v4_apply_review.py --index runs/v4/main/review.index.json \\
        --codes 'A1: 1Q 2I 3X' --codes 'A2: 11I 12Q'

Q and I copy the detector's row wholesale, bboxes included, because the reviewer
was choosing between those lists and a list without boxes cannot carry a spatial
question afterwards. A typed list has no boxes and is used only for the verdict.

The card number is global across the pages of one run, so the page prefix is a
label and nothing is keyed on it. What is checked is that every number given
exists, that no number is answered twice, and that the count matches -- a
mistyped digit silently relabelling the wrong image is the failure this file is
built to prevent, so all three are errors rather than warnings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TOKEN = re.compile(r'(\d+)(?:([A-Za-z])|="([^"]*)")')


def parse_codes(text: str) -> dict[int, tuple[str, str]]:
    """{card number: (code, payload)} from any number of pasted lines."""
    answers: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        body = line.split(":", 1)[1] if ":" in line else line
        position = 0
        for match in TOKEN.finditer(body):
            if body[position:match.start()].strip():
                raise SystemExit(
                    f"could not read {body[position:match.start()].strip()!r} "
                    f"in: {line.strip()}")
            position = match.end()
            number = int(match.group(1))
            if number in answers:
                raise SystemExit(f"card {number} answered twice")
            if match.group(2):
                answers[number] = (match.group(2).upper(), "")
            else:
                answers[number] = ("O", match.group(3))
        if body[position:].strip():
            raise SystemExit(
                f"could not read {body[position:].strip()!r} in: {line.strip()}")
    return answers


def parse_list(text: str) -> list[dict[str, Any]]:
    """`2 blue mug, white plate` -> two mugs and a plate, same as the page."""
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []
    out: list[dict[str, Any]] = []
    for chunk in stripped.split(","):
        words = chunk.split()
        if not words:
            continue
        count = 1
        if words[0].isdigit():
            count = int(words.pop(0))
        if len(words) < 2:
            raise SystemExit(f"needs a colour and a noun: {chunk.strip()!r}")
        noun = words[-1]
        colour = " ".join(words[:-1])
        out.extend({"object": noun, "color": colour} for _ in range(count))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--codes", action="append", default=[],
                        help="repeat per page; omit to read them from stdin")
    parser.add_argument("--out", type=Path,
                        help="defaults to human_labels.jsonl in the run")
    parser.add_argument("--partial", action="store_true",
                        help="write the cards that were answered instead of "
                             "insisting on all of them")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from selfsight.v4.detectors import cached_detections

    index = json.loads(args.index.read_text(encoding="utf-8"))
    run = Path(index["run"])
    images: list[str] = index["images"]
    detectors = {
        code: cached_detections(run / f"detections.{name}.jsonl")
        for code, name in index["codes"].items()
        if name not in {"unnameable", "typed by the reviewer"}
    }

    text = "\n".join(args.codes) if args.codes else sys.stdin.read()
    answers = parse_codes(text)
    unknown = sorted(n for n in answers if not 1 <= n <= len(images))
    if unknown:
        raise SystemExit(f"no such card: {unknown} (there are {len(images)})")
    missing = sorted(set(range(1, len(images) + 1)) - set(answers))
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} cards unanswered: {missing}")

    out = args.out or run / "human_labels.jsonl"
    lines = []
    tally: dict[str, int] = {}
    for number in sorted(answers):
        code, payload = answers[number]
        image = images[number - 1]
        if code == "X":
            detections = [{"object": "unnameable", "color": ""}]
        elif code == "O":
            detections = parse_list(payload)
        elif code in detectors:
            detections = detectors[code].get(image, [])
        else:
            raise SystemExit(f"card {number}: unknown code {code!r}")
        tally[code] = tally.get(code, 0) + 1
        lines.append(json.dumps(
            {"image_path": image, "index": number, "code": code,
             "detections": detections}, ensure_ascii=False))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}  {len(lines)} labels")
    for code, count in sorted(tally.items()):
        print(f"  {code}  {count:>3}  {index['codes'].get(code, '?')}")
    if missing:
        print(f"  {len(missing)} still unanswered: {missing}")


if __name__ == "__main__":
    main()
