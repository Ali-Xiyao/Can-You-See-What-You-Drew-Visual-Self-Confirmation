"""Deterministic, blinded packet and scoring for the manual reference-verifier audit."""

from __future__ import annotations

import csv
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from selfsight.data.questions import normalize_answer
from selfsight.data.verifier import verify_image
from selfsight.schemas import Atom, AtomicQuestion, QuestionFormat
from selfsight.utils.hashing import sha256_json
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def _sample_stratified(records: list[dict[str, Any]], per_family: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["atom"]["family"])].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        values = sorted(grouped[family], key=lambda item: item["scene"]["scene_id"])
        if len(values) < per_family:
            raise ValueError(f"Need {per_family} records for {family}; found {len(values)}")
        selected.extend(rng.sample(values, per_family))
    rng.shuffle(selected)
    return selected


def _contact_sheets(rows: list[dict[str, str]], output: Path, rows_per_sheet: int = 4) -> list[str]:
    sheet_paths: list[str] = []
    font = ImageFont.load_default(size=20)
    for start in range(0, len(rows), rows_per_sheet):
        page = Image.new("RGB", (1500, rows_per_sheet * 650), "white")
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(rows[start : start + rows_per_sheet]):
            top = offset * 650
            with Image.open(row["image_path"]) as source:
                rgb = source.convert("RGB")
                rgb.thumbnail((512, 512))
                page.paste(rgb, (30, top + 70))
            draw.text((30, top + 20), row["audit_id"], fill="black", font=font)
            wrapped = textwrap.wrap(row["question"], width=72)
            draw.multiline_text((580, top + 90), "\n".join(wrapped), fill="black", font=font, spacing=10)
            draw.text(
                (580, top + 310),
                "Record only what is visible. Do not infer the generating prompt.",
                fill=(80, 80, 80),
                font=font,
            )
        page_number = start // rows_per_sheet + 1
        path = output / "sheets" / f"manual-audit-{page_number:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.save(path, format="PNG", optimize=True)
        sheet_paths.append(str(path.resolve()))
    return sheet_paths


def export_manual_audit(
    manifest: str | Path,
    output: str | Path,
    *,
    per_family: int = 20,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Create a blinded CSV/contact-sheet packet and a separate immutable answer key."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = _sample_stratified(list(read_jsonl(manifest)), per_family, seed)
    review_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        atom = Atom.from_dict(record["atom"])
        question = AtomicQuestion.from_dict(record["questions"][0])
        verifier = verify_image(record["reference_image"], [atom]).answers[atom.atom_id]
        audit_id = f"manual-{index:03d}"
        review_rows.append(
            {
                "audit_id": audit_id,
                "family": atom.family.value,
                "image_path": str(Path(record["reference_image"]).resolve()),
                "question": question.text,
                "human_answer": "",
                "parseable_yes_no": "",
                "reviewer_id": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "scene_id": record["scene"]["scene_id"],
                "atom": record["atom"],
                "question": record["questions"][0],
                "expected_answer": atom.answer,
                "verifier_answer": verifier,
            }
        )
    review_path = output / "review_blinded.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    key = {
        "schema_version": 1,
        "seed": seed,
        "per_family": per_family,
        "total": len(key_rows),
        "rows": key_rows,
    }
    key["selection_digest"] = sha256_json(key_rows)
    key_path = atomic_write_json(output / "answer_key.json", key)
    sheets = _contact_sheets(review_rows, output)
    report = {
        "schema_version": 1,
        "status": "awaiting_blinded_human_annotations",
        "review_csv": str(review_path.resolve()),
        "answer_key": str(key_path.resolve()),
        "contact_sheets": sheets,
        "total": len(review_rows),
        "family_counts": {
            family: sum(row["family"] == family for row in review_rows)
            for family in sorted({row["family"] for row in review_rows})
        },
        "selection_digest": key["selection_digest"],
    }
    atomic_write_json(output / "packet_manifest.json", report)
    return report


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def score_manual_audit(
    review_csv: str | Path,
    answer_key: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Score only complete blinded annotations; 98% agreement is the registered pass rule."""

    import json

    key = json.loads(Path(answer_key).read_text(encoding="utf-8"))
    if sha256_json(key["rows"]) != key["selection_digest"]:
        raise ValueError("Manual audit answer key digest mismatch")
    keyed = {row["audit_id"]: row for row in key["rows"]}
    with Path(review_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["audit_id"] for row in rows} != set(keyed):
        raise ValueError("Review CSV IDs do not exactly match the answer key")
    family_total: dict[str, int] = defaultdict(int)
    family_agree: dict[str, int] = defaultdict(int)
    incomplete: list[str] = []
    disagreements: list[dict[str, Any]] = []
    complete = agree = parseable = 0
    for row in rows:
        parseable_value = _parse_bool(row.get("parseable_yes_no", ""))
        raw_answer = row.get("human_answer", "").strip()
        reviewer_id = row.get("reviewer_id", "").strip()
        if parseable_value is None or not raw_answer or not reviewer_id:
            incomplete.append(row["audit_id"])
            continue
        target = keyed[row["audit_id"]]
        question = AtomicQuestion.from_dict(target["question"])
        if question.question_format != QuestionFormat.OPEN:
            raise ValueError("Manual audit answer key must contain open questions")
        normalized = normalize_answer(raw_answer, question) if parseable_value else None
        matched = normalized == target["verifier_answer"]
        complete += 1
        parseable += int(parseable_value)
        agree += int(matched)
        family = str(target["atom"]["family"])
        family_total[family] += 1
        family_agree[family] += int(matched)
        if not matched:
            disagreements.append(
                {
                    "audit_id": row["audit_id"],
                    "family": family,
                    "human_raw": raw_answer,
                    "human_normalized": normalized,
                    "verifier_answer": target["verifier_answer"],
                    "parseable": parseable_value,
                }
            )
    total = int(key["total"])
    agreement = agree / complete if complete else 0.0
    report = {
        "schema_version": 1,
        "registered_threshold": 0.98,
        "required_annotations": total,
        "complete_annotations": complete,
        "incomplete_ids": incomplete,
        "human_parseable_rate": parseable / complete if complete else 0.0,
        "verifier_human_agreement": agreement,
        "family_agreement": {
            family: family_agree[family] / count for family, count in family_total.items()
        },
        "disagreements": disagreements,
        "manual_reference_gate_pass": complete == total and agreement >= 0.98,
    }
    if output is not None:
        atomic_write_json(output, report)
    return report
