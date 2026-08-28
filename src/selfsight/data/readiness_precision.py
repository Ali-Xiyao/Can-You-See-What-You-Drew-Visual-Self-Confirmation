"""Blinded human audit packets for v2.2 generated-image verifier precision."""

from __future__ import annotations

import csv
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from selfsight.data.questions import normalize_answer
from selfsight.schemas import AtomicQuestion, QuestionFormat
from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def _contact_sheets(rows: list[dict[str, str]], output: Path, per_sheet: int = 4) -> list[str]:
    paths: list[str] = []
    font = ImageFont.load_default(size=20)
    for start in range(0, len(rows), per_sheet):
        page = Image.new("RGB", (1500, per_sheet * 650), "white")
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(rows[start : start + per_sheet]):
            top = offset * 650
            with Image.open(row["image_path"]) as source:
                rgb = source.convert("RGB")
                rgb.thumbnail((512, 512))
                page.paste(rgb, (30, top + 70))
            draw.text((30, top + 20), row["audit_id"], fill="black", font=font)
            question = "\n".join(textwrap.wrap(row["question"], width=72))
            draw.multiline_text((580, top + 90), question, fill="black", font=font, spacing=10)
            draw.text(
                (580, top + 310),
                "Record only visible pixels. The generating prompt is intentionally hidden.",
                fill=(80, 80, 80),
                font=font,
            )
        path = output / "sheets" / f"generated-precision-{start // per_sheet + 1:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.save(path, format="PNG", optimize=True)
        paths.append(str(path.resolve()))
    return paths


def export_generated_precision_audit(
    generated_report: str | Path,
    output: str | Path,
    *,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Export all verifier-answered K=1 cases without leaking intended scene information."""

    report_path = Path(generated_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_path = Path(str(report["rows"])).resolve()
    if sha256_file(rows_path) != report["rows_sha256"]:
        raise RuntimeError("Generated readiness rows SHA-256 mismatch")
    candidates = [
        row
        for row in read_jsonl(rows_path)
        if int(row["candidate_index"]) == 0 and bool(row["primary_answer_covered"])
    ]
    if not candidates:
        raise RuntimeError("No answered K=1 verifier cases are available for blind audit")
    rng = random.Random(seed)
    candidates.sort(key=lambda row: str(row["scene_id"]))
    rng.shuffle(candidates)

    review_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates, start=1):
        audit_id = f"generated-{index:03d}"
        question = AtomicQuestion.from_dict(row["primary_question"])
        review_rows.append(
            {
                "audit_id": audit_id,
                "image_path": str(Path(row["image_path"]).resolve()),
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
                "scene_id": row["scene_id"],
                "family": row["family"],
                "primary_question": row["primary_question"],
                "verifier_answer": row["primary_answer"],
                "image_rgb_sha256": row["rgb_sha256"],
            }
        )

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    review_path = output / "review_blinded.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    key = {
        "schema_version": 2,
        "stage": "minus_2c_blind_human_precision_key",
        "model_id": report["model_id"],
        "revision": report["revision"],
        "source_revision": report["source_revision"],
        "dependency_revisions": report["dependency_revisions"],
        "generated_report": str(report_path),
        "generated_report_sha256": sha256_file(report_path),
        "seed": seed,
        "selection_rule": "all verifier-answered candidate_index=0 cases",
        "rows": key_rows,
        "selection_digest": sha256_json(key_rows),
    }
    key_path = atomic_write_json(output / "answer_key.json", key)
    manifest = {
        "schema_version": 2,
        "status": "awaiting_blinded_human_annotations",
        "model_id": report["model_id"],
        "revision": report["revision"],
        "review_csv": str(review_path),
        "answer_key": str(key_path),
        "contact_sheets": _contact_sheets(review_rows, output),
        "total": len(review_rows),
        "family_counts": {
            family: sum(row["family"] == family for row in key_rows)
            for family in sorted({str(row["family"]) for row in key_rows})
        },
        "selection_digest": key["selection_digest"],
        "blind_fields_excluded": [
            "prompt",
            "expected_answer",
            "verifier_answer",
            "scene_graph",
            "generator_seed",
        ],
    }
    atomic_write_json(output / "packet_manifest.json", manifest)
    return manifest


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def score_generated_precision_audit(
    review_csv: str | Path,
    answer_key: str | Path,
    *,
    families: list[str],
    threshold: float = 0.95,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Score complete blind annotations; missing family evidence fails closed at precision zero."""

    review_path = Path(review_csv).resolve()
    key_path = Path(answer_key).resolve()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if sha256_json(key["rows"]) != key["selection_digest"]:
        raise RuntimeError("Generated precision answer-key digest mismatch")
    keyed = {str(row["audit_id"]): row for row in key["rows"]}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {str(row["audit_id"]) for row in rows} != set(keyed):
        raise RuntimeError("Blind review IDs do not exactly match the answer key")

    totals: dict[str, int] = defaultdict(int)
    agreements: dict[str, int] = defaultdict(int)
    incomplete: list[str] = []
    disagreements: list[dict[str, Any]] = []
    complete = agree = human_parseable = 0
    for row in rows:
        parseable = _parse_bool(row.get("parseable_yes_no", ""))
        raw = row.get("human_answer", "").strip()
        reviewer = row.get("reviewer_id", "").strip()
        if parseable is None or not raw or not reviewer:
            incomplete.append(str(row["audit_id"]))
            continue
        target = keyed[str(row["audit_id"])]
        question = AtomicQuestion.from_dict(target["primary_question"])
        if question.question_format != QuestionFormat.OPEN:
            raise RuntimeError("Generated precision packet must contain open questions")
        normalized = normalize_answer(raw, question) if parseable else None
        matched = normalized == target["verifier_answer"]
        family = str(target["family"])
        totals[family] += 1
        agreements[family] += int(matched)
        complete += 1
        agree += int(matched)
        human_parseable += int(parseable)
        if not matched:
            disagreements.append(
                {
                    "audit_id": row["audit_id"],
                    "family": family,
                    "human_normalized": normalized,
                    "verifier_answer": target["verifier_answer"],
                    "human_marked_parseable": parseable,
                }
            )
    required = len(keyed)
    family_precision = {
        family: agreements[family] / totals[family] if totals[family] else 0.0
        for family in families
    }
    overall = agree / complete if complete else 0.0
    report = {
        "schema_version": 2,
        "stage": "minus_2c_blind_human_precision",
        "model_id": key["model_id"],
        "revision": key["revision"],
        "source_revision": key["source_revision"],
        "dependency_revisions": key["dependency_revisions"],
        "review_csv": str(review_path),
        "review_csv_sha256": sha256_file(review_path),
        "answer_key": str(key_path),
        "answer_key_sha256": sha256_file(key_path),
        "selection_digest": key["selection_digest"],
        "blind": True,
        "required_annotations": required,
        "complete_annotations": complete,
        "incomplete_ids": incomplete,
        "human_parseable_rate": human_parseable / complete if complete else 0.0,
        "overall_precision": overall,
        "family_precision": family_precision,
        "family_audited_counts": {family: totals[family] for family in families},
        "disagreements": disagreements,
        "threshold": threshold,
        "passed": complete == required
        and overall >= threshold
        and all(totals[family] > 0 for family in families),
    }
    if output is not None:
        if Path(output).exists():
            raise FileExistsError(f"Refusing to overwrite blind precision report: {output}")
        atomic_write_json(output, report)
    return report
