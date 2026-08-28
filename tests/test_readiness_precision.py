import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_generated_precision import _assert_unique_candidate_artifacts
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.readiness import build_minimal_scenes
from selfsight.data.readiness_precision import (
    export_generated_precision_audit,
    score_generated_precision_audit,
)
from selfsight.schemas import QuestionFormat, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl


def test_generated_precision_packet_is_blind_and_scores(tmp_path: Path) -> None:
    rows = []
    families = []
    for scene in build_minimal_scenes("canary", per_family=1):
        families.append(scene.family.value)
        image_path = tmp_path / f"{scene.scene_id}.png"
        Image.new("RGB", (32, 32), "white").save(image_path)
        atom = build_primary_atom(scene)
        question = build_question(atom, QuestionFormat.OPEN)
        rows.append(
            {
                "scene_id": scene.scene_id,
                "family": scene.family.value,
                "candidate_index": 0,
                "image_path": str(image_path),
                "rgb_sha256": rgb_sha256(image_path),
                "primary_question": as_serializable(question),
                "primary_answer": atom.answer,
                "primary_answer_covered": True,
            }
        )
    rows_path = atomic_write_jsonl(tmp_path / "rows.jsonl", rows)
    generated_path = atomic_write_json(
        tmp_path / "generated.json",
        {
            "model_id": "showlab/show-o2-1.5B",
            "revision": "locked",
            "source_revision": "source-locked",
            "dependency_revisions": {"dependency": "locked"},
            "rows": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
        },
    )
    packet = export_generated_precision_audit(generated_path, tmp_path / "packet")
    review = Path(packet["review_csv"])
    text = review.read_text(encoding="utf-8-sig")
    assert "expected_answer" not in text
    assert "verifier_answer" not in text
    assert "prompt" not in text.lower()

    key = json.loads(Path(packet["answer_key"]).read_text(encoding="utf-8"))
    keyed = {row["audit_id"]: row for row in key["rows"]}
    with review.open("r", encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    for row in annotations:
        row["human_answer"] = keyed[row["audit_id"]]["verifier_answer"]
        row["parseable_yes_no"] = "yes"
        row["reviewer_id"] = "blind-reviewer"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    report = score_generated_precision_audit(
        review,
        packet["answer_key"],
        families=families,
        output=tmp_path / "human-report.json",
    )
    assert report["passed"]
    assert report["overall_precision"] == 1.0
    assert all(value == 1.0 for value in report["family_precision"].values())
    assert report["review_csv"] == str(review.resolve())
    assert report["review_csv_sha256"] == sha256_file(review)


def test_generated_precision_incomplete_annotations_fail(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    review.write_text(
        "audit_id,human_answer,parseable_yes_no,reviewer_id\na1,yes,,reviewer\n",
        encoding="utf-8-sig",
    )
    key_rows = [
        {
            "audit_id": "a1",
            "family": "existence",
            "primary_question": {
                "question_id": "q1",
                "atom_id": "a1",
                "family": "existence",
                "text": "Is there a square?",
                "expected_answer": "yes",
                "question_format": "open",
                "choices": [],
                "choice_order_seed": 0,
            },
            "verifier_answer": "yes",
        }
    ]
    from selfsight.utils.hashing import sha256_json

    key_path = atomic_write_json(
        tmp_path / "key.json",
        {
            "model_id": "model",
            "revision": "revision",
            "source_revision": "source-revision",
            "dependency_revisions": {"dependency": "revision"},
            "rows": key_rows,
            "selection_digest": sha256_json(key_rows),
        },
    )
    report = score_generated_precision_audit(review, key_path, families=["existence", "count"])
    assert not report["passed"]
    assert report["complete_annotations"] == 0
    assert report["family_precision"]["count"] == 0.0


def test_generated_precision_rejects_candidate_path_collisions(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "duplicate",
            "image_path": str(tmp_path / "same.png"),
            "rgb_sha256": "0" * 64,
        },
        {
            "candidate_id": "duplicate",
            "image_path": str(tmp_path / "same.png"),
            "rgb_sha256": "0" * 64,
        },
    ]
    with pytest.raises(RuntimeError, match="candidate identity collision"):
        _assert_unique_candidate_artifacts(rows)
