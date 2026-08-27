from __future__ import annotations

import csv
from pathlib import Path

from selfsight.data.manifest import render_split_manifest
from selfsight.data.manual_audit import export_manual_audit, score_manual_audit
from selfsight.schemas import Color, QuestionFamily, SceneObject, SceneSpec, Shape, Size


def _scene(index: int, family: QuestionFamily) -> SceneSpec:
    scene_id = f"manual-{family.value}-{index:03d}"
    return SceneSpec(
        scene_id=scene_id,
        split="audit",
        family=family,
        template_id=f"manual-template-{family.value}-{index:03d}",
        prompt="A red circle.",
        objects=(SceneObject("object-0", Shape.CIRCLE, Color.RED, Size.SMALL, (256, 256)),),
        metadata={"target_shape": "circle", "target_color": "red", "positive": True},
    )


def test_manual_audit_export_and_score(tmp_path: Path) -> None:
    scenes = [_scene(index, QuestionFamily.EXISTENCE) for index in range(2)]
    manifest = render_split_manifest(scenes, tmp_path / "data", "audit")
    packet = export_manual_audit(manifest, tmp_path / "packet", per_family=2, seed=7)
    assert packet["total"] == 2
    review_path = Path(packet["review_csv"])
    rows = list(csv.DictReader(review_path.open("r", encoding="utf-8-sig", newline="")))
    for row in rows:
        row["human_answer"] = "yes"
        row["parseable_yes_no"] = "yes"
        row["reviewer_id"] = "test-reviewer"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = score_manual_audit(review_path, packet["answer_key"])
    assert report["manual_reference_gate_pass"] is True
    assert report["verifier_human_agreement"] == 1.0
