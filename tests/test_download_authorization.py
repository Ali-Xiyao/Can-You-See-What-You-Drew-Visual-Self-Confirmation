import csv
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.download_models import (
    _run_resumable_command,
    _validate_fallback_download_authorization,
)
from selfsight.data.readiness_precision import score_generated_precision_audit
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json


def _decision(tmp_path: Path, *, human_stop: bool = False) -> Path:
    evidence = {}
    for label in ("backbone_config", "readiness_config", "canary", "reference"):
        path = tmp_path / f"{label}.json"
        if human_stop and label == "readiness_config":
            path.write_text(
                Path("configs/readiness_v2.2.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            path.write_text(label, encoding="utf-8")
        evidence[label] = {"path": str(path), "sha256": sha256_file(path)}
    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(image_path)
    rows_path = tmp_path / "generated-rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-0",
                "image_path": str(image_path),
                "rgb_sha256": rgb_sha256(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generated_path = tmp_path / "generated.json"
    generated_path.write_text(
        json.dumps(
            {
                "rows": str(rows_path),
                "rows_sha256": sha256_file(rows_path),
                "candidates": 1,
                "unique_candidate_ids": 1,
                "unique_image_paths": 1,
            }
        ),
        encoding="utf-8",
    )
    evidence["generated"] = {
        "path": str(generated_path),
        "sha256": sha256_file(generated_path),
    }
    human_evidence = None
    if human_stop:
        families = ("existence", "count", "color", "size", "spatial", "binding")
        answers = {
            "existence": "yes",
            "count": "1",
            "color": "red",
            "size": "small",
            "spatial": "yes",
            "binding": "red",
        }
        key_rows = []
        annotations = []
        for index, family in enumerate(families):
            audit_id = f"a{index}"
            key_rows.append(
                {
                    "audit_id": audit_id,
                    "family": family,
                    "primary_question": {
                        "question_id": f"q{index}",
                        "atom_id": audit_id,
                        "family": family,
                        "text": "Answer from visible pixels.",
                        "expected_answer": answers[family],
                        "question_format": "open",
                        "choices": [],
                        "choice_order_seed": 0,
                    },
                    "verifier_answer": answers[family],
                }
            )
            annotations.append(
                {
                    "audit_id": audit_id,
                    "human_answer": "no" if index == 0 else answers[family],
                    "parseable_yes_no": "yes",
                    "reviewer_id": "reviewer",
                }
            )
        review_path = tmp_path / "review.csv"
        with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(annotations[0]))
            writer.writeheader()
            writer.writerows(annotations)
        key_path = tmp_path / "answer-key.json"
        key_path.write_text(
            json.dumps(
                {
                    "model_id": "showlab/show-o2-1.5B-HQ",
                    "revision": "revision",
                    "source_revision": "source-revision",
                    "dependency_revisions": {"dependency": "revision"},
                    "rows": key_rows,
                    "selection_digest": sha256_json(key_rows),
                }
            ),
            encoding="utf-8",
        )
        human_path = tmp_path / "human.json"
        human_report = score_generated_precision_audit(
            review_path,
            key_path,
            families=list(families),
            threshold=0.95,
        )
        human_path.write_text(json.dumps(human_report), encoding="utf-8")
        human_evidence = {"path": str(human_path), "sha256": sha256_file(human_path)}
    evidence.update({"human": human_evidence, "lora": None, "predecessor": None})
    report = {
        "gate": "minus_2_joint_readiness",
        "decision_mode": (
            "stop_after_human_before_a4" if human_stop else "upstream_stop_before_human_and_a4"
        ),
        "model_id": "showlab/show-o2-1.5B-HQ" if human_stop else "showlab/show-o2-1.5B",
        "candidate_rank": 2 if human_stop else 1,
        "passed": False,
        "checks": {
            "minus_2a_unified_functionality": False,
            "minus_2b_reference_observation": True,
            "minus_2c_generated_measurability": False,
            "minus_2d_joint_families": False,
        },
        "skipped_by_stop_rule": (
            ["a4_lora_backward_resume"]
            if human_stop
            else ["blind_human_precision", "a4_lora_backward_resume"]
        ),
        "evidence": evidence,
        "fallback": {
            "next_model_id": "showlab/show-o2-7B" if human_stop else "showlab/show-o2-1.5B-HQ"
        },
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_hq_download_requires_exact_hashed_red_predecessor(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    authorization = _validate_fallback_download_authorization("readiness_fallback_hq", decision)
    assert authorization["authorized_model_id"] == "showlab/show-o2-1.5B-HQ"
    assert authorization["sha256"] == sha256_file(decision)


def test_fallback_download_rejects_missing_or_tampered_evidence(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="predecessor-decision"):
        _validate_fallback_download_authorization("readiness_fallback_hq", None)
    decision = _decision(tmp_path)
    (tmp_path / "generated.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)


def test_fallback_download_rejects_ladder_skip(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    with pytest.raises(RuntimeError, match="candidate rank"):
        _validate_fallback_download_authorization("readiness_fallback_7b", decision)


def test_fallback_download_rejects_incomplete_evidence_set(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    report = json.loads(decision.read_text(encoding="utf-8"))
    del report["evidence"]["reference"]
    decision.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence set"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)


def test_fallback_download_rejects_nested_a3_rows_tamper(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    with (tmp_path / "generated-rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="rows SHA-256 mismatch"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)


def test_7b_download_accepts_hashed_human_stop_predecessor(tmp_path: Path) -> None:
    decision = _decision(tmp_path, human_stop=True)
    authorization = _validate_fallback_download_authorization("readiness_fallback_7b", decision)
    assert authorization["authorized_model_id"] == "showlab/show-o2-7B"


def test_resumable_command_restarts_transient_process_failures(monkeypatch) -> None:
    attempts = []
    delays = []

    def fake_run(command, *, check):
        attempts.append((command, check))
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("scripts.download_models.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.download_models.sleep", delays.append)
    _run_resumable_command(
        ["aria2c", "--continue=true"],
        label="model/shard",
        max_attempts=3,
        base_delay_seconds=2,
    )

    assert attempts == [
        (["aria2c", "--continue=true"], True),
        (["aria2c", "--continue=true"], True),
        (["aria2c", "--continue=true"], True),
    ]
    assert delays == [2, 4]
