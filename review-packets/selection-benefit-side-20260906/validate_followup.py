"""Validate a prepared label request without assigning any image labels."""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    public_path = OUT / "followup-label-request.public.jsonl"
    private_path = OUT / "followup-label-pairs.private.json"
    summary_path = OUT / "followup-label-request.summary.json"
    public = [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines() if line]
    pairs = json.loads(private_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lookup = {row["review_id"]: row for row in public}
    assert len(lookup) == len(public) == 44
    assert len(pairs) == 24
    assert not any(set(row) & {"role", "score", "atomic_score", "selected_candidate_id"} for row in public)
    assert len({(pair["round"], pair["prompt_id"]) for pair in pairs}) == 24
    seen = set()
    for pair in pairs:
        directory = ROOT / "runs/v4/decoupling-pilot-20260906/rounds" / f"round-{pair['round']:03d}"
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        decisions = {d["prompt_id"]: d for d in selection["decisions"]["naive"]}
        decision = decisions[pair["prompt_id"]]
        assert pair["selected_candidate_id"] == decision["selected_candidate_id"]
        assert pair["entered_training"] == (pair["prompt_id"] in selection["paired_prompt_ids"])
        assert pair["uniform_control_candidate_id"] == random.Random(pair["control_seed_rule_input"]).choice(sorted(decision["candidate_pool_ids"]))
        assert pair["same_candidate"] == (pair["selected_candidate_id"] == pair["uniform_control_candidate_id"])
        if pair["same_candidate"]:
            assert pair["labels"] == {} and pair["gain_if_same_candidate"] == 0
            continue
        assert set(pair["labels"]) == {"selected", "uniform_control"}
        for role, item in pair["labels"].items():
            assert item["label"] is None
            rid = item["review_id"]
            assert rid not in seen
            seen.add(rid)
            row = lookup[rid]
            assert row["spec_id"] == pair["prompt_id"]
            assert row["spec"]["spec_id"] == row["spec_id"] and row["spec"]["prompt"] == row["prompt"]
            path = Path(row["image_path"])
            assert path.stem == pair[role + "_candidate_id"]
            assert digest(path) == row["image_file_sha256"]
            # Match src/selfsight/utils/hashing.py: dimensions are part of RGB identity.
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                header = f"{rgb.width}x{rgb.height}:RGB:".encode("ascii")
                assert hashlib.sha256(header + rgb.tobytes()).hexdigest() == row["recorded_rgb_sha256"]
    assert seen == set(lookup)
    assert sum(p["entered_training"] for p in pairs) == 18
    assert sum(p["same_candidate"] for p in pairs) == 2
    for path, expected in summary["input_sha256"].items():
        assert digest(Path(path)) == expected
    assert digest(OUT / "prepare_followup.py") == summary["preparation_script_sha256"]
    result = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_preparation_integrity_only",
        "n_pools": len(pairs), "n_entered_training": 18,
        "n_same_candidate_exact_zero": 2,
        "n_images_checked_file_and_decoded_rgb": len(public), "n_labels_assigned": 0,
        "checks": [
            "Source selection and entered-training membership",
            "Deterministic uniform control resampling",
            "Role mapping and unique review IDs",
            "Original spec/prompt consistency",
            "PNG SHA256 and dimension-prefixed decoded RGB SHA256",
            "All input hashes unchanged",
            "Public request contains no role or score fields",
        ],
        "limitations": [
            "No detector or human labels supplied yet.",
            "Original paths are visible, so blinding is limited.",
            "Sampled controls add sampling variation; this is not exact full-pool expectation.",
            "Rounds use different prompts, so a round contrast alone is not a paired training-time effect.",
        ],
        "output_sha256": {p.name: digest(p) for p in (public_path, private_path, summary_path)},
        "validation_script_sha256": digest(Path(__file__)),
    }
    (OUT / "followup-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
