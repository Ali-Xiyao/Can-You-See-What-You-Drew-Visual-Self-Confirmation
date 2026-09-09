"""Freeze a small independent-label request; do not run models or invent labels."""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/v4/decoupling-pilot-20260906"
SOURCES = {}


def load(path, jsonl=False):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)


def main():
    cases, pairs = {}, []
    for index in (1, 2):
        directory = RUN / f"rounds/round-{index:03d}"
        selection = load(directory / "selection.json")
        observations = load(directory / "observations/naive.jsonl", True)
        # Only prompt/spec metadata is reused from Gold; no image verdict is read.
        manifest = load(directory / "ladder/rfo_gold/manifest.jsonl", True)
        obs = {row["candidate_id"]: row for row in observations}
        specs = {}
        for row in manifest:
            if row["spec_id"] in specs:
                assert specs[row["spec_id"]] == row["spec"]
            specs[row["spec_id"]] = row["spec"]
        for decision in sorted(selection["decisions"]["naive"], key=lambda row: row["prompt_id"]):
            pid = decision["prompt_id"]
            candidates = sorted(decision["candidate_pool_ids"])
            assert len(candidates) == len(set(candidates)) == 8
            selected = decision["selected_candidate_id"]
            assert selected in candidates
            seed_text = f"side-selection-paired-control-v1:20260906:round{index}:{pid}"
            control = random.Random(seed_text).choice(candidates)
            pair = {"round": index, "selector_optimizer_step": index * 8,
                    "prompt_id": pid, "entered_training": pid in selection["paired_prompt_ids"],
                    "n_candidates": len(candidates), "selected_candidate_id": selected,
                    "uniform_control_candidate_id": control, "control_seed_rule_input": seed_text,
                    "same_candidate": selected == control, "labels": {},
                    "gain_if_same_candidate": 0 if selected == control else None}
            if selected != control:
                for role, cid in (("selected", selected), ("uniform_control", control)):
                    observation = obs[cid]
                    assert observation["prompt_id"] == pid
                    path = Path(observation["image_path"])
                    assert path.exists()
                    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    SOURCES[str(path)] = source_hash
                    case_id = hashlib.sha256(f"selection-label-v1:{index}:{pid}:{cid}".encode()).hexdigest()[:20]
                    assert case_id not in cases
                    seed, candidate_index = map(int, cid.rsplit("-", 2)[-2:])
                    cases[case_id] = {"review_id": case_id, "spec_id": pid,
                                      "candidate_index": candidate_index, "seed": seed,
                                      "prompt": specs[pid]["prompt"], "image_path": str(path),
                                      "spec": specs[pid], "image_file_sha256": source_hash,
                                      "recorded_rgb_sha256": observation["observation"]["rgb_sha256"]}
                    pair["labels"][role] = {"review_id": case_id, "label": None}
            pairs.append(pair)
    public = list(cases.values())
    random.Random("side-selection-review-display-order-v1:20260906").shuffle(public)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "prepared_only_no_observer_started_no_labels_assigned",
               "population": "Naive candidates in completed training rounds 1 and 2; scoring checkpoints 8 and 16",
               "n_pools": len(pairs), "n_entered_training": sum(p["entered_training"] for p in pairs),
               "n_same_candidate_zero_gain": sum(p["same_candidate"] for p in pairs),
               "n_images_requiring_independent_labels": len(public),
               "estimator": "Equal-prompt mean of correctness(actual selection) minus correctness(one uniformly sampled same-pool control). Same-candidate pairs have exactly zero gain even without labels.",
               "interpretation": "A sampled random-policy comparison, not the exact finite-pool random expectation. Adds control-sampling variance. Rounds use different prompts and cannot yield a paired training-time causal effect.",
               "blinding_scope": "Public request omits selected/control roles and model scores. Original source paths remain visible; do not claim complete blinding.",
               "execution_boundary": "Preparation only. Do not compete with the active main GPU schedule. Independent detector/human results must carry their own provenance; existing Gold image labels are not transferred.",
               "input_sha256": SOURCES, "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert len(pairs) == 24
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in SOURCES.items())
    (OUT / "followup-label-request.public.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public), encoding="utf-8")
    (OUT / "followup-label-pairs.private.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "followup-label-request.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("input_sha256",)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
