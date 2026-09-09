"""Describe ranking and ties behind fixed-bank step24-to-step32 choices."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SIDE = Path(__file__).resolve().parent
RUN = SIDE.parents[1] / "runs/v4/decoupling-pilot-20260906"
hashes = {}


def read(path):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


bank = read(RUN / "probe-bank/bank.json")
arms = {}
for arm in ("naive", "rfo_gold"):
    decisions = {}
    for step in (24, 32):
        directory = RUN / f"gradient-probes/{arm}/step-{step:05d}"
        selection, report = read(directory / "selection.json"), read(directory / "report.json")
        assert selection["fingerprint"] == report["fingerprint"]
        assert report["bank_fingerprint"] == bank["fingerprint"]
        decisions[step] = {r["prompt_id"]: r for r in selection["rows"]}
    rows = []
    for pool in bank["pools"]:
        pid = pool["prompt_id"]
        truth = {c["candidate_id"]: c["correct"] for c in pool["candidates"]}
        before, after = decisions[24][pid], decisions[32][pid]
        old_id, new_id = before["selected"]["naive"], after["selected"]["naive"]
        scores = after["scores"]["naive"]
        assert scores[new_id] == max(scores.values())
        best_correct = max(score for cid, score in scores.items() if truth[cid])
        best_wrong = max(score for cid, score in scores.items() if not truth[cid])
        label = ("correct_preferred" if best_correct > best_wrong else "wrong_preferred" if best_correct < best_wrong else "correct_wrong_top_tie")
        rows.append({"prompt_id": pid, "spec_id": pool["spec_id"], "selected24": old_id, "selected32": new_id,
                     "correct24": truth[old_id], "correct32": truth[new_id],
                     "changed_candidate": old_id != new_id, "ranking32": label,
                     "best_correct_score32": best_correct, "best_wrong_score32": best_wrong})
    arms[arm] = {"n_pools": len(rows), "changed_candidate_pools": sum(r["changed_candidate"] for r in rows),
                 "correct_to_wrong": sum(r["correct24"] and not r["correct32"] for r in rows),
                 "wrong_to_correct": sum(not r["correct24"] and r["correct32"] for r in rows),
                 "ranking_of_wrong_selections32": dict(Counter(r["ranking32"] for r in rows if not r["correct32"])),
                 "ranking_of_correct_to_wrong_flips": dict(Counter(r["ranking32"] for r in rows if r["correct24"] and not r["correct32"])), "per_pool": rows}
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
report = {"created_utc": datetime.now(timezone.utc).isoformat(), "arms": arms, "input_sha256": hashes,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "Descriptive fixed balanced bank only; top ties describe the choice path, not a causal mechanism. This does not validate a replacement tie-break policy or establish population-level deterioration."}
with (SIDE / "fixed-bank32-choice-diagnosis.json").open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({arm: {k: v for k, v in data.items() if k != "per_pool"} for arm, data in arms.items()}))
