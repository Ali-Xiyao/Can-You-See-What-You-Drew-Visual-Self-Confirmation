"""L2 runner contracts.

The vocabulary defect (EVIDENCE_LOG section 11) was a silent wrong-number bug:
metadata claimed one thing, the data said another, nothing raised. The gradient
probe has the same failure shape available to it -- if abstention removed a
prompt from one criterion but not the others, the stores would hold different
prompt sets and every paired Gram matrix would be built on mismatched rows.
These tests pin the joins and the symmetry rule.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.schemas import (
    AtomicObservation,
    CandidateRecord,
    ObservationResult,
)

_SPEC = importlib.util.spec_from_file_location(
    "run_v3_gradient_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "run_v3_gradient_probe.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _candidate(cid, seed, prompt="p1"):
    return CandidateRecord(
        candidate_id=cid,
        prompt_id=prompt,
        scene_id=prompt,
        sampling_seed=seed,
        image_path=f"{cid}.png",
        rgb_sha256=f"sha-{cid}",
        generator_id="showlab/show-o2-1.5B-HQ",
        generator_revision="d3a220ec",
        checkpoint_id="base",
    )


def _pool(gold_scores, seeds=None):
    seeds = seeds or {cid: 700_000_000 + i for i, cid in enumerate(gold_scores)}
    return {
        "prompt_id": "p1",
        "family": "existence",
        "scene": None,
        "questions": (),
        "candidates": [_candidate(cid, seeds[cid]) for cid in sorted(gold_scores)],
        "gold": dict(gold_scores),
    }


# --------------------------------------------------------------------------- gold


def test_gold_selection_prefers_a_verifier_correct_candidate():
    pool = _pool({"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0})
    assert pool["gold"][runner.gold_selection(pool)] == 1.0


def test_gold_selection_ties_break_exactly_like_the_model_arms():
    """Gold must use rfo.selection's key, not the bank's ordering helper.

    All three criteria score the same pool, so a tie has to resolve the same way
    for each of them; otherwise two criteria would "disagree" on a pool where
    the scores were actually identical, and that fake disagreement would land
    straight in the cross-criterion cosine.

    The shared rule is max on (score, -sampling_seed, candidate_id): lower
    sampling seed wins, then higher candidate id. Note this differs from
    v3.bank.paired_pool_assignment, which sorts ascending on (seed, id) --
    matching the arms is what matters here.
    """

    pool = _pool({"a": 1.0, "b": 1.0}, seeds={"a": 700_000_005, "b": 700_000_001})
    assert runner.gold_selection(pool) == "b"

    same_seed = _pool({"a": 1.0, "b": 1.0}, seeds={"a": 700_000_001, "b": 700_000_001})
    assert runner.gold_selection(same_seed) == "b"


def test_gold_selection_is_order_independent():
    pool = _pool({"a": 0.0, "b": 1.0, "c": 1.0})
    reversed_pool = dict(pool)
    reversed_pool["candidates"] = list(reversed(pool["candidates"]))
    assert runner.gold_selection(pool) == runner.gold_selection(reversed_pool)


# --------------------------------------------------------------------------- joins


def _write_shard(root, *, pool_rows, packets):
    (root / "packets").mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(packets):
        (root / "packets" / f"{index:04d}-{payload['scene_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (root / "pools.jsonl").write_text(
        "\n".join(json.dumps(row) for row in pool_rows) + "\n", encoding="utf-8"
    )


def _packet(scene_id, cids, scores):
    return {
        "scene_id": scene_id,
        "family": "existence",
        "candidates": [_candidate(cid, 700_000_000 + i, scene_id).to_dict()
                       for i, cid in enumerate(cids)],
        "scored": [
            {"candidate_id": cid, "sampling_seed": 700_000_000 + i,
             "gold_score": scores[cid], "abstained": False}
            for i, cid in enumerate(cids)
        ],
    }


def _pool_row(scene_id, cids, *, balanced=True, family="existence"):
    return {
        "prompt_id": scene_id,
        "family": family,
        "candidate_ids": cids,
        "correct_ids": cids[:2],
        "incorrect_ids": cids[2:],
        "balanced": balanced,
        "informative": True,
        "searched": 16,
        "abstained": 0,
        "reason": "balanced" if balanced else "informative_but_unbalanced",
    }


def _scene_index(scene_ids):
    return {sid: (object(), ()) for sid in scene_ids}


def test_load_pools_keeps_only_balanced_pools_in_scope(tmp_path):
    cids = ["a", "b", "c", "d"]
    scores = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    _write_shard(
        tmp_path / "shard",
        pool_rows=[
            _pool_row("s1", cids),
            _pool_row("s2", cids, balanced=False),
            _pool_row("s3", cids, family="color"),
        ],
        packets=[_packet(sid, cids, scores) for sid in ("s1", "s2", "s3")],
    )
    pools = runner.load_pools(
        [tmp_path / "shard"], {"existence", "spatial"}, _scene_index(["s1", "s2", "s3"])
    )
    assert [pool["prompt_id"] for pool in pools] == ["s1"]
    assert pools[0]["gold"] == scores


def test_load_pools_is_sorted_so_every_criterion_sees_one_order(tmp_path):
    cids = ["a", "b", "c", "d"]
    scores = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    _write_shard(
        tmp_path / "shard",
        pool_rows=[_pool_row(sid, cids) for sid in ("s3", "s1", "s2")],
        packets=[_packet(sid, cids, scores) for sid in ("s3", "s1", "s2")],
    )
    pools = runner.load_pools(
        [tmp_path / "shard"], {"existence"}, _scene_index(["s1", "s2", "s3"])
    )
    assert [pool["prompt_id"] for pool in pools] == ["s1", "s2", "s3"]
    assert [c.candidate_id for c in pools[0]["candidates"]] == cids


def test_load_pools_fails_when_the_packet_lost_a_candidate(tmp_path):
    """A pool referencing a candidate the packet does not hold must not go quiet."""

    scores = {"a": 1.0, "b": 1.0, "c": 0.0}
    _write_shard(
        tmp_path / "shard",
        pool_rows=[_pool_row("s1", ["a", "b", "c", "d"])],
        packets=[_packet("s1", ["a", "b", "c"], scores)],
    )
    with pytest.raises(SystemExit, match="lost candidates"):
        runner.load_pools([tmp_path / "shard"], {"existence"}, _scene_index(["s1"]))


def test_load_pools_fails_when_no_source_record_exists(tmp_path):
    cids = ["a", "b", "c", "d"]
    scores = dict.fromkeys(cids, 1.0)
    _write_shard(
        tmp_path / "shard",
        pool_rows=[_pool_row("s1", cids)],
        packets=[_packet("s1", cids, scores)],
    )
    with pytest.raises(SystemExit, match="No source record"):
        runner.load_pools([tmp_path / "shard"], {"existence"}, {})


# ------------------------------------------------------------------- abstention


class _FakeObserver:
    """Answers correctly unless the candidate is listed as an abstention."""

    def __init__(self, observer_id, abstain_on=(), prefer=None):
        self.model_id = observer_id
        self.observer_id = observer_id
        self.revision = "rev"
        self._abstain_on = set(abstain_on)
        self._prefer = prefer

    def _result(self, image_path, questions):
        cid = Path(image_path).stem
        answers = tuple(
            AtomicObservation(
                question_id=question.question_id,
                raw_answer="",
                normalized_answer=(
                    None
                    if cid in self._abstain_on
                    else (question.expected_answer if cid == self._prefer else "wrong")
                ),
                abstain=cid in self._abstain_on,
            )
            for question in questions
        )
        return ObservationResult(
            request_id=cid,
            observer_id=self.observer_id,
            observer_revision=self.revision,
            rgb_sha256=f"sha-{cid}",
            answers=answers,
        )

    # trainable-model interface
    def observe_atoms(self, image_path, questions):
        return self._result(image_path, questions)

    # observer-service interface
    def observe(self, request):
        return self._result(request.image_path, request.questions)


def _question(qid="q1", expected="yes"):
    from selfsight.schemas import AtomicQuestion, QuestionFamily, QuestionFormat

    return AtomicQuestion(
        question_id=qid,
        atom_id="atom1",
        family=QuestionFamily.EXISTENCE,
        text="Is there a green box in the image? Answer yes or no.",
        expected_answer=expected,
        question_format=QuestionFormat.OPEN,
    )


def _selection_pool(prompt_id):
    pool = _pool({"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0})
    pool["prompt_id"] = prompt_id
    pool["candidates"] = [_candidate(cid, 700_000_000 + i, prompt_id)
                          for i, cid in enumerate(["a", "b", "c", "d"])]
    pool["questions"] = (_question(),)
    return pool


def test_abstention_drops_the_prompt_from_every_criterion(tmp_path):
    """The core pairing invariant: no criterion may keep a prompt another dropped.

    If this regressed, the three stores would hold different prompt sets and
    gram_matrices would either raise or -- worse, if counts happened to match --
    silently pair unrelated prompts.
    """

    pools = [_selection_pool("s1"), _selection_pool("s2")]
    # The detector abstains on every candidate of s2 only.
    naive = _FakeObserver("showlab/show-o2-1.5B-HQ", prefer="a")

    class _PartialDetector(_FakeObserver):
        """Abstains on every candidate of s2, and on none of s1."""

        def observe(self, request):
            self._abstain_on = (
                {"a", "b", "c", "d"} if request.request_id.startswith("s2:") else set()
            )
            self._prefer = "b"
            return super().observe(request)

    detector = _PartialDetector("Qwen/Qwen3-VL-8B-Instruct")
    decisions = runner.run_selection(
        pools, naive, detector, {"observer_id": "Qwen/Qwen3-VL-8B-Instruct",
                                 "revision": "dir-sha256:abc"}, tmp_path
    )

    assert set(decisions) == {"s1"}, "s2 abstained for rfo and must vanish everywhere"
    assert set(decisions["s1"]) == {"naive", "rfo", "gold"}

    rows = [json.loads(line) for line in (tmp_path / "selection.jsonl").read_text(
        encoding="utf-8").splitlines()]
    dropped = [row for row in rows if row.get("dropped")]
    assert [row["prompt_id"] for row in dropped] == ["s2"]
    assert dropped[0]["dropped"] == "selector_abstained"


def test_surviving_prompts_carry_all_three_selections(tmp_path):
    pools = [_selection_pool("s1")]
    naive = _FakeObserver("showlab/show-o2-1.5B-HQ", prefer="a")
    detector = _FakeObserver("Qwen/Qwen3-VL-8B-Instruct", prefer="b")

    decisions = runner.run_selection(
        pools, naive, detector,
        {"observer_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": "dir-sha256:abc"},
        tmp_path,
    )
    chosen = decisions["s1"]
    assert chosen["naive"] == "a"
    assert chosen["rfo"] == "b"
    # gold ignores both observers and takes a verifier-correct candidate.
    assert chosen["gold"] in {"a", "b"}
