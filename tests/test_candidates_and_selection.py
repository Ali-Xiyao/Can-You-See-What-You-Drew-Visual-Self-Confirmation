from __future__ import annotations

from selfsight.data.candidates import assert_paired_candidate_pools
from selfsight.schemas import CandidateRecord


def _candidate(candidate_id: str, prompt_id: str, seed: int, rgb_hash: str) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        prompt_id=prompt_id,
        scene_id=prompt_id,
        sampling_seed=seed,
        image_path="H:\\selfsight-data\\dummy.png",
        rgb_sha256=rgb_hash,
        generator_id="test",
        generator_revision="0" * 40,
        checkpoint_id="base",
    )


def test_same_candidate_pool_is_required_for_gradient_criteria():
    left = [_candidate("p0-k0", "p0", 1, "a" * 64), _candidate("p0-k1", "p0", 2, "b" * 64)]
    right = list(left)
    assert_paired_candidate_pools(left, right, expected_k=2)


def test_candidate_pool_mismatch_is_rejected():
    left = [_candidate("p0-k0", "p0", 1, "a" * 64), _candidate("p0-k1", "p0", 2, "b" * 64)]
    right = [_candidate("p0-k0", "p0", 1, "a" * 64), _candidate("p0-k1", "p0", 2, "c" * 64)]
    try:
        assert_paired_candidate_pools(left, right, expected_k=2)
    except ValueError as exc:
        assert "Unpaired candidate pools" in str(exc)
    else:
        raise AssertionError("A mismatched RGB pool must be rejected")
