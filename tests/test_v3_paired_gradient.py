"""Paired gradient inference replaces the mis-scaled split-half noise floor."""

from __future__ import annotations

import numpy as np
import pytest

from selfsight.v3.paired import (
    PerPromptGradientStore,
    cosine_from_gram,
    early_safety_check,
    gate_b_instrument_report,
    gram_matrices,
    paired_bootstrap_cosine,
    paired_bootstrap_difference,
    sample_noise_diagnostic,
)


def _stores(tmp_path, left_vectors, right_vectors, dimension):
    left = PerPromptGradientStore(
        tmp_path / "left.dat", criterion="naive", dimension=dimension, capacity=len(left_vectors)
    )
    right = PerPromptGradientStore(
        tmp_path / "right.dat", criterion="rfo", dimension=dimension, capacity=len(right_vectors)
    )
    for index, (a, b) in enumerate(zip(left_vectors, right_vectors, strict=True)):
        prompt = f"p{index:03d}"
        left.add(prompt, a)
        right.add(prompt, b)
    return left.finalize(), right.finalize()


def test_gram_matrices_are_exact_under_chunking(tmp_path):
    rng = np.random.default_rng(11)
    dimension = 257
    left_vectors = rng.normal(size=(9, dimension))
    right_vectors = rng.normal(size=(9, dimension))
    left, right = _stores(tmp_path, left_vectors, right_vectors, dimension)

    # A chunk width that does not divide the dimension exercises the remainder path.
    gram = gram_matrices(left, right, chunk_elements=9 * 40)

    assert np.allclose(gram.g_ll, left_vectors @ left_vectors.T, atol=1e-4)
    assert np.allclose(gram.g_lr, left_vectors @ right_vectors.T, atol=1e-4)
    expected = float(
        left_vectors.mean(0)
        @ right_vectors.mean(0)
        / (np.linalg.norm(left_vectors.mean(0)) * np.linalg.norm(right_vectors.mean(0)))
    )
    assert cosine_from_gram(gram) == pytest.approx(expected, abs=1e-4)


def test_identical_criteria_give_cosine_one_and_zero_width_ci(tmp_path):
    """The v2.3 `identical` control measured exactly 1.000; the CI must agree."""

    rng = np.random.default_rng(3)
    dimension = 64
    vectors = rng.normal(size=(12, dimension))
    left, right = _stores(tmp_path, vectors, vectors.copy(), dimension)
    result = paired_bootstrap_cosine(gram_matrices(left, right), resamples=500, seed=7)

    assert result["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert result["ci_width"] == pytest.approx(0.0, abs=1e-6)
    assert result["control"] == "prompt_level_paired_bootstrap"


def test_paired_bootstrap_is_far_tighter_than_split_half(tmp_path):
    """This is the whole point: the two controls are not on the same scale.

    Both criteria share a weak common direction plus large independent per-prompt
    noise. The paired contrast is precise; the disjoint split-half of a single
    criterion is not. v2.x gated the former with the latter.
    """

    rng = np.random.default_rng(20260901)
    dimension = 512
    n = 40
    shared = rng.normal(size=(n, dimension))
    common = rng.normal(size=dimension) * 0.35
    left_vectors = shared + common
    right_vectors = shared + common * 0.9
    left, right = _stores(tmp_path, left_vectors, right_vectors, dimension)
    gram = gram_matrices(left, right)

    paired = paired_bootstrap_cosine(gram, resamples=2000, seed=5)
    split_half = sample_noise_diagnostic(gram, splits=8, seed=5)

    assert paired["ci_width"] < 0.05
    assert split_half["high"] - split_half["low"] > paired["ci_width"]
    assert split_half["is_gate"] is False
    assert split_half["role"] == "per_sample_gradient_snr_diagnostic_only"


def test_paired_difference_uses_a_shared_resample(tmp_path):
    rng = np.random.default_rng(101)
    dimension = 128
    base = rng.normal(size=(16, dimension))
    near = base + rng.normal(size=(16, dimension)) * 0.05
    far = rng.normal(size=(16, dimension))
    left, right_near = _stores(tmp_path / "a", base, near, dimension)
    left_again, right_far = _stores(tmp_path / "b", base, far, dimension)

    first = gram_matrices(left, right_near)
    second = gram_matrices(left_again, right_far)
    result = paired_bootstrap_difference(first, second, resamples=1000, seed=9)

    assert result["difference"] > 0.0
    assert result["excludes_zero"] is True
    assert result["control"] == "shared_resample_paired_difference"


def test_difference_requires_the_same_prompt_order(tmp_path):
    rng = np.random.default_rng(2)
    dimension = 32
    vectors = rng.normal(size=(6, dimension))
    left, right = _stores(tmp_path / "a", vectors, vectors, dimension)
    other = PerPromptGradientStore(
        tmp_path / "c.dat", criterion="naive", dimension=dimension, capacity=6
    )
    for index in range(6):
        other.add(f"other{index}", vectors[index])
    other.finalize()

    first = gram_matrices(left, right)
    second = gram_matrices(other, other)
    with pytest.raises(ValueError, match="same prompts"):
        paired_bootstrap_difference(first, second)


def test_store_rejects_unpaired_or_malformed_input(tmp_path):
    store = PerPromptGradientStore(
        tmp_path / "s.dat", criterion="naive", dimension=8, capacity=2
    )
    store.add("p0", np.ones(8))
    with pytest.raises(ValueError, match="Duplicate prompt"):
        store.add("p0", np.ones(8))
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add("p1", np.ones(9))
    with pytest.raises(FloatingPointError):
        store.add("p2", np.full(8, np.nan))


def test_gram_rejects_mismatched_prompt_order(tmp_path):
    rng = np.random.default_rng(4)
    left = PerPromptGradientStore(
        tmp_path / "l.dat", criterion="naive", dimension=16, capacity=3
    )
    right = PerPromptGradientStore(
        tmp_path / "r.dat", criterion="rfo", dimension=16, capacity=3
    )
    for index in range(3):
        left.add(f"p{index}", rng.normal(size=16))
    for index in reversed(range(3)):
        right.add(f"p{index}", rng.normal(size=16))
    with pytest.raises(ValueError, match="same prompts in the same order"):
        gram_matrices(left.finalize(), right.finalize())


def test_gate_b_fails_on_wide_intervals():
    wide = [{"cosine": 0.8, "ci_width": 0.31}, {"cosine": 0.7, "ci_width": 0.12}]
    report = gate_b_instrument_report(wide, identical_cosine=1.0)
    assert report["passed"] is False
    assert report["worst_ci_width"] == pytest.approx(0.31)
    assert "informative pools" in report["action_if_failed"]

    narrow = [{"cosine": 0.8, "ci_width": 0.04}, {"cosine": 0.62, "ci_width": 0.07}]
    assert gate_b_instrument_report(narrow, identical_cosine=1.0)["passed"] is True


def test_gate_b_fails_when_identical_control_is_broken():
    """Reproduces the real v2.3 regression: unseeded LoRA dropout gave 0.9889."""

    narrow = [{"cosine": 0.8, "ci_width": 0.03}]
    report = gate_b_instrument_report(narrow, identical_cosine=0.9889)
    assert report["identical_cosine_ok"] is False
    assert report["passed"] is False


def test_early_safety_detects_constant_offset():
    reference = {"ci_low": 0.95, "ci_high": 1.0}
    assert early_safety_check([{"cosine": 0.98}], reference=reference)["passed"] is True

    separated = early_safety_check([{"cosine": 0.30}], reference=reference)
    assert separated["passed"] is False
    assert "constant difference" in separated["meaning_if_failed"]
