"""Prompt-level paired inference for cross-criterion gradient cosines.

v2.x gated `cos(g_naive, g_rfo)` against the cosine between two *disjoint* halves
of the probe set. Those two quantities are not on the same scale: the
cross-criterion cosine is computed on the *same* prompts (noise is shared and
cancels pairwise), while the split-half cosine is computed on disjoint samples
(noise is independent and doubled). The measured consequence was
`identical == 1.000` and `naive/rfo_self == 1.000` alongside a split-half floor of
`0.077` -- see `docs/EVIDENCE_LOG.md` section 5.

This module replaces that control with a prompt-level paired bootstrap: resample
the prompt set with replacement and recompute the cosine of the two *mean*
gradients on the same resample. The bootstrap is exact and cheap because for a
resample with per-prompt counts ``c``::

    <g_L(c), g_R(c)>  = c' G_LR c / n^2
    ||g_L(c)||^2      = c' G_LL c / n^2

so the ``1/n^2`` factors cancel in the cosine and only the three n-by-n Gram
matrices are needed. Per-prompt gradient vectors are streamed to a memmap and
consumed in parameter-space chunks, so peak memory is bounded by the chunk size
rather than by ``n * D``.

The disjoint split-half quantity is retained as `sample_noise_diagnostic`: it is
a genuine measure of per-sample gradient SNR and is worth reporting, but it is no
longer a gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 5_000
DEFAULT_CHUNK_ELEMENTS = 1 << 22


def _as_vector(value: Any) -> np.ndarray:
    """Accept a torch tensor or anything array-like without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise FloatingPointError("Per-prompt gradient contains non-finite values")
    return vector


class PerPromptGradientStore:
    """Memmap-backed per-prompt LoRA gradients for one selection criterion.

    One row per prompt. Rows must be added in a deterministic prompt order and
    the same order must be used for every criterion, otherwise the Gram matrices
    are not paired.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        criterion: str,
        dimension: int,
        capacity: int,
        dtype: str = "float32",
    ) -> None:
        if dimension <= 0 or capacity <= 0:
            raise ValueError("dimension and capacity must be positive")
        if dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be float32 or float16")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.criterion = criterion
        self.dimension = int(dimension)
        self.capacity = int(capacity)
        self.dtype = dtype
        self._prompt_ids: list[str] = []
        self._memmap = np.memmap(
            self.path, dtype=dtype, mode="w+", shape=(self.capacity, self.dimension)
        )
        self._finalized = False

    def add(self, prompt_id: str, vector: Any) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add to a finalized gradient store")
        if prompt_id in self._prompt_ids:
            raise ValueError(f"Duplicate prompt in {self.criterion} store: {prompt_id}")
        if len(self._prompt_ids) >= self.capacity:
            raise ValueError(f"Gradient store is full at capacity {self.capacity}")
        data = _as_vector(vector)
        if data.size != self.dimension:
            raise ValueError(
                f"Gradient dimension mismatch for {prompt_id}: {data.size} != {self.dimension}"
            )
        self._memmap[len(self._prompt_ids), :] = data.astype(self.dtype, copy=False)
        self._prompt_ids.append(prompt_id)

    def finalize(self) -> PerPromptGradientStore:
        self._memmap.flush()
        self._finalized = True
        return self

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(self._prompt_ids)

    @property
    def count(self) -> int:
        return len(self._prompt_ids)

    def rows(self, start: int, stop: int) -> np.ndarray:
        """Return a parameter-space slice of every stored prompt, as float64."""

        if not 0 <= start < stop <= self.dimension:
            raise ValueError(f"Invalid parameter slice [{start}, {stop})")
        return np.asarray(self._memmap[: self.count, start:stop], dtype=np.float64)

    def close(self) -> None:
        self._memmap.flush()


@dataclass(frozen=True)
class GramMatrices:
    """Exact inner-product matrices for two paired per-prompt gradient sets."""

    left: str
    right: str
    prompt_ids: tuple[str, ...]
    g_ll: np.ndarray
    g_rr: np.ndarray
    g_lr: np.ndarray

    @property
    def n(self) -> int:
        return len(self.prompt_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "n": self.n,
            "prompt_ids": list(self.prompt_ids),
        }


def gram_matrices(
    left: PerPromptGradientStore,
    right: PerPromptGradientStore,
    *,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> GramMatrices:
    """Accumulate the three exact Gram matrices in bounded memory.

    Peak memory is ``2 * n * chunk_width * 8`` bytes, independent of ``D``.
    """

    if left.prompt_ids != right.prompt_ids:
        raise ValueError("Paired stores must hold the same prompts in the same order")
    if left.dimension != right.dimension:
        raise ValueError("Paired stores must share a parameter dimension")
    n = left.count
    if n < 2:
        raise ValueError("Paired bootstrap requires at least two prompts")
    chunk_width = max(1, int(chunk_elements) // max(1, n))
    g_ll = np.zeros((n, n), dtype=np.float64)
    g_rr = np.zeros((n, n), dtype=np.float64)
    g_lr = np.zeros((n, n), dtype=np.float64)
    for start in range(0, left.dimension, chunk_width):
        stop = min(start + chunk_width, left.dimension)
        block_left = left.rows(start, stop)
        block_right = right.rows(start, stop)
        g_ll += block_left @ block_left.T
        g_rr += block_right @ block_right.T
        g_lr += block_left @ block_right.T
    return GramMatrices(
        left=left.criterion,
        right=right.criterion,
        prompt_ids=left.prompt_ids,
        g_ll=g_ll,
        g_rr=g_rr,
        g_lr=g_lr,
    )


def _weighted_cosine(gram: GramMatrices, counts: np.ndarray) -> np.ndarray:
    """Cosine of the two mean gradients under resample counts (rows of `counts`)."""

    numerator = ((counts @ gram.g_lr) * counts).sum(axis=1)
    left_norm = ((counts @ gram.g_ll) * counts).sum(axis=1)
    right_norm = ((counts @ gram.g_rr) * counts).sum(axis=1)
    denominator = np.sqrt(np.clip(left_norm, 0.0, None) * np.clip(right_norm, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return np.clip(value, -1.0, 1.0)


def cosine_from_gram(gram: GramMatrices) -> float:
    """Point estimate: cosine of the two equally weighted mean gradients."""

    counts = np.ones((1, gram.n), dtype=np.float64)
    return float(_weighted_cosine(gram, counts)[0])


def _resample_counts(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)


def paired_bootstrap_cosine(
    gram: GramMatrices,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260901,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Prompt-level paired bootstrap CI for a cross-criterion gradient cosine.

    This is the v3.0 replacement for the disjoint split-half noise floor. The
    resample is shared by both criteria, so the CI reflects only prompt-sampling
    uncertainty in the *contrast*, not the much larger uncertainty in either
    absolute gradient direction.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    counts = _resample_counts(gram.n, resamples, seed)
    draws = _weighted_cosine(gram, counts)
    finite = draws[np.isfinite(draws)]
    if finite.size < resamples // 2:
        raise FloatingPointError("Bootstrap produced too many degenerate resamples")
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(finite, alpha))
    high = float(np.quantile(finite, 1.0 - alpha))
    return {
        "left": gram.left,
        "right": gram.right,
        "n_prompts": gram.n,
        "cosine": cosine_from_gram(gram),
        "ci_low": low,
        "ci_high": high,
        "ci_width": high - low,
        "bootstrap_median": float(np.median(finite)),
        "bootstrap_std": float(finite.std(ddof=1)),
        "confidence": confidence,
        "resamples": int(resamples),
        "finite_resamples": int(finite.size),
        "seed": int(seed),
        "control": "prompt_level_paired_bootstrap",
    }


def paired_bootstrap_difference(
    first: GramMatrices,
    second: GramMatrices,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260901,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """CI for `cosine(first) - cosine(second)` under a shared prompt resample.

    Use this to compare, for example, `cos(g_naive, g_rfo)` against
    `cos(g_naive, g_gold)` at one checkpoint, or the same cosine at two
    checkpoints. Comparing two independently computed CIs by eye badly
    understates power because the resample is shared.
    """

    if first.prompt_ids != second.prompt_ids:
        raise ValueError("Difference requires the same prompts in the same order")
    counts = _resample_counts(first.n, resamples, seed)
    draws = _weighted_cosine(first, counts) - _weighted_cosine(second, counts)
    finite = draws[np.isfinite(draws)]
    if finite.size < resamples // 2:
        raise FloatingPointError("Bootstrap produced too many degenerate resamples")
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(finite, alpha))
    high = float(np.quantile(finite, 1.0 - alpha))
    return {
        "first": f"{first.left}|{first.right}",
        "second": f"{second.left}|{second.right}",
        "n_prompts": first.n,
        "difference": cosine_from_gram(first) - cosine_from_gram(second),
        "ci_low": low,
        "ci_high": high,
        "ci_width": high - low,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "confidence": confidence,
        "resamples": int(resamples),
        "seed": int(seed),
        "control": "shared_resample_paired_difference",
    }


def sample_noise_diagnostic(
    gram: GramMatrices,
    *,
    splits: int = 8,
    seed: int = 20260901,
) -> dict[str, Any]:
    """Disjoint split-half cosine, reported as per-sample gradient SNR only.

    v2.x used this as the gate for cross-criterion cosines; it measured a
    different quantity and produced two misleading red gates. It is retained
    because it does answer a real question -- how much a mean gradient direction
    depends on which prompts went into it -- but it must never gate a paired
    contrast. Reported values near zero mean the *absolute* direction is
    sample-dominated, which says nothing about the paired contrast.
    """

    if gram.n < 4:
        raise ValueError("Split-half diagnostic requires at least four prompts")
    rng = np.random.default_rng(seed)
    half = gram.n // 2
    values: list[float] = []
    for _ in range(int(splits)):
        order = rng.permutation(gram.n)
        left_counts = np.zeros((1, gram.n), dtype=np.float64)
        right_counts = np.zeros((1, gram.n), dtype=np.float64)
        left_counts[0, order[:half]] = 1.0
        right_counts[0, order[half : 2 * half]] = 1.0
        numerator = float((left_counts @ gram.g_ll @ right_counts.T)[0, 0])
        left_norm = float((left_counts @ gram.g_ll @ left_counts.T)[0, 0])
        right_norm = float((right_counts @ gram.g_ll @ right_counts.T)[0, 0])
        denominator = np.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0))
        if denominator > 0.0:
            values.append(float(np.clip(numerator / denominator, -1.0, 1.0)))
    if len(values) < 2:
        raise ValueError("Split-half diagnostic produced too few finite comparisons")
    array = np.asarray(values, dtype=np.float64)
    return {
        "criterion": gram.left,
        "splits": len(values),
        "half_size": half,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "low": float(np.quantile(array, 0.025)),
        "high": float(np.quantile(array, 0.975)),
        "std": float(array.std(ddof=1)),
        "role": "per_sample_gradient_snr_diagnostic_only",
        "is_gate": False,
    }


def gate_b_instrument_report(
    checkpoints: Sequence[dict[str, Any]],
    *,
    max_ci_width: float = 0.10,
    identical_cosine_min: float = 0.999,
    identical_cosine: float | None = None,
) -> dict[str, Any]:
    """Gate B: is the gradient instrument sharp enough to support a claim?

    `checkpoints` are `paired_bootstrap_cosine` outputs in training order.
    """

    if not checkpoints:
        raise ValueError("Gate B requires at least one checkpoint measurement")
    widths = [float(item["ci_width"]) for item in checkpoints]
    worst = max(widths)
    ci_ok = worst <= max_ci_width
    identical_ok = identical_cosine is None or float(identical_cosine) >= identical_cosine_min
    degenerate = [
        item for item in checkpoints if abs(float(item["cosine"])) < 10.0 * float(item["ci_width"])
        and abs(float(item["cosine"])) < 0.02
    ]
    return {
        "gate": "b_instrument",
        "checkpoints": len(checkpoints),
        "max_ci_width": max_ci_width,
        "worst_ci_width": worst,
        "median_ci_width": float(np.median(widths)),
        "ci_width_ok": bool(ci_ok),
        "identical_cosine": identical_cosine,
        "identical_cosine_min": identical_cosine_min,
        "identical_cosine_ok": bool(identical_ok),
        "degenerate_checkpoints": len(degenerate),
        "subspace_ok": bool(not degenerate),
        "passed": bool(ci_ok and identical_ok and not degenerate),
        "action_if_failed": (
            "Increase the number of informative pools in the probe until the paired CI "
            "narrows. If the registered ceiling is reached without narrowing, fall back "
            "to entropy-class unlabeled signals and report that the gradient signal is "
            "not measurable at this scale."
        ),
    }


def early_safety_check(
    early: Iterable[dict[str, Any]],
    *,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Claim (1) H-2: early divergence must lie inside the start-point CI.

    A cross-criterion cosine that is already separated at step 0 measures a
    constant definitional difference, not a training dynamic.
    """

    rows = []
    inside = 0
    for item in early:
        value = float(item["cosine"])
        contained = float(reference["ci_low"]) <= value <= float(reference["ci_high"])
        inside += int(contained)
        rows.append({"cosine": value, "inside_start_ci": contained})
    total = len(rows)
    if total == 0:
        raise ValueError("Early-safety check requires at least one early checkpoint")
    return {
        "check": "early_safety",
        "reference_ci": [float(reference["ci_low"]), float(reference["ci_high"])],
        "early_checkpoints": total,
        "inside": inside,
        "rows": rows,
        "passed": inside == total,
        "meaning_if_failed": (
            "Divergence is present from the first checkpoint, so it reflects a constant "
            "difference between the two selection criteria rather than a training dynamic."
        ),
    }
