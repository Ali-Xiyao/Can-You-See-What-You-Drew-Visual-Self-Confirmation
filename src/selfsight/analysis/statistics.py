"""Seed-level paired bootstrap and registered correlation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def paired_bootstrap_lead(
    d_star_by_seed: Mapping[int, float],
    d_g_by_seed: Mapping[int, float],
    *,
    resamples: int = 20_000,
    seed: int = 20260827,
) -> dict[str, float | list[float]]:
    shared = sorted(set(d_star_by_seed).intersection(d_g_by_seed))
    if len(shared) < 3:
        raise ValueError("Formal paired bootstrap requires all three registered seeds")
    leads = np.asarray([d_star_by_seed[item] - d_g_by_seed[item] for item in shared], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(leads), size=(resamples, len(leads)))
    bootstrap = np.median(leads[indices], axis=1)
    return {
        "seeds": [float(item) for item in shared],
        "lead_by_seed": leads.tolist(),
        "median": float(np.median(leads)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "resamples": float(resamples),
    }


def bootstrap_median(
    values_by_seed: Mapping[int, float],
    *,
    resamples: int = 20_000,
    seed: int = 20260827,
) -> dict[str, float | list[float]]:
    if len(values_by_seed) < 3:
        raise ValueError("Formal seed-level bootstrap requires all three registered seeds")
    seeds = sorted(values_by_seed)
    values = np.asarray([values_by_seed[item] for item in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap = np.median(values[indices], axis=1)
    return {
        "seeds": [float(item) for item in seeds],
        "values_by_seed": values.tolist(),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "resamples": float(resamples),
    }


def checkpoint_spearman(
    gda_free_by_seed: Mapping[int, Sequence[float]],
    gda_gold_by_seed: Mapping[int, Sequence[float]],
) -> dict[int, float]:
    output = {}
    for seed in sorted(set(gda_free_by_seed).intersection(gda_gold_by_seed)):
        left = np.asarray(gda_free_by_seed[seed], dtype=float)
        right = np.asarray(gda_gold_by_seed[seed], dtype=float)
        if left.shape != right.shape or left.size < 3:
            raise ValueError(f"Invalid paired checkpoint series for seed {seed}")
        output[seed] = float(spearmanr(left, right).statistic)
    return output
