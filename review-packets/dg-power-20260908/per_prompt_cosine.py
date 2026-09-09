"""The gradient probe already saves everything needed for a better statistic.

`gda_free` is reported as cosine(mean_L, mean_R) -- one cosine formed from two
averaged gradient fields. That is a ratio of sums over 16 prompts, and its
bootstrap distribution is badly behaved: the pilot's intervals run
[-0.3095, +0.3587] and [-0.2279, +0.6080], and for naive/step-16 the point
estimate (-0.3122) falls outside its own lower bound (-0.3095).

But `grams.npz` stores the full 16x16 ll / rr / lr matrices at every checkpoint,
so the diagonal gives a per-prompt cosine

    cos_i = lr[i,i] / sqrt(ll[i,i] * rr[i,i])

and the mean of 16 bounded per-prompt cosines is an ordinary sample mean with an
ordinary standard error. Nothing new has to be measured on a GPU: this is a
re-read of gradient probes that already ran.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[2] / "runs" / "v4" / "decoupling-pilot-20260906"
PROBES = RUN / "gradient-probes"


def cosine_of_means(ll: np.ndarray, rr: np.ndarray, lr: np.ndarray) -> float:
    """The estimand as currently reported: one cosine between two mean fields."""
    return float(lr.sum() / np.sqrt(ll.sum() * rr.sum()))


def per_prompt_cosines(ll: np.ndarray, rr: np.ndarray, lr: np.ndarray) -> np.ndarray:
    return np.diag(lr) / np.sqrt(np.diag(ll) * np.diag(rr))


def load(arm: str, step: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = PROBES / arm / f"step-{step:05d}" / "grams.npz"
    z = np.load(path, allow_pickle=True)
    return (z["prompt_ids"], z[f"{kind}_ll"], z[f"{kind}_rr"], z[f"{kind}_lr"])


def main() -> None:
    kind = "gda_free"
    sensitivity = json.loads((RUN / "gradient_sensitivity.json").read_text(encoding="utf-8"))
    retained = set(sensitivity["arms"]["naive"][0]["delta_from_base"][kind]["prompt_ids"])

    base_ids, *base_grams = load("base", 0, kind)
    base_cos = per_prompt_cosines(*base_grams)
    keep = np.array([pid in retained for pid in base_ids])
    print(f"{len(base_ids)} probe prompts, {int(keep.sum())} retained after scene-disjointness\n")

    print(f"{'arm':9s}{'step':>5}  {'cosine-of-means':>16s}   "
          f"{'mean per-prompt cosine +- SEM':>31s}   {'paired delta vs base +- SEM':>29s}")
    for arm in ("base", "naive", "rfo_gold"):
        for step in ([0] if arm == "base" else [8, 16, 24, 32]):
            ids, ll, rr, lr = load(arm, step, kind)
            assert list(ids) == list(base_ids), "probe bank is not fixed across checkpoints"
            grand = cosine_of_means(ll[np.ix_(keep, keep)], rr[np.ix_(keep, keep)],
                                    lr[np.ix_(keep, keep)])
            cos = per_prompt_cosines(ll, rr, lr)[keep]
            n = len(cos)
            sem = float(cos.std(ddof=1) / np.sqrt(n))
            delta = cos - base_cos[keep]
            dsem = float(delta.std(ddof=1) / np.sqrt(n))
            print(f"{arm:9s}{step:>5}  {grand:>16.4f}   "
                  f"{cos.mean():>18.4f} +- {sem:<8.4f}   {delta.mean():>+14.4f} +- {dsem:<8.4f}")

    # What the two error bars mean for the detection rule.
    all_sems, all_dsems = [], []
    for arm in ("naive", "rfo_gold"):
        for step in (8, 16, 24, 32):
            _, ll, rr, lr = load(arm, step, kind)
            cos = per_prompt_cosines(ll, rr, lr)[keep]
            all_sems.append(cos.std(ddof=1) / np.sqrt(len(cos)))
            all_dsems.append((cos - base_cos[keep]).std(ddof=1) / np.sqrt(len(cos)))
    marginal_sd = 0.8262 / 3.92          # measured median width of the reported intervals
    print(f"\nreported cosine-of-means SD (median CI width / 3.92) : {marginal_sd:.4f}")
    print(f"mean per-prompt cosine, median SEM                   : {np.median(all_sems):.4f}"
          f"   ({marginal_sd / np.median(all_sems):.1f}x tighter)")
    print(f"paired per-prompt delta, median SEM                  : {np.median(all_dsems):.4f}"
          f"   ({marginal_sd / np.median(all_dsems):.1f}x tighter)")


if __name__ == "__main__":
    main()
