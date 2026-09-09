"""How good a predictor of "this image is right" is each score?

`bsv_selection.py` answers the operational question -- keep the argmax, how
often is the kept image correct. A reviewer will ask the prior one: is the blind
score a better *predictor*, or does it just happen to win the argmax on this
corpus? Same files, no GPU.

Two AUCs, because they answer different questions and only one of them is the
one a selection loop faces:

    across candidates   pooled over everything. Inflated by prompt difficulty:
                        an easy description produces both a high score and a
                        correct image, and that correlation is not the model
                        knowing anything about the picture in front of it.
    within pool         concordance among the candidates for the same
                        description, so difficulty is held fixed by
                        construction. This is the quantity selection uses.
"""
from __future__ import annotations

import random

from bsv_selection import by_spec, load


def auc(scores: list[tuple[float, bool]]) -> float:
    """Mann-Whitney U as a probability, ties counted as half a win.

    Ties matter here more than usual: four binary questions give five distinct
    scores, so a rule that counted them as wins would report the ceiling.
    """

    right = [score for score, correct in scores if correct]
    wrong = [score for score, correct in scores if not correct]
    if not right or not wrong:
        return float("nan")
    wins = sum(
        1.0 if r > w else 0.5 if r == w else 0.0
        for r in right for w in wrong
    )
    return wins / (len(right) * len(wrong))


def within_pool_auc(pools: list[list[dict]], key: str) -> tuple[float, int]:
    """Concordance among candidates for the same description.

    Every (right, wrong) pair inside a pool votes once. Pools that are all
    right or all wrong have no pairs and contribute nothing, which is correct:
    they carry no information about ordering.
    """

    wins = 0.0
    pairs = 0
    for pool in pools:
        for right in (item for item in pool if item["correct"]):
            for wrong in (item for item in pool if not item["correct"]):
                pairs += 1
                if right[key] > wrong[key]:
                    wins += 1
                elif right[key] == wrong[key]:
                    wins += 0.5
    return (wins / pairs if pairs else float("nan")), pairs


def paired_bootstrap(pools: list[list[dict]], draws: int = 20000,
                     seed: int = 20260908) -> tuple[float, float]:
    """Resample pools: the pairs inside one pool are not independent."""

    rng = random.Random(seed)
    n = len(pools)
    deltas = []
    for _ in range(draws):
        sample = [pools[rng.randrange(n)] for _ in range(n)]
        blind, pairs = within_pool_auc(sample, "blind")
        told, _ = within_pool_auc(sample, "prompted")
        if pairs:
            deltas.append(blind - told)
    deltas.sort()
    return deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]


def main() -> None:
    pools = list(by_spec(load()).values())

    print("Is the blind score a better predictor, or just a luckier argmax?")
    print("=" * 74)
    print()
    print("across candidates (confounded by how hard each description is)")
    for key in ("blind", "prompted"):
        flat = [(item[key], item["correct"]) for pool in pools for item in pool]
        print(f"    {key:<10} AUC {auc(flat):.3f}   n={len(flat)}")
    print()

    print("within pool (difficulty held fixed -- the quantity selection uses)")
    for key in ("blind", "prompted"):
        value, pairs = within_pool_auc(pools, key)
        print(f"    {key:<10} AUC {value:.3f}   {pairs} right/wrong pairs")
    blind, _ = within_pool_auc(pools, "blind")
    told, _ = within_pool_auc(pools, "prompted")
    low, high = paired_bootstrap(pools)
    print(f"    difference {blind - told:+.3f} [{low:+.3f},{high:+.3f}]")
    print()

    print("what each score actually looks like")
    for key in ("blind", "prompted"):
        values = [item[key] for pool in pools for item in pool]
        at_ceiling = sum(1 for value in values if value == 1.0) / len(values)
        print(f"    {key:<10} mean {sum(values) / len(values):.3f}   "
              f"share at 1.000 {at_ceiling:.3f}")


if __name__ == "__main__":
    main()
