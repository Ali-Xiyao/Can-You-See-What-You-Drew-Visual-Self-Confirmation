"""Does the model's internal score carry information about its own errors?

This is the project's title question in its most direct form, and it does not
need a knot, a breakpoint, or a timestamp. At each checkpoint the run already
records, per prompt, an internal cycle score and an external correctness bit.
The AUC of the first predicting the second is the amount of self-knowledge
available for self-improvement to exploit:

    AUC = 0.50   the internal score is uninformative about its own errors, and
                 no training scheme built on it can work, at any budget
    AUC > 0.50   there is signal, and its decline over training is what
                 "self-confirmation" would mean, measured as a contrast rather
                 than as the location of a bend

Everything here reads artefacts the pilot already wrote.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[2] / "runs" / "v4" / "decoupling-pilot-20260906"
ARMS = ("naive", "rfo_gold")


def auc_and_se(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, int, int]:
    """Mann-Whitney AUC with the Hanley-McNeil standard error."""
    positive, negative = scores[labels], scores[~labels]
    n_pos, n_neg = len(positive), len(negative)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), n_pos, n_neg
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(n_pos + n_neg, dtype=float)
    ranks[order] = np.arange(1, n_pos + n_neg + 1, dtype=float)
    values = np.concatenate([positive, negative])
    for value in np.unique(values):                    # average ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
                + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(auc), float(np.sqrt(max(variance, 0.0))), n_pos, n_neg


def load(arm: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    directory = RUN / "evaluations" / arm / f"step-{step:05d}"
    cycle = json.loads((directory / "cycle.json").read_text(encoding="utf-8"))["scores"]
    correct: dict[str, bool] = {}
    with open(directory / "verified.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            correct[row["spec_id"]] = bool(row["image_correct"])
    shared = sorted(set(cycle) & set(correct))
    return (np.array([cycle[key] for key in shared], dtype=float),
            np.array([correct[key] for key in shared], dtype=bool))


def main() -> None:
    print("AUC of the internal cycle score predicting external image_correct")
    print("(0.50 = the model's self-evaluation knows nothing about its own errors)\n")
    print(f"{'arm':9s}{'step':>5}{'n':>5}{'external':>10}{'AUC':>9}{'SE':>8}"
          f"{'95% CI':>18}{'informative?':>14}")
    for arm in ARMS:
        for step in (0, 8, 16, 24, 32):
            scores, labels = load(arm, step)
            auc, se, n_pos, n_neg = auc_and_se(scores, labels)
            low, high = auc - 1.96 * se, auc + 1.96 * se
            verdict = "yes" if low > 0.5 else ("no" if high < 0.5 else "cannot tell")
            print(f"{arm:9s}{step:>5}{n_pos + n_neg:>5}{n_pos / (n_pos + n_neg):>10.1%}"
                  f"{auc:>9.3f}{se:>8.3f}   [{low:>5.3f}, {high:>5.3f}]{verdict:>14}")
    print("\nn is prompts with both a cycle score and an adjudicated image;")
    print("'external' is the share of them that were externally correct.")


if __name__ == "__main__":
    main()
