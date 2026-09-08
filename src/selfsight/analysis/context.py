"""What the generating description does to a model looking at its own output.

Written 2026-09-08, before `runs/v4/decoupling-main-20260908` produced a single
checkpoint metric, and registered in `.planning/2026-09-08-iclr-redesign/PREREG.md`
as the analysis for endpoints 2 and 3. The cross-sectional version of the same
arithmetic is `review-packets/bsv-selection-20260908/bsv_selection.py`, which is
frozen; this is the module the curve uses, so the two cannot drift apart by
someone editing one of them.

The inputs are answer rows in the schema `v4_run_pipeline.py observe` writes:
one row per (image, question), carrying `correct` (graded against the
detections), `gold_source` (whether the description and the pixels agree),
`image_correct` (the adjudicated verdict on the whole image) and, for a curve,
`checkpoint`.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

CONFLICT = "image_differs_from_spec"
AGREEMENT = "spec_matches_image"
SCORABLE = (AGREEMENT, CONFLICT)


def spec_agreement(row: dict) -> bool:
    """Did the model answer the way the description implies, whatever the pixels say?

    This is what a self-selection loop computes. It never sees the detections;
    it asks what the description implies and counts matches. Here it is
    recovered from `correct` rather than by re-parsing the answer, because the
    pipeline already normalised the free text once and a second parser would be
    a second set of bugs: the model agreed with the detections iff `correct`,
    and the description agrees with the detections iff the trial is an AGREEMENT
    trial.
    """

    if row["gold_source"] not in SCORABLE:
        raise ValueError(f"{row['gold_source']} trials have no description-implied answer")
    return bool(row["correct"]) if row["gold_source"] == AGREEMENT else not bool(row["correct"])


def accuracy(rows: Iterable[dict], *, gold_source: str | None = None) -> float:
    """Accuracy against the pixels -- the ordinary reading of `correct`."""

    kept = [row for row in rows
            if not row["abstain"] and (gold_source is None or row["gold_source"] == gold_source)]
    return sum(bool(row["correct"]) for row in kept) / len(kept) if kept else float("nan")


def context_effect(blind: Sequence[dict], prompted: Sequence[dict]) -> float:
    """How much perception the description costs, on the trials that can show it.

    Positive means blind reads the picture better. On AGREEMENT trials the two
    conditions cannot separate -- both hypotheses predict the same answer -- so
    only CONFLICT trials enter. This is the quantity that measured -0.318
    cross-sectionally.
    """

    return accuracy(blind, gold_source=CONFLICT) - accuracy(prompted, gold_source=CONFLICT)


def pools(blind: Sequence[dict], prompted: Sequence[dict]) -> dict:
    """Group the two conditions into the choice a selection loop actually faces.

    Keyed by spec: the loop draws K images for one description and keeps one.
    A candidate the two conditions did not both answer about is dropped rather
    than scored on half the evidence.
    """

    scored: dict = defaultdict(lambda: {"blind": [], "prompted": [], "correct": None})
    for condition, rows in (("blind", blind), ("prompted", prompted)):
        for row in rows:
            if row["abstain"] or row["gold_source"] not in SCORABLE:
                continue
            record = scored[(row["spec_id"], row["candidate_index"])]
            record[condition].append(spec_agreement(row))
            if record["correct"] is None:
                record["correct"] = bool(row["image_correct"])
            elif record["correct"] != bool(row["image_correct"]):
                raise ValueError(f"{row['spec_id']}/{row['candidate_index']} is both right and wrong")

    grouped: dict = defaultdict(list)
    for (spec_id, index), record in scored.items():
        if not record["blind"] or not record["prompted"]:
            continue
        grouped[spec_id].append({
            "index": index,
            "blind": sum(record["blind"]) / len(record["blind"]),
            "prompted": sum(record["prompted"]) / len(record["prompted"]),
            "correct": record["correct"],
        })
    return {spec_id: sorted(pool, key=lambda item: item["index"])
            for spec_id, pool in grouped.items() if len(pool) >= 2}


def kept(pool: Sequence[dict], key: str) -> float:
    """Expected correctness of the candidate this rule keeps.

    Ties are the common case -- four binary questions give five distinct scores
    -- and are resolved as the expectation under a uniform random tie-break,
    which is what an implementation that shuffles its pool gets on average.
    """

    best = max(item[key] for item in pool)
    tied = [item for item in pool if item[key] == best]
    return sum(item["correct"] for item in tied) / len(tied)


def selection_gain(pools_by_spec: dict) -> float:
    """External correctness of blind selection minus that of prompted selection."""

    if not pools_by_spec:
        return float("nan")
    values = [kept(pool, "blind") - kept(pool, "prompted") for pool in pools_by_spec.values()]
    return sum(values) / len(values)


def discrimination_gap(pools_by_spec: dict, key: str = "blind") -> float:
    """Does the score tell a right image from a wrong one, per candidate?

    Endpoint 2. The mean score of the candidates that are actually correct minus
    the mean score of those that are not, over candidates rather than over
    pools, so a checkpoint whose pools are all-correct still contributes what it
    has. NaN when one side is empty: a gap needs both.
    """

    right = [item[key] for pool in pools_by_spec.values() for item in pool if item["correct"]]
    wrong = [item[key] for pool in pools_by_spec.values() for item in pool if not item["correct"]]
    if not right or not wrong:
        return float("nan")
    return sum(right) / len(right) - sum(wrong) / len(wrong)


def ols(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Slope and intercept, the two-variable case written out.

    numpy is in the environment; this is here so the registered analysis has no
    dependency that could resolve differently a year from now.
    """

    usable = [(x, y) for x, y in points if not (math.isnan(x) or math.isnan(y))]
    if len(usable) < 2:
        return float("nan"), float("nan")
    mean_x = sum(x for x, _ in usable) / len(usable)
    mean_y = sum(y for _, y in usable) / len(usable)
    spread = sum((x - mean_x) ** 2 for x, _ in usable)
    if spread == 0:
        return float("nan"), float("nan")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in usable) / spread
    return slope, mean_y - slope * mean_x


def dose_response(points: Sequence[tuple[float, float]], *, draws: int = 20000,
                  seed: int = 20260908) -> dict:
    """Endpoint 3: does a stronger confirmation bias buy a bigger selection gain?

    One point per (arm, checkpoint). The bootstrap resamples checkpoints, which
    is the honest unit -- the images inside one checkpoint are not independent
    evidence about the slope -- and with 26 points it is a wide interval by
    construction. That width is the prediction's cost, and it was accepted in
    the prereg rather than discovered afterwards.
    """

    import random

    slope, intercept = ols(points)
    usable = [(x, y) for x, y in points if not (math.isnan(x) or math.isnan(y))]
    rng = random.Random(seed)
    slopes = []
    for _ in range(draws):
        sample = [usable[rng.randrange(len(usable))] for _ in range(len(usable))]
        drawn, _ = ols(sample)
        if not math.isnan(drawn):
            slopes.append(drawn)
    slopes.sort()
    low = slopes[int(0.025 * len(slopes))] if slopes else float("nan")
    high = slopes[int(0.975 * len(slopes))] if slopes else float("nan")
    return {"slope": slope, "intercept": intercept, "low": low, "high": high,
            "n": len(usable)}


# --------------------------------------------------------------------- E4
#
# The cross-model replication reports the same two numbers the cross-sectional
# result did, so the three models can sit in one table with the 1.5B. Both are
# byte-for-byte the estimators in
# `review-packets/context-ablation-20260908/context_ablation.py`, moved here
# rather than imported: that packet is frozen, and a frozen file is not a
# library. `test_context_analysis.py` checks the two agree on the real rows.


def wilson(hits: int, total: int) -> tuple[float, float, float]:
    """Wilson interval: the accuracies here run close to 1.0, where Wald lies."""

    if total == 0:
        return float("nan"), float("nan"), float("nan")
    point = hits / total
    z = 1.96
    denominator = 1 + z**2 / total
    centre = (point + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(point * (1 - point) / total + z**2 / (4 * total**2)) / denominator
    return point, centre - margin, centre + margin


def mcnemar(pairs: Sequence[tuple[bool, bool]], *, sided: int = 2) -> tuple[int, int, float]:
    """Exact McNemar on the discordant pairs.

    Two-sided reproduces the frozen packet. E4 registered a direction --
    prompted below image_only -- so it reads the one-sided tail, and takes the
    tail on the registered side rather than on whichever side is smaller: a
    result that goes the wrong way must come back with a large p, not a small
    one with a sign attached.
    """

    only_first = sum(1 for a, b in pairs if a and not b)
    only_second = sum(1 for a, b in pairs if b and not a)
    n = only_first + only_second
    if n == 0:
        return only_first, only_second, 1.0
    if sided == 2:
        k = min(only_first, only_second)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        return only_first, only_second, min(1.0, 2 * tail)
    if sided != 1:
        raise ValueError(f"sided must be 1 or 2, not {sided}")
    # P(second wins this few or fewer | fair coin): small exactly when the
    # first condition -- blind -- is the one that won.
    tail = sum(math.comb(n, i) for i in range(only_second + 1)) / 2**n
    return only_first, only_second, min(1.0, tail)


def conflict_pairs(blind: Sequence[dict], prompted: Sequence[dict]) -> list[tuple[bool, bool]]:
    """(blind correct, prompted correct) per decisive trial both conditions answered.

    Keyed by (image, question): the same picture and the same words, differing
    only in whether the description was in the context. A trial either side
    abstained on is dropped rather than counted as wrong -- an abstention is
    the model declining to answer, and scoring it as an error would let the
    prompted condition look worse for being more cautious.
    """

    def index(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
        return {(row["image_path"], row["question"]): row
                for row in rows
                if not row["abstain"] and row["gold_source"] == CONFLICT}

    left, right = index(blind), index(prompted)
    return [(bool(left[key]["correct"]), bool(right[key]["correct"]))
            for key in sorted(left.keys() & right.keys())]
