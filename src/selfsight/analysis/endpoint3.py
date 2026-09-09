"""E3 endpoint 3: dose-response between context effect and selection gain.

Secondary, and registered before any of it was measurable. The offline replay
found the two corpus runs differing five-fold in selection gain (+0.190 vs
+0.034) with the difference tracking the strength of the confirmation bias
itself (context effect -0.440 vs -0.214 on conflict trials). The registered
prediction is that this is a dose: **plot one point per (arm, checkpoint),
x = the context effect there, y = the selection gain there, and the slope is
positive.** Deviation 9.4 scales the point set to five seeds and moves the
cluster unit from checkpoint to (seed, checkpoint).

Deviation 13.2 pins the 口径 to `review-packets/bsv-selection-20260908/`
`bsv_selection.py` rather than deriving it again. That file is frozen, so
this module restates its rules and `tests/test_endpoint3.py` checks the two
agree on the same inputs -- a restatement that can drift is worse than an
import, and importing out of a review packet is worse than both.

The registered downgrade is as binding as the prediction. If the slope is not
significantly positive, the heterogeneity is written up as an **unexplained
moderation** and not as a dose-response. `Endpoint3Verdict.wording` carries
that sentence so it cannot be lost between here and the paper.

Both axes need the `blind` and `prompted` conditions over four candidates and
the detector's question set, none of which the run records. The pass that
measures them is `scripts/v4_e3_selection_observe.py`; if it does not finish,
deviation 13.2 point 7 says endpoint 3 reads *not done* and the corpus
replay is not allowed to stand in for per-checkpoint points.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Deviation 13.2 point 1, verbatim from bsv_selection.py. `gold_source ==
# "image"` means the question asks about something the description never
# mentioned, so there is no description-implied answer and a selection loop
# could not have scored it.
USABLE_GOLD = ("spec_matches_image", "image_differs_from_spec")

# Deviation 13.2 point 2.
CONFLICT_GOLD = "image_differs_from_spec"

# Deviation 13.2 point 5, which is deviation 9.4's change: the cluster unit is
# (seed, checkpoint), resampled seeds first and then checkpoints within them.
FINAL_RESAMPLES = 20000
FINAL_BOOTSTRAP_SEED = 20260908

# Deviation 13.2 point 4: a pool of one is not a choice.
MIN_CANDIDATES = 2

CONDITIONS = ("blind", "prompted")

DOWNGRADE = ("unexplained moderation, not a dose-response "
             "(pre-registration, endpoint 3's registered downgrade)")


def spec_agreement(row: dict) -> bool:
    """Did the model answer the way the description implies, whatever the pixels say?

    Restated from `bsv_selection.py`. Every family is a forced binary choice
    and the pipeline has already normalised the answer into `correct` against
    the detections, so the model's choice is recoverable: it agreed with the
    detections iff `correct`, and the description agrees with the detections
    iff `gold_source` is `spec_matches_image`.

    This is the quantity a self-selection loop actually computes. That loop
    never sees the detections; it asks what the description implies and counts
    matches.
    """

    if row["gold_source"] == "spec_matches_image":
        return bool(row["correct"])
    return not bool(row["correct"])


def mean_agreement(answers: list[tuple[bool, bool]]) -> float:
    """Mean spec agreement over a candidate's usable trials, NaN when there are none."""

    return sum(hit for hit, _ in answers) / len(answers) if answers else float("nan")


def pick(pool: list[dict], key: str, *, first_index: bool = False) -> float:
    """Expected external correctness of the candidate this rule keeps.

    Ties are constant -- roughly four binary questions per candidate -- and
    deviation 13.2 point 4 resolves them as the expectation under a uniform
    random tie-break, which is what a loop that shuffles its pool gets on
    average. The first-index variant is reported beside it, not instead.
    """

    best = max(item[key] for item in pool)
    tied = [item for item in pool if item[key] == best]
    if first_index:
        return float(tied[0]["correct"])
    return sum(item["correct"] for item in tied) / len(tied)


@dataclass(frozen=True)
class CheckpointPoint:
    """One (seed, arm, checkpoint): the dose on x, the response on y."""

    seed: int
    arm: str
    step: int
    context_effect: float
    selection_gain: float
    selection_gain_first_index: float
    pools: int
    candidates: int
    conflict_trials: dict[str, int] = field(default_factory=dict)

    @property
    def cluster(self) -> tuple[int, int]:
        """Deviation 13.2 point 5: seed then checkpoint, both arms inside."""

        return (self.seed, self.step)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_pools(rows: list[dict]) -> list[list[dict]]:
    """Group answered questions into per-prompt candidate pools.

    Abstentions and `gold_source == "image"` rows drop out per deviation 13.2
    point 1. A candidate that loses every trial that way has no score and
    leaves; a prompt left with fewer than two candidates is no longer a choice
    and leaves with it.
    """

    per_candidate: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"blind": [], "prompted": [], "correct": None})
    for row in rows:
        if row.get("abstain") or row["gold_source"] not in USABLE_GOLD:
            continue
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Unknown condition {row['condition']!r}; "
                             f"endpoint 3 needs both of {CONDITIONS}")
        key = (str(row["spec_id"]), int(row["candidate_index"]))
        record = per_candidate[key]
        record[row["condition"]].append((spec_agreement(row),
                                         row["gold_source"] == CONFLICT_GOLD))
        correct = bool(row["image_correct"])
        if record["correct"] is None:
            record["correct"] = correct
        elif record["correct"] != correct:
            raise ValueError(f"{key} is both externally correct and incorrect")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for (spec_id, index), record in sorted(per_candidate.items()):
        # Both conditions or neither: a candidate scored blind but not prompted
        # would enter one side of the selection difference and not the other.
        if not record["blind"] or not record["prompted"]:
            continue
        grouped[spec_id].append({
            "index": index,
            "blind": mean_agreement(record["blind"]),
            "prompted": mean_agreement(record["prompted"]),
            "correct": record["correct"],
            "asked": len(record["blind"]),
            "conflicts": sum(conflict for _, conflict in record["blind"]),
        })
    return [sorted(pool, key=lambda item: item["index"])
            for _, pool in sorted(grouped.items()) if len(pool) >= MIN_CANDIDATES]


def context_effect(rows: list[dict]) -> tuple[float, dict[str, int]]:
    """Deviation 13.2 point 3: blind minus prompted `correct` on conflict trials.

    `correct` here is agreement with the *pixels*, which on a conflict trial is
    the opposite of agreeing with the description. So this is the same batch of
    answers the selection score reads, read the other way round -- one
    measurement, two projections, not two measurements.
    """

    counts = {f"{condition}_{suffix}": 0
              for condition in CONDITIONS for suffix in ("trials", "correct")}
    for row in rows:
        if row.get("abstain") or row["gold_source"] != CONFLICT_GOLD:
            continue
        counts[f"{row['condition']}_trials"] += 1
        counts[f"{row['condition']}_correct"] += bool(row["correct"])
    rates = {}
    for condition in CONDITIONS:
        trials = counts[f"{condition}_trials"]
        rates[condition] = counts[f"{condition}_correct"] / trials if trials else float("nan")
    return rates["blind"] - rates["prompted"], counts


def checkpoint_point(rows: list[dict], *, seed: int, arm: str, step: int) -> CheckpointPoint:
    pools = build_pools(rows)
    if not pools:
        raise ValueError(f"seed {seed} {arm} step {step}: no prompt kept "
                         f"{MIN_CANDIDATES} scorable candidates")
    dose, counts = context_effect(rows)
    gain = sum(pick(pool, "blind") - pick(pool, "prompted") for pool in pools) / len(pools)
    strict = sum(pick(pool, "blind", first_index=True)
                 - pick(pool, "prompted", first_index=True) for pool in pools) / len(pools)
    return CheckpointPoint(seed=seed, arm=arm, step=step,
                           context_effect=dose, selection_gain=gain,
                           selection_gain_first_index=strict,
                           pools=len(pools),
                           candidates=sum(len(pool) for pool in pools),
                           conflict_trials=counts)


def fit_slope(points: list[CheckpointPoint], *, first_index: bool = False) -> float:
    """Least squares slope of selection gain on context effect.

    `first_index` swaps the y axis to the strict tie-break. Section 3 registers
    the uniform-random expectation as primary and the first index as a
    robustness contrast, and says to report both; deviation 15.3 reads that as
    both fits rather than both columns, because a slope and an interval are
    what endpoint 3 reports.

    NaN when x has no spread. That is arithmetic rather than a rule: with every
    checkpoint at the same dose there is no dose-response to measure, and the
    caller counts how often it happened instead of choosing a threshold nobody
    registered.

    Degeneracy is tested as `x.min() == x.max()` on the raw doses, *before*
    centring, and not as a zero sum of squares afterwards. Three copies of 0.2
    do not centre to zero in binary floating point -- their mean is
    0.20000000000000004 -- so the sum of squares comes out around 2e-33 and the
    slope comes out around -0.33: a confident number from no information. The
    degenerate resample deviation 13.2 point 5 has to survive is exactly the
    one that repeats a cluster, so the values there are bit-identical and the
    exact comparison is the right one.
    """

    if len(points) < 2:
        return float("nan")
    x = np.array([point.context_effect for point in points], dtype=float)
    y = np.array([point.selection_gain_first_index if first_index
                  else point.selection_gain for point in points], dtype=float)
    usable = ~(np.isnan(x) | np.isnan(y))
    if int(usable.sum()) < 2:
        return float("nan")
    x, y = x[usable], y[usable]
    if x.min() == x.max():
        return float("nan")
    centred = x - x.mean()
    spread = float((centred ** 2).sum())
    if spread == 0.0:
        return float("nan")
    return float((centred * (y - y.mean())).sum() / spread)


@dataclass(frozen=True)
class SlopeDraws:
    slopes: np.ndarray
    slopes_first_index: np.ndarray
    discarded_resamples: int
    resamples: int
    seeds: int
    clusters: int


def nested_bootstrap(points: list[CheckpointPoint], *,
                     resamples: int = FINAL_RESAMPLES,
                     seed: int = FINAL_BOOTSTRAP_SEED) -> SlopeDraws:
    """Deviation 13.2 point 5: resample seeds, then checkpoints within a seed.

    Both arms of a drawn (seed, checkpoint) travel together, because that pair
    is the cluster and the two arms at one checkpoint of one run are the thing
    that is not independent. Resampling the 120 points flat would treat them as
    120 independent observations of the dose, which they are not: a seed that
    happens to sit at a strong context effect contributes twelve of them.
    """

    by_seed: dict[int, dict[int, list[CheckpointPoint]]] = defaultdict(
        lambda: defaultdict(list))
    for point in points:
        by_seed[point.seed][point.step].append(point)
    seeds = sorted(by_seed)
    if not seeds:
        raise ValueError("Endpoint 3 needs at least one seed")
    steps_by_seed = {value: sorted(by_seed[value]) for value in seeds}
    clusters = sum(len(steps) for steps in steps_by_seed.values())

    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    strict: list[float] = []
    discarded = 0
    for _ in range(resamples):
        drawn: list[CheckpointPoint] = []
        for index in rng.integers(0, len(seeds), size=len(seeds)):
            chosen = seeds[int(index)]
            steps = steps_by_seed[chosen]
            for inner in rng.integers(0, len(steps), size=len(steps)):
                drawn.extend(by_seed[chosen][steps[int(inner)]])
        slope = fit_slope(drawn)
        # Both tie-breaks come off the same draw, so the two intervals are a
        # contrast between calibers and not between resamples. Either failing
        # voids the draw for both, the way endpoint 2 voids a pair when either
        # arm's slope is undefined -- otherwise the robustness fit would quietly
        # rest on a different set of resamples than the one it is compared to.
        strict_slope = fit_slope(drawn, first_index=True)
        if np.isnan(slope) or np.isnan(strict_slope):
            discarded += 1
            continue
        slopes.append(slope)
        strict.append(strict_slope)
    if not slopes:
        raise ValueError(f"All {resamples} resamples were discarded; nothing to fit")
    return SlopeDraws(slopes=np.array(slopes), slopes_first_index=np.array(strict),
                      discarded_resamples=discarded,
                      resamples=resamples, seeds=len(seeds), clusters=clusters)


@dataclass(frozen=True)
class Endpoint3Verdict:
    """The registered prediction and the registered downgrade, kept together."""

    slope: float
    interval: tuple[float, float]
    dose_response: bool
    wording: str
    slope_first_index: float
    interval_first_index: tuple[float, float]
    dose_response_first_index: bool
    points: int
    seeds: int
    clusters: int
    discarded_resamples: int

    @property
    def tie_breaks_agree(self) -> bool:
        """Section 3 says report both; deviation 15.3 says say so when they part.

        The verdict is the primary fit either way -- section 3 writes the
        uniform-random expectation as primary and the first index as the
        robustness contrast, so a disagreement is a sentence in the paper and
        not a second chance at the decision.
        """

        return self.dose_response == self.dose_response_first_index


def verdict(points: list[CheckpointPoint], draws: SlopeDraws) -> Endpoint3Verdict:
    """Slope > 0, read the way endpoints 1 and 2 read an interval.

    Significantly positive is the 2.5th percentile of the bootstrap slopes
    lying above zero. Anything else takes the registered downgrade, including
    a slope that is large and positive with an interval that spans zero --
    "the point estimate is encouraging" is not a clause the pre-registration
    contains.

    The first-index fit is computed and reported beside it (section 3, read by
    deviation 15.3) and decides nothing.
    """

    slope = fit_slope(points)
    low, high = (float(value) for value in np.percentile(draws.slopes, [2.5, 97.5]))
    positive = low > 0.0
    strict = fit_slope(points, first_index=True)
    strict_low, strict_high = (float(value) for value
                               in np.percentile(draws.slopes_first_index, [2.5, 97.5]))
    return Endpoint3Verdict(
        slope=slope, interval=(low, high), dose_response=positive,
        wording=("dose-response: selection gain scales with the context effect"
                 if positive else DOWNGRADE),
        slope_first_index=strict, interval_first_index=(strict_low, strict_high),
        dose_response_first_index=strict_low > 0.0,
        points=len(points), seeds=draws.seeds, clusters=draws.clusters,
        discarded_resamples=draws.discarded_resamples)


def selection_path(run: Path, arm: str, step: int) -> Path:
    """Where `scripts/v4_e3_selection_observe.py` puts one checkpoint's answers."""

    return Path(run) / "analysis" / "selection" / arm / f"step-{step:05d}.jsonl"


def available_steps(run: Path, arm: str) -> list[int]:
    directory = Path(run) / "analysis" / "selection" / arm
    if not directory.is_dir():
        return []
    return sorted(int(child.stem.split("-")[1]) for child in directory.glob("step-*.jsonl"))


def load_point(run: Path, arm: str, step: int, *, seed: int) -> CheckpointPoint:
    path = selection_path(run, arm, step)
    if not path.exists():
        raise FileNotFoundError(f"No selection pass for {arm} at step {step}: {path}")
    return checkpoint_point(_rows(path), seed=seed, arm=arm, step=step)
