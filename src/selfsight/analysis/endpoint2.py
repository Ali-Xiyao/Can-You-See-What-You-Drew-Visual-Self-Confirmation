"""E3 endpoint 2: the blind discrimination gap, and whether training flattens it.

The registered sentence is: fit the gap against training step, one OLS slope
per arm, paired bootstrap over prompt; **confirmed when A's slope is
significantly negative and B's slope is significantly greater than A's**.

Four amendments this module implements literally.

- **deviation 11.1** -- the external label comes from `candidate_index == 0`,
  the draw the self-report was actually computed on. `s_select.jsonl` holds
  one row per prompt and `internal_curve_scope` is `first_draw_only`, so
  taking the label from any other candidate pairs one image's self-report
  with another image's verdict.
- **deviation 11.3** -- the score is the `image_only` condition, measured by
  the offline blind pass. The prompted score the run already holds is
  reported beside it and never substituted for it; if the blind pass did not
  run, endpoint 2 reads *not done*, which is why `load_series` raises
  `BlindPassMissing` rather than falling back to what is on disk.
- **deviation 13.1** -- the seven rules that turn "paired bootstrap over
  prompt" into a number: one resampled prompt multiset shared by both arms
  and every checkpoint, percentile readings for "significant", the
  20000/20260908 branch, degenerate-resample handling, `step` as the
  abscissa, and missing checkpoints reported rather than dropped.
- **deviation 9.4** -- across the five seeds, A's slope significantly
  negative in at least 4, *and* a one-sided exact sign test on the five
  `slope_B - slope_A` values. Bootstrapping those five is banned.

The bootstrap pairs across arms, not within a checkpoint. Inside one
checkpoint a prompt sits either on the right side of the gap or on the wrong
side and cannot be paired with itself, so the only place a pair can live is
between two arms scored on the same resampled prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from selfsight.analysis.endpoint1 import _verdicts

# Deviation 13.1 point 3. Deviation 6.4 split the bootstrap into a
# running-report branch (2000 / 20260906) and a final-analysis branch
# (20000 / 20260908); the confirmatory slope fit is a final analysis and takes
# the second. That is a *reading* of 6.4 rather than a restatement of it,
# which is why 13.1 registers it instead of this module inheriting it quietly.
FINAL_RESAMPLES = 20000
FINAL_BOOTSTRAP_SEED = 20260908

# Deviation 13.1 point 4. A guard, not a policy: with 64 prompts and roughly a
# third of them externally correct, a resample that empties one side of the
# gap has probability around 1e-11. It is written down so that the arithmetic
# cannot quietly turn into a NaN.
MIN_CHECKPOINTS_PER_FIT = 3

# Deviation 9.4. Both halves are required and neither is a fallback for the
# other; ALPHA applies to the sign test, not to the per-seed percentile reads.
A_SLOPE_MAJORITY = 4
REGISTERED_SEED_COUNT = 5
ALPHA = 0.05

# Deviation 11.3. The blind pass writes exactly this condition and the
# analysis refuses anything else, so a prompted file copied into place under a
# blind file's name is an error rather than a silent substitution.
BLIND_CONDITION = "image_only"

# The registered consequence of failure condition 2, kept as a string so the
# driver prints it rather than leaving it to whoever reads the slopes.
# Deviation 15.1: "B also collapses" had no code path at all -- not the
# sentence, and not the quantity the sentence is about.
FALSIFIED = ("BSV does not prevent perception from being eaten: arm B's blind "
             "discrimination gap collapses too, so the mechanism in section 2 "
             "is falsified and the paper must say so "
             "(pre-registration, failure condition 2)")

# Scores are floats; labels are 1, 0, or this.
ABSENT = -1


class BlindPassMissing(FileNotFoundError):
    """The `image_only` pass has not been run for this arm and step.

    A distinct type because deviation 11.3 gives it a distinct fate: endpoint
    2 reads *not done*. Catching FileNotFoundError generally would let a
    missing manifest -- a different and much worse problem -- take the same
    branch.
    """


@dataclass(frozen=True)
class BlindSeries:
    """One arm's blind pass: prompts down, checkpoints across.

    `scores[i, j]` is prompt i's blind `s_select` at step j, and
    `labels[i, j]` is 1, 0, or ABSENT. Absent is a real third state --
    deviation 12's vanished row and deviation 10's undecided row both land
    here -- and it is kept as a value rather than a NaN so that a prompt can
    drop out of one checkpoint without dropping out of the series.
    """

    arm: str
    steps: tuple[int, ...]
    prompts: tuple[str, ...]
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.prompts), len(self.steps))
        if self.scores.shape != shape or self.labels.shape != shape:
            raise ValueError(f"{self.arm}: expected {shape}, got scores "
                             f"{self.scores.shape} and labels {self.labels.shape}")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError(f"{self.arm}: a step appears twice in {self.steps}")
        if list(self.steps) != sorted(self.steps):
            raise ValueError(f"{self.arm}: steps out of order: {self.steps}")
        if len(set(self.prompts)) != len(self.prompts):
            raise ValueError(f"{self.arm}: a prompt appears twice")


def gap_series(series: BlindSeries, rows: np.ndarray) -> np.ndarray:
    """Every checkpoint's gap, over the given (possibly repeated) prompt rows.

    `rows` is an index array, so a bootstrap resample that draws a prompt
    twice counts it twice. A checkpoint with nothing on one side under that
    resample yields NaN, which is deviation 13.1 point 4's undefined gap.
    """

    scores, labels = series.scores[rows], series.labels[rows]
    out = np.full(len(series.steps), np.nan)
    for column in range(len(series.steps)):
        column_scores, column_labels = scores[:, column], labels[:, column]
        right = column_scores[column_labels == 1]
        wrong = column_scores[column_labels == 0]
        if len(right) and len(wrong):
            out[column] = float(right.mean() - wrong.mean())
    return out


def ols_slope(steps: np.ndarray, gaps: np.ndarray) -> float:
    """Least squares slope of gap on step, undefined checkpoints dropped.

    Deviation 13.1 point 5 fixes the abscissa as the step number rather than
    the checkpoint index. The two are equally spaced so nothing here changes,
    which is exactly why it is written down: an equivalence is the kind of
    choice that gets made differently the second time.

    Point 4's per-arm reading lives here. The registered text says a
    checkpoint whose gap is undefined leaves *that* fit, and the other arm's
    gap at the same checkpoint is perfectly well defined, so the two arms can
    end up fitted on different checkpoint sets. Dropping it from both would
    also be defensible; at 1e-11 neither can move a published number, and the
    literal reading is the one that does not invent a rule.
    """

    usable = ~np.isnan(gaps)
    if int(usable.sum()) < MIN_CHECKPOINTS_PER_FIT:
        return float("nan")
    x = steps[usable].astype(float)
    y = gaps[usable]
    centred = x - x.mean()
    spread = float((centred ** 2).sum())
    if spread == 0.0:
        return float("nan")
    return float((centred * (y - y.mean())).sum() / spread)


@dataclass(frozen=True)
class SlopeDraws:
    """What one seed's bootstrap produced, and what it had to throw away."""

    arm_a: np.ndarray
    arm_b: np.ndarray
    difference: np.ndarray
    dropped_checkpoints: int
    discarded_resamples: int
    resamples: int


def paired_bootstrap(
    series_a: BlindSeries,
    series_b: BlindSeries,
    *,
    resamples: int = FINAL_RESAMPLES,
    seed: int = FINAL_BOOTSTRAP_SEED,
) -> SlopeDraws:
    """Deviation 13.1 point 1: one prompt multiset, both arms, every checkpoint.

    The multiset is drawn once per resample and reused across arms and across
    checkpoints. Drawing it per arm would break the pairing the registered
    text asks for; drawing it per checkpoint would additionally destroy the
    within-arm correlation that makes a slope interval narrower than the
    interval on any one checkpoint's gap.
    """

    if series_a.prompts != series_b.prompts:
        raise ValueError("The arms must be scored on the same prompts, in the same order")
    if series_a.steps != series_b.steps:
        raise ValueError(f"The arms must be scored at the same steps: "
                         f"{series_a.steps} vs {series_b.steps}")
    steps = np.asarray(series_a.steps)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(series_a.prompts), size=(resamples, len(series_a.prompts)))
    left: list[float] = []
    right: list[float] = []
    dropped = 0
    discarded = 0
    for row in picks:
        gaps_a, gaps_b = gap_series(series_a, row), gap_series(series_b, row)
        dropped += int(np.isnan(gaps_a).sum()) + int(np.isnan(gaps_b).sum())
        slope_a, slope_b = ols_slope(steps, gaps_a), ols_slope(steps, gaps_b)
        if np.isnan(slope_a) or np.isnan(slope_b):
            # Either arm failing voids the pair, because the quantity being
            # paired is the difference of the two slopes.
            discarded += 1
            continue
        left.append(slope_a)
        right.append(slope_b)
    if not left:
        raise ValueError(f"All {resamples} resamples were discarded; nothing to fit")
    arm_a, arm_b = np.array(left), np.array(right)
    return SlopeDraws(arm_a=arm_a, arm_b=arm_b, difference=arm_b - arm_a,
                      dropped_checkpoints=dropped, discarded_resamples=discarded,
                      resamples=resamples)


@dataclass(frozen=True)
class SeedVerdict:
    """One seed's half of endpoint 2, with the two clauses kept apart."""

    seed: int
    steps: tuple[int, ...]
    missing_steps: tuple[int, ...]
    slope_a: float
    slope_b: float
    a_interval: tuple[float, float]
    b_interval: tuple[float, float]
    difference_interval: tuple[float, float]
    a_collapses: bool
    b_collapses: bool
    b_exceeds_a: bool
    dropped_checkpoints: int
    discarded_resamples: int

    @property
    def confirmed(self) -> bool:
        return self.a_collapses and self.b_exceeds_a


def seed_verdict(series_a: BlindSeries, series_b: BlindSeries, draws: SlopeDraws,
                 *, seed: int, missing_steps: tuple[int, ...] = ()) -> SeedVerdict:
    """Deviation 13.1 point 2, applied literally.

    "A's slope significantly negative" is the 97.5th percentile of A's draws
    lying below zero. "B significantly greater than A" is the 2.5th percentile
    of the paired difference lying above zero. Both are one-sided readings of
    a 95% percentile interval, which is how endpoint 1 reads its interval too.

    "B also collapses" is deviation 15.1, and it is the same reading applied to
    B's own draws, on the same resampled prompt multiset. It is not the
    negation of `b_exceeds_a`: B can fail to beat A without collapsing, and
    failure condition 2 is about the collapse and not about the contrast.

    `missing_steps` is carried rather than recomputed: deviation 13.1 point 6
    requires the fit to say how many checkpoints it used and which it could
    not get, and a verdict that silently described a shorter series as the
    whole thing is the failure that rule names.
    """

    steps = np.asarray(series_a.steps)
    every = np.arange(len(series_a.prompts))
    point_a = ols_slope(steps, gap_series(series_a, every))
    point_b = ols_slope(steps, gap_series(series_b, every))
    a_low, a_high = (float(value) for value in np.percentile(draws.arm_a, [2.5, 97.5]))
    b_low, b_high = (float(value) for value in np.percentile(draws.arm_b, [2.5, 97.5]))
    d_low, d_high = (float(value) for value in np.percentile(draws.difference, [2.5, 97.5]))
    return SeedVerdict(
        seed=seed, steps=series_a.steps, missing_steps=missing_steps,
        slope_a=point_a, slope_b=point_b,
        a_interval=(a_low, a_high), b_interval=(b_low, b_high),
        difference_interval=(d_low, d_high),
        a_collapses=a_high < 0.0, b_collapses=b_high < 0.0, b_exceeds_a=d_low > 0.0,
        dropped_checkpoints=draws.dropped_checkpoints,
        discarded_resamples=draws.discarded_resamples)


@dataclass(frozen=True)
class AcrossSeeds:
    """Deviation 9.4's two halves, and the conjunction they have to reach."""

    collapsing: int
    b_collapsing: int
    total: int
    sign_supporting: int
    sign_p: float
    a_half: bool
    b_half: bool
    b_collapse_half: bool
    confirmatory: bool

    @property
    def confirmed(self) -> bool:
        return self.confirmatory and self.a_half and self.b_half

    @property
    def falsified(self) -> bool:
        """Failure condition 2, read at the study level (deviation 15.1).

        The registered sentence is "only A collapsing while B also collapses",
        so A's half has to hold as well: with A intact there was nothing for
        BSV to prevent and the mechanism is not falsified. Confirmed and
        falsified are not each other's negation and both can be false.
        """

        return self.a_half and self.b_collapse_half


def across_seeds(verdicts: list[SeedVerdict], *, alpha: float = ALPHA) -> AcrossSeeds:
    """Deviation 9.4 by way of 13.1 point 7.

    Two halves, both required. A's slope has to be significantly negative in
    at least 4 of the 5 seeds -- that 4 was fixed before any seed ran. The
    other half is the one-sided exact sign test on the five slope differences,
    and deviation 9.3 bans bootstrapping them, so they enter as five signs and
    nothing else. A difference of exactly 0 stays in the denominator and
    counts against, matching `endpoint1.sign_test`.

    `b_collapsing` counts the seeds where B collapses on its own terms, and
    deviation 15.1 reads failure condition 2 off the same 4-of-5 threshold.
    Deviation 14.3 refused to borrow that 4 for deviation 7.2, and the reason
    the two differ is written there: 7.2 is a differently shaped quantity,
    while this is the same endpoint, the same slope and the same seeds with
    the arms swapped. Between 4 and 5 it is also the threshold the unfavourable
    reading reaches more easily.
    """

    if not verdicts:
        raise ValueError("Endpoint 2 across seeds needs at least one seed")
    total = len(verdicts)
    collapsing = sum(1 for verdict in verdicts if verdict.a_collapses)
    b_collapsing = sum(1 for verdict in verdicts if verdict.b_collapses)
    supporting = sum(1 for verdict in verdicts if verdict.slope_b - verdict.slope_a > 0)
    p_value = float(binomtest(supporting, total, 0.5, alternative="greater").pvalue)
    return AcrossSeeds(
        collapsing=collapsing, b_collapsing=b_collapsing, total=total,
        sign_supporting=supporting, sign_p=p_value,
        a_half=collapsing >= A_SLOPE_MAJORITY, b_half=p_value <= alpha,
        b_collapse_half=b_collapsing >= A_SLOPE_MAJORITY,
        confirmatory=total == REGISTERED_SEED_COUNT)


def blind_path(run: Path, arm: str, step: int) -> Path:
    """Where `scripts/v4_e3_blind_observe.py` puts one checkpoint's blind pass."""

    return Path(run) / "analysis" / "blind_observe" / arm / f"step-{step:05d}.jsonl"


def available_steps(run: Path, arm: str) -> list[int]:
    directory = Path(run) / "analysis" / "blind_observe" / arm
    if not directory.is_dir():
        return []
    return sorted(int(child.stem.split("-")[1]) for child in directory.glob("step-*.jsonl"))


def load_series(run: Path, arm: str, steps: list[int]) -> BlindSeries:
    """Read one arm's blind pass and attach deviation 11.1's external labels.

    The prompt axis is the union over the requested steps, sorted, so a prompt
    the blind pass missed at one checkpoint keeps its row and is ABSENT there
    rather than deleting itself from the whole series.
    """

    if not steps:
        raise ValueError(f"{arm}: no steps requested")
    per_step: dict[int, dict[str, float]] = {}
    labels_per_step: dict[int, dict[str, bool | None]] = {}
    for step in steps:
        path = blind_path(run, arm, step)
        if not path.exists():
            raise BlindPassMissing(f"No image_only pass for arm {arm} at step {step}: {path}")
        scores: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") != BLIND_CONDITION:
                raise ValueError(f"{path}: condition is {row.get('condition')!r}, not "
                                 f"{BLIND_CONDITION!r}; deviation 11.3 forbids substituting "
                                 f"the prompted score")
            if int(row.get("candidate_index", 0)) != 0:
                raise ValueError(f"{path}: candidate_index {row['candidate_index']}; "
                                 f"deviation 11.1 fixes the blind pass to candidate 0")
            prompt = str(row["prompt_id"])
            if prompt in scores:
                raise ValueError(f"{path}: duplicate row for prompt {prompt}")
            scores[prompt] = float(row["s_select"])
        per_step[step] = scores
        verified = Path(run) / "evaluations" / arm / f"step-{step:05d}" / "verified.jsonl"
        if not verified.exists():
            raise FileNotFoundError(f"No verdicts for arm {arm} at step {step}: {verified}")
        labels_per_step[step] = {key[0]: value
                                 for key, value in _verdicts(verified).items()
                                 if key[1] == 0}

    prompts = tuple(sorted({prompt for scores in per_step.values() for prompt in scores}))
    scores_out = np.full((len(prompts), len(steps)), np.nan)
    labels_out = np.full((len(prompts), len(steps)), ABSENT, dtype=int)
    for column, step in enumerate(steps):
        for index, prompt in enumerate(prompts):
            score = per_step[step].get(prompt)
            label = labels_per_step[step].get(prompt)
            # Deviation 12: a vanished verdict and an undecided one arrive here
            # as the same absence and get the same fate, and a score with no
            # label is as useless as a label with no score.
            if score is None or label is None:
                continue
            scores_out[index, column] = score
            labels_out[index, column] = int(bool(label))
    return BlindSeries(arm=arm, steps=tuple(steps), prompts=prompts,
                       scores=scores_out, labels=labels_out)
