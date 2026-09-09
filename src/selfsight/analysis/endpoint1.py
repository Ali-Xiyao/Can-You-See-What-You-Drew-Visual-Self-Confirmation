"""E3 endpoint 1: final external correctness, B over A.

Registered in `.planning/2026-09-08-iclr-redesign/PREREG.md` under E3
endpoint 1, with three later amendments this module implements literally:

- **deviation 6.4** -- the final analysis uses 20000 resamples and seed
  20260908. The per-checkpoint descriptive report keeps running at 20260906
  and 2000; those are two branches and neither inherits the other's numbers,
  so the defaults here are spelled out rather than read from a config.
- **deviation 9.3** -- five seeds feed a one-sided exact sign test on the
  five within-seed paired differences. Bootstrapping those five values is
  banned: a resample distribution over five atoms is a near relative of the
  range, whose type-I error this pipeline has already measured at 9.5%
  against a nominal 5%.
- **deviation 10** -- a (prompt, draw) unadjudicated in either arm is
  dropped from both, because a paired estimand cannot rest on two arms whose
  denominators are different images.
- **deviation 12** -- a (prompt, draw) that is *absent* from an arm's
  verdicts is dropped the same way. The detector returning nothing for an
  image makes the row vanish rather than arrive undecided
  (`skipped_no_detection`), which deviation 10 did not cover and this module
  used to raise on. The two causes are counted separately, and the
  denominator comes from the manifest so that a vanished row cannot take its
  own denominator with it.

The estimand is fixed in advance and so is the threshold: a point estimate
below +0.042 whose interval contains 0 is *not detected*, and there is no
rescue analysis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

# Deviation 6.4. Not read from a config: the running report reads its own
# from config["seed"] and gradient_probe.resamples, and those are the other
# branch. Silently inheriting them is the exact confusion 6.4 was written to
# end, so these are literals with the deviation number attached.
FINAL_RESAMPLES = 20000
FINAL_BOOTSTRAP_SEED = 20260908

# Deviation 10.2 point 4. A disclosure trigger, not an exclusion rule: a
# checkpoint below this still enters the analysis, it just gets flagged.
LOW_COVERAGE_PAIRS = 192

# E3 endpoint 1, frozen before any outcome data existed.
DETECTABLE_EFFECT = 0.042

# evaluate.py's UNADJUDICATED, restated rather than imported: this is the
# analysis side and should not silently track a constant the instrument side
# is free to change. A test pins the two together so the copy cannot drift.
UNADJUDICATED = {"pending_human", "unnameable"}

# The registered consequence of failure condition 1. Endpoint 3 keeps its
# downgrade wording as a constant and deviation 7.2 keeps both of its
# verdicts; this one -- the heaviest of the four -- lived only in the
# pre-registration until deviation 15.2, which is a rule waiting to be
# executed by hand after the number is on screen.
NOT_DETECTED_CONSEQUENCE = (
    "the paper may not claim that BSV improves generation; it may claim only "
    "that BSV improves the quality of the selection signal, which the offline "
    "replay already supports, and the claim retreats to inference time "
    "(pre-registration, failure condition 1)")


@dataclass(frozen=True)
class Coverage:
    """What deviation 10.2 point 4 requires every checkpoint to report."""

    pairs_kept: int
    pairs_total: int
    prompts_kept: int
    prompts_total: int
    unadjudicated_a: int
    unadjudicated_b: int
    # Deviation 12.2 point 2. Not folded into unadjudicated_*: an undecided
    # row is the adjudicator hesitating, an absent row is the detector seeing
    # nothing in the picture. The second is evidence about the image.
    skipped_a: int = 0
    skipped_b: int = 0

    @property
    def low(self) -> bool:
        return self.pairs_kept < LOW_COVERAGE_PAIRS


@dataclass(frozen=True)
class PairedCheckpoint:
    """One (seed, checkpoint): the kept pairs, grouped by prompt.

    `outcomes[prompt_id]` is an (n_draws, 2) array of 0/1, column 0 for arm A
    and column 1 for arm B. Grouping by prompt rather than flattening is not
    presentation -- the bootstrap resamples prompts, so the cluster has to
    survive into the data structure.
    """

    step: int
    outcomes: dict[str, np.ndarray]
    coverage: Coverage

    @property
    def prompts(self) -> list[str]:
        return sorted(self.outcomes)


def _rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _adjudicated(row: dict) -> bool:
    return (row.get("resolution") not in UNADJUDICATED
            and isinstance(row.get("image_correct"), bool))


def _verdicts(path: Path) -> dict[tuple[str, int], bool | None]:
    """(spec_id, candidate_index) -> verdict, or None when unadjudicated.

    None and absent are deliberately different. An image the run never
    produced is a hole in the instrument and raises in the caller; an image
    it produced and could not adjudicate is deviation 10's case and is
    dropped from both arms.
    """

    out: dict[tuple[str, int], bool | None] = {}
    for row in _rows(path):
        key = (str(row["spec_id"]), int(row["candidate_index"]))
        if key in out:
            raise ValueError(f"{path}: duplicate verdict for {key}")
        out[key] = bool(row["image_correct"]) if _adjudicated(row) else None
    return out


def _manifest_keys(run: Path, arm: str, step: int) -> set[tuple[str, int]]:
    """The (prompt, draw) universe the run set out to measure.

    Deviation 12.2 point 3. Counting the denominator off verified.jsonl would
    let a skipped image delete itself from both numerator and denominator, so
    the deletion rate would read 0 however many rows went missing.
    """

    path = Path(run) / "evaluations" / arm / f"step-{step:05d}" / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No manifest for arm {arm} at step {step}: {path}")
    return {(str(row["spec_id"]), int(row["candidate_index"])) for row in _rows(path)}


def completed_steps(run: Path, arm: str) -> list[int]:
    """The checkpoints this arm has an adjudicated outcome set for.

    Endpoints 1 and 2 both need this and need it to mean the same thing: a
    step counts once `verified.jsonl` exists, whatever else in the checkpoint
    directory is or is not finished. It lives here rather than in either
    driver script because two copies would be free to drift, and the two
    endpoints would then disagree about which checkpoints the run has.
    """

    directory = Path(run) / "evaluations" / arm
    if not directory.is_dir():
        return []
    return sorted(int(child.name.split("-")[1]) for child in directory.glob("step-*")
                  if (child / "verified.jsonl").exists())


def load_checkpoint(run: Path, step: int, arm_a: str, arm_b: str) -> PairedCheckpoint:
    """Read one checkpoint from both arms and apply deviation 10's rule."""

    def verdicts(arm: str) -> dict[tuple[str, int], bool | None]:
        path = Path(run) / "evaluations" / arm / f"step-{step:05d}" / "verified.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No verdicts for arm {arm} at step {step}: {path}")
        return _verdicts(path)

    # Deviation 12.2 point 4. Two arms drawing different (prompt, draw) sets
    # is not a detector skipping an image -- it means they did not draw the
    # same things, and endpoint 1's paired estimand has no premise left. That
    # still stops the analysis. It is checked on the manifests, because the
    # verdict files are exactly what a skip is allowed to shorten.
    universe = _manifest_keys(run, arm_a, step)
    other = _manifest_keys(run, arm_b, step)
    if universe != other:
        only_a = sorted(universe - other)[:3]
        only_b = sorted(other - universe)[:3]
        raise ValueError(
            f"step {step}: arms were asked for different (prompt, draw) keys; "
            f"{len(universe ^ other)} differ, e.g. only in {arm_a}: {only_a}, "
            f"only in {arm_b}: {only_b}")

    left, right = verdicts(arm_a), verdicts(arm_b)
    stray = (set(left) | set(right)) - universe
    if stray:
        raise ValueError(
            f"step {step}: {len(stray)} verdict(s) for keys no manifest asked for, "
            f"e.g. {sorted(stray)[:3]}")

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for key in universe:
        # Absent and undecided both arrive here as None, and deviation 12.2
        # point 1 gives them the same fate. They are counted apart below.
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            continue
        grouped[key[0]].append((int(a), int(b)))

    coverage = Coverage(
        pairs_kept=sum(len(rows) for rows in grouped.values()),
        pairs_total=len(universe),
        prompts_kept=len(grouped),
        prompts_total=len({key[0] for key in universe}),
        unadjudicated_a=sum(1 for key, value in left.items()
                            if value is None and key in universe),
        unadjudicated_b=sum(1 for key, value in right.items()
                            if value is None and key in universe),
        skipped_a=len(universe - set(left)),
        skipped_b=len(universe - set(right)),
    )
    outcomes = {prompt: np.asarray(rows, dtype=np.int8) for prompt, rows in grouped.items()}
    return PairedCheckpoint(step=step, outcomes=outcomes, coverage=coverage)


def prompt_differences(checkpoint: PairedCheckpoint) -> np.ndarray:
    """Per-prompt mean of B minus mean of A, in `checkpoint.prompts` order."""

    if not checkpoint.outcomes:
        raise ValueError(f"step {checkpoint.step}: every pair was dropped")
    return np.array([float(checkpoint.outcomes[prompt][:, 1].mean()
                           - checkpoint.outcomes[prompt][:, 0].mean())
                     for prompt in checkpoint.prompts])


def paired_difference(checkpoint: PairedCheckpoint) -> float:
    """theta_B - theta_A, the mean over prompts of the within-prompt mean.

    Deviation 10.2 point 2 fixes this form. With four surviving draws
    everywhere it equals the difference of the two overall rates; once
    deletion makes the draw counts unequal the two part, and the prompt-mean
    form is the one that does not reweight prompts by how many of their
    draws happened to survive adjudication.
    """

    return float(prompt_differences(checkpoint).mean())


def bootstrap_interval(
    checkpoint: PairedCheckpoint,
    *,
    resamples: int = FINAL_RESAMPLES,
    seed: int = FINAL_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile 95% interval, resampling prompts with replacement.

    Prompts, not images: the four draws of one prompt share a latent and a
    spec and are not independent, which E3 endpoint 1 says in as many words.
    """

    per_prompt = prompt_differences(checkpoint)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_prompt), size=(resamples, len(per_prompt)))
    draws = per_prompt[picks].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def exact_mcnemar(checkpoint: PairedCheckpoint) -> tuple[int, int, float]:
    """One-sided exact McNemar, direction B > A. Returns (b_only, a_only, p).

    Registered as-is and reported as-is, with a caveat that belongs printed
    beside it: this treats each (prompt, draw) as independent, and they are
    not. It is anti-conservative relative to the bootstrap interval above,
    which is the clustered one. Both are reported because both were
    registered, not because they are expected to agree.
    """

    stacked = np.concatenate([rows for rows in checkpoint.outcomes.values()])
    arm_a, arm_b = stacked[:, 0], stacked[:, 1]
    b_only = int(np.sum((arm_b == 1) & (arm_a == 0)))
    a_only = int(np.sum((arm_a == 1) & (arm_b == 0)))
    discordant = b_only + a_only
    if discordant == 0:
        return b_only, a_only, 1.0
    return b_only, a_only, float(
        binomtest(b_only, discordant, 0.5, alternative="greater").pvalue)


def sign_test(differences: list[float]) -> tuple[int, int, float]:
    """Deviation 9.3: one-sided exact sign test over seeds, B > A.

    Returns (supporting, total, p). A difference of exactly 0 stays in the
    denominator and counts against -- 9.3 says so in as many words, and the
    conservative direction is the one that keeps the test exact.
    """

    if not differences:
        raise ValueError("The sign test needs at least one seed")
    total = len(differences)
    supporting = sum(1 for value in differences if value > 0)
    return supporting, total, float(
        binomtest(supporting, total, 0.5, alternative="greater").pvalue)


@dataclass(frozen=True)
class Endpoint1Verdict:
    """The registered decision, and nothing beyond it."""

    point: float
    ci_low: float
    ci_high: float
    detected: bool
    reason: str


def verdict(point: float, ci_low: float, ci_high: float) -> Endpoint1Verdict:
    """E3 endpoint 1's failure condition, applied literally.

    The registered sentence is: if the point estimate of B minus A is below
    0.042 *and* the interval contains 0, judge it not detected and do not
    rescue it by changing the estimand. Both clauses, joined by and -- so a
    small estimate whose interval clears 0 is still detected, and a large
    estimate with a wide interval is not automatically thrown away. Writing
    it as one condition either way would be a change to the failure
    condition, which deviation 10.4 puts on the do-not-move list.
    """

    contains_zero = ci_low <= 0.0 <= ci_high
    below = point < DETECTABLE_EFFECT
    interval = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
    if below and contains_zero:
        return Endpoint1Verdict(
            point, ci_low, ci_high, False,
            f"not detected: point {point:+.4f} < {DETECTABLE_EFFECT} "
            f"and CI {interval} contains 0")
    if below:
        return Endpoint1Verdict(
            point, ci_low, ci_high, True,
            f"detected: point {point:+.4f} < {DETECTABLE_EFFECT} "
            f"but CI {interval} excludes 0")
    if contains_zero:
        return Endpoint1Verdict(
            point, ci_low, ci_high, True,
            f"detected: point {point:+.4f} >= {DETECTABLE_EFFECT} "
            f"though CI {interval} contains 0")
    return Endpoint1Verdict(
        point, ci_low, ci_high, True,
        f"detected: point {point:+.4f} >= {DETECTABLE_EFFECT} "
        f"and CI {interval} excludes 0")


@dataclass(frozen=True)
class SeedResult:
    """One replicate's endpoint 1 at its final checkpoint, plus its curve."""

    seed: int
    run: Path
    step: int
    point: float
    ci_low: float
    ci_high: float
    mcnemar_b_only: int
    mcnemar_a_only: int
    mcnemar_p: float
    coverage: Coverage
    trajectory: tuple[tuple[int, float], ...] = field(default=())


def between_seed_spread(results: list[SeedResult]) -> tuple[float, float | None]:
    """Deviation 9.3's reported-but-not-a-test pair: SD (df = n-1) and SD/effect.

    Reported, never tested against. The ratio is None when the mean effect is
    zero, which is a real state and not an error -- five differences that
    average to nothing have no scale to compare their spread to.
    """

    values = np.array([result.point for result in results], dtype=float)
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    mean = float(values.mean())
    return spread, (abs(spread / mean) if mean != 0.0 else None)
