"""Deviation 7.2's run-to-run drift bound, and deviation 7.3's round bootstrap.

The main run trains `naive` + `rfo_gold`; each replicate trains `naive` +
`blind_self`. So `naive` gets run more than once, at zero marginal cost, and
the gap between those runs bounds how much of an A-B difference could be
run-to-run drift rather than the arm.

    |theta_A - theta_A'|  <  theta_B - theta_A'   ->  the difference is larger
                                                     than the drift bound

It is an upper bound and not a clean seed replication, because the two `naive`
columns saw different training subsets: the main run's A intersects with
`rfo_gold`'s selections and the replicate's A with `blind_self`'s. Deviation
7.2 says so in the sentence that registers it, and the bound errs in the
conservative direction, which is the direction that makes it usable.

Deviation 14 pins the parts 7.2 left to whoever wrote the script: all three
thetas come off one key set (14.2), the five replicates each get their own
verdict and no aggregate rule is registered (14.3), and the round bootstrap
drops step 0 because both arms are bit-identical there (14.5).

Deviation 7.2's second clause is as binding as its first. When the drift bound
is not smaller than the effect, the paper says the training-period claim is
not separable from run-level drift and retreats to inference time. It does not
re-cut the 口径, report the favourable half, or discover that the two A columns
were never comparable -- that was known when the rule was written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from selfsight.analysis.endpoint1 import (
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    _verdicts,
    completed_steps,
)

# Deviation 14.5: the shared untrained base. Both arms evaluate the same model
# on the same latents there, so their verdicts are bit-identical and the paired
# difference is 0 by construction. Averaging it in would drag the mean toward
# zero with a number that measures nothing.
BASE_STEP = 0

SEPARABLE = "the A-B difference exceeds a drift bound that includes pairing"
NOT_SEPARABLE = ("the training-period claim is not separable from run-level "
                 "drift; the claim retreats to inference time "
                 "(pre-registration, deviation 7.2)")


@dataclass(frozen=True)
class Columns:
    """Three arms' verdicts on one key set, plus what reaching it cost."""

    step: int
    keys: tuple[tuple[str, int], ...]
    theta_a: float
    theta_a_prime: float
    theta_b: float
    marginal: dict[str, float] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def drift(self) -> float:
        """|theta_A - theta_A'|, deviation 7.2's left-hand side."""

        return abs(self.theta_a - self.theta_a_prime)

    @property
    def effect(self) -> float:
        """theta_B - theta_A', the right-hand side. Signed: B below A' is not
        an effect this bound could clear, and reporting |.| here would hide
        that."""

        return self.theta_b - self.theta_a_prime

    @property
    def separable(self) -> bool:
        return self.drift < self.effect

    @property
    def wording(self) -> str:
        return SEPARABLE if self.separable else NOT_SEPARABLE


# `split.json` records when the split stage ran. That stamp is not part of
# what the split is: `v4_train.split_digest` is a function of (runs, seed,
# outcome, probe) and says "and nothing else". Two runs that hold out the
# same prompts under the same config write `created` at two different times
# by construction -- the main run's says 2026-09-08T14:33:14Z -- so hashing
# it makes the guard below fire on every legitimate pair.
IDENTITY_EXCLUDES = ("created",)


def split_digest(run: Path) -> str:
    """The evaluation split's identity, which is what makes two runs comparable.

    Deviation 7.2 leans on the two runs having the same split -- without it
    `theta_A` and `theta_A'` are rates over different questions and the
    subtraction means nothing. It says a digest check will confirm it, so this
    is that check rather than a comment saying it holds.

    Everything the file records except the timestamp goes into the hash, which
    is stricter than reading the `digest` field it already carries: that field
    is a hash of the recipe, and this is a hash of the recipe together with the
    prompt lists the recipe produced. They can only come apart if something
    edited the file, which is exactly the case a guard is for.
    """

    path = Path(run) / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"No split.json in {run}; cannot confirm the two runs "
                                f"measured the same prompts (deviation 7.2)")
    import hashlib

    identity = {key: value
                for key, value in json.loads(path.read_text(encoding="utf-8")).items()
                if key not in IDENTITY_EXCLUDES}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def final_common_step(columns: list[tuple[Path, str]]) -> int:
    """Deviation 14.4: the smallest of the three columns' largest steps.

    A gapped ladder -- that step adjudicated in one column and not another --
    is not something 14.4 anticipated, so it raises instead of silently
    sliding down to a step all three happen to share.
    """

    ladders = {}
    for run, arm in columns:
        steps = completed_steps(run, arm)
        if not steps:
            raise ValueError(f"{run.name} {arm}: no adjudicated checkpoint")
        ladders[(run.name, arm)] = steps
    step = min(max(steps) for steps in ladders.values())
    missing = [name for name, steps in ladders.items() if step not in steps]
    if missing:
        raise ValueError(f"step {step} is the smallest final step but is not adjudicated "
                         f"in {missing}; deviation 14.4 has no rule for a gapped ladder")
    return step


def _column(run: Path, arm: str, step: int) -> dict[tuple[str, int], bool]:
    path = Path(run) / "evaluations" / arm / f"step-{step:05d}" / "verified.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No verdicts for {run.name} {arm} at step {step}")
    return {key: value for key, value in _verdicts(path).items() if value is not None}


def load_columns(main: Path, replicate: Path, *, step: int | None = None,
                 arm_a: str = "naive", arm_b: str = "blind_self") -> Columns:
    """Deviation 14.1 and 14.2: one key set, three rates on it.

    The intersection is taken over keys adjudicated in all three columns, so
    the two quantities 7.2 compares share a denominator. Each column's rate on
    its own keys is reported beside it, and so is what the intersection cost,
    because an intersection that throws away a lot is itself a fact about the
    run rather than a detail of the arithmetic.
    """

    plan = [(main, arm_a), (replicate, arm_a), (replicate, arm_b)]
    if step is None:
        step = final_common_step(plan)
    if split_digest(main) != split_digest(replicate):
        raise ValueError(f"{main.name} and {replicate.name} have different splits; "
                         f"their naive columns are rates over different prompts")
    columns = {"a": _column(main, arm_a, step),
               "a_prime": _column(replicate, arm_a, step),
               "b": _column(replicate, arm_b, step)}
    shared = sorted(set.intersection(*(set(column) for column in columns.values())))
    if not shared:
        raise ValueError(f"step {step}: no key is adjudicated in all three columns")
    rate = {name: prompt_mean(column, shared) for name, column in columns.items()}
    return Columns(
        step=step, keys=tuple(shared),
        theta_a=rate["a"], theta_a_prime=rate["a_prime"], theta_b=rate["b"],
        marginal={name: (prompt_mean(column, sorted(column)) if column else float("nan"))
                  for name, column in columns.items()},
        dropped={name: len(column) - len(shared) for name, column in columns.items()})


def prompt_mean(column: dict[tuple[str, int], bool],
                keys: list[tuple[str, int]]) -> float:
    """Deviation 14.1: within a prompt first, then across prompts.

    Endpoint 1 aggregates this way (deviation 10.2 point 2) and 7.2 says to
    use endpoint 1's caliber. The flat mean over keys agrees with it while
    every prompt keeps all four draws, and parts from it as soon as deletion
    makes the counts unequal -- at which point the flat form reweights prompts
    by how many of their draws survived adjudication.
    """

    by_prompt: dict[str, list[bool]] = {}
    for spec_id, _ in keys:
        by_prompt.setdefault(spec_id, [])
    for key in keys:
        by_prompt[key[0]].append(column[key])
    return float(np.mean([np.mean(values) for values in by_prompt.values()]))


@dataclass(frozen=True)
class RoundBootstrap:
    """Deviation 7.3's round axis: is the effect steady from round to round?"""

    steps: tuple[int, ...]
    differences: tuple[float, ...]
    mean: float
    interval: tuple[float, float]
    resamples: int
    seed: int

    @property
    def spans_zero(self) -> bool:
        return self.interval[0] <= 0.0 <= self.interval[1]


def round_bootstrap(differences: dict[int, float], *, resamples: int = FINAL_RESAMPLES,
                    seed: int = FINAL_BOOTSTRAP_SEED) -> RoundBootstrap:
    """Resample rounds, not prompts. Deviation 14.5.

    Descriptive on purpose: deviation 7.3 registers this as something the paper
    reports and attaches no failure condition to it, so an interval spanning
    zero is written down as an interval spanning zero. It is neither a negative
    result nor a rescue.
    """

    rounds = {step: value for step, value in differences.items() if step != BASE_STEP}
    if not rounds:
        raise ValueError("No round to resample; step 0 is excluded by deviation 14.5")
    steps = tuple(sorted(rounds))
    values = np.array([rounds[step] for step in steps], dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = (float(value) for value in np.percentile(draws, [2.5, 97.5]))
    return RoundBootstrap(steps=steps, differences=tuple(values.tolist()),
                          mean=float(values.mean()), interval=(low, high),
                          resamples=resamples, seed=seed)
