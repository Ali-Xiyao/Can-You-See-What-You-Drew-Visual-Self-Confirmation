"""Two curves per checkpoint, and the split that keeps them honest.

D* is defined as the earliest training step where internal cycle consistency is
still rising while external verifiable correctness has stopped. Measuring it
needs both quantities at every checkpoint, and needs them to be comparable, so
three choices here matter more than the code does:

**Both curves come off the same images.** The config inherits separate `probe`
and `outcome` sets, and it would be natural to score internal on one and
external on the other. It would also be wrong: a gap opening between two curves
measured on two prompt sets is a gap between two prompt sets until proven
otherwise, and D* is exactly a claim about a gap. The outcome set carries both
curves. The probe set is spent on the gradient probe, which is what it was
named for -- that feeds D_g, a different estimate on a different quantity.

**External correctness is measured by the corpus's own instrument.** This module
writes a manifest and stops. `scripts/v4_run_pipeline.py detect` and `verify`
then run over it unchanged -- same two detectors, same three-level adjudication
ladder, same `image_correct`. A second, cheaper verifier written for the
training loop would make the external curve incomparable to every number in
sections 11 through 27, and the first thing anyone would ask of a D* is how it
sits against those.

**The two internal measurements remain separate.** The legacy caption recovery
likelihood is retained as `internal_cycle`. The dynamic pilot additionally
records `s_select`: the prompted atomic score used to select training images.
Both use the outcome images, and every requested atomic answer remains in the
score denominator, including missing answers, errors and abstentions.

Evaluation prompts are held out from training by `split_prompts`. If a prompt
the model trained on appeared in the outcome set, the external curve would be
measuring memorisation, and it would be *rising* where the real one falls.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from selfsight.analysis.breakpoints import (
    DivergenceEstimate,
    GradientWarningEstimate,
    estimate_d_g,
    estimate_d_star,
    estimate_lead,
)
from selfsight.training.paired import _seed_from_parts
from selfsight.schemas import AtomicQuestion, ObservationResult
from selfsight.utils.hashing import rgb_sha256, sha256_json
from selfsight.v4.observe import observe_naive
from selfsight.v4.probe import spec_questions
from selfsight.v4.spec import SceneSpec

METRIC_FIELDS = (
    "arm",
    "round_index",
    "step",
    "internal_cycle",
    "internal_sem",
    "internal_n",
    "external_correct",
    "external_n",
    "external_unadjudicated",
    "s_select",
    "s_select_sem",
    "s_select_n",
    "s_select_available",
    "s_select_total",
)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSplit:
    train: tuple[str, ...]
    outcome: tuple[str, ...]
    probe: tuple[str, ...]

    def assert_disjoint(self) -> None:
        parts = {"train": set(self.train), "outcome": set(self.outcome), "probe": set(self.probe)}
        for left in parts:
            for right in parts:
                if left >= right:
                    continue
                overlap = sorted(parts[left].intersection(parts[right]))
                if overlap:
                    raise ValueError(
                        f"{left} and {right} splits share {len(overlap)} prompts: {overlap[:5]}"
                    )


def split_prompts(
    prompt_ids: Sequence[str],
    *,
    outcome: int,
    probe: int,
    seed: int,
) -> PromptSplit:
    """Carve the bank into training, outcome and gradient-probe sets.

    Held out first, trained on second. The order matters only in that it makes
    the sizes of the evaluation sets a decision rather than a remainder: with
    228 prompts, an outcome set that quietly shrinks because training took what
    it wanted is how an external curve ends up too noisy to fit a breakpoint to.
    """

    import random

    unique = sorted(dict.fromkeys(prompt_ids))
    if outcome + probe >= len(unique):
        raise ValueError(
            f"Evaluation wants {outcome + probe} of {len(unique)} prompts, leaving "
            f"{len(unique) - outcome - probe} to train on"
        )
    shuffled = list(unique)
    random.Random(_seed_from_parts(seed, "split")).shuffle(shuffled)
    held_outcome = tuple(sorted(shuffled[:outcome]))
    held_probe = tuple(sorted(shuffled[outcome:outcome + probe]))
    train = tuple(sorted(shuffled[outcome + probe:]))
    result = PromptSplit(train=train, outcome=held_outcome, probe=held_probe)
    result.assert_disjoint()
    return result


# --------------------------------------------------------------------------
# the external side: hand the corpus instrument something to chew on
# --------------------------------------------------------------------------


def evaluation_seed(*, seed: int, arm: str, step: int, prompt_id: str) -> int:
    """One latent per (checkpoint, prompt), shared by both arms.

    Both arms drawing the outcome set from the same latents means a difference
    between their external curves is a difference between the models, not
    between two draws from the same model.
    """

    return _seed_from_parts(seed, "eval", step, prompt_id)


def write_evaluation_manifest(
    directory: str | Path,
    *,
    specs: dict[str, SceneSpec],
    prompt_ids: Sequence[str],
    images: dict[str, str],
    seeds: dict[str, int],
) -> Path:
    """A manifest in the shape `detect` and `verify` already read.

    `candidate_index` is 0 for every row: evaluation draws one image per prompt,
    where the corpus drew K. The verifier does not care, and reusing its schema
    exactly is the point -- the moment this file invents its own, the external
    curve stops being the same measurement as section 27's.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.jsonl"
    rows = []
    for prompt_id in prompt_ids:
        if prompt_id not in images:
            raise KeyError(f"No generated image for evaluation prompt {prompt_id}")
        spec = specs[prompt_id]
        rows.append({
            "spec_id": spec.spec_id,
            "candidate_index": 0,
            "seed": int(seeds[prompt_id]),
            "prompt": spec.prompt,
            "image_path": images[prompt_id],
            "spec": spec.to_dict(),
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


UNADJUDICATED = {"pending_human", "unnameable"}


def external_correctness(verified_path: str | Path) -> tuple[float | None, int, int]:
    """Correct rate over adjudicated images, plus how many were not adjudicated.

    Unadjudicated images are excluded from the rate rather than counted wrong.
    Counting them wrong would let the external curve fall simply because later
    checkpoints draw messier pictures that the two detectors argue about more --
    a real phenomenon, but not the one D* is about, and it would bias the
    breakpoint early in precisely the direction the hypothesis predicts.
    """

    path = Path(verified_path)
    if not path.exists():
        return None, 0, 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    adjudicated = [row for row in rows if row["resolution"] not in UNADJUDICATED]
    unadjudicated = len(rows) - len(adjudicated)
    if not adjudicated:
        return None, 0, unadjudicated
    rate = sum(bool(row["image_correct"]) for row in adjudicated) / len(adjudicated)
    return rate, len(adjudicated), unadjudicated


# --------------------------------------------------------------------------
# the internal side
# --------------------------------------------------------------------------


def fixed_atomic_score(
    observation: ObservationResult, questions: Sequence[AtomicQuestion],
) -> dict[str, float | int]:
    """All requested questions count; unavailable answers never raise the score."""
    wanted = {question.question_id: question for question in questions}
    if not wanted or len(wanted) != len(questions):
        raise ValueError("Atomic score requires nonempty, unique question IDs")
    answers = {}
    for answer in observation.answers:
        if answer.question_id not in wanted:
            raise ValueError(f"Unexpected answer: {answer.question_id}")
        if answer.question_id in answers:
            raise ValueError(f"Duplicate answer: {answer.question_id}")
        answers[answer.question_id] = answer
    available = correct = errors = abstained = 0
    for question_id, question in wanted.items():
        answer = answers.get(question_id)
        if answer is None:
            continue
        if answer.error is not None:
            errors += 1
        elif answer.abstain or answer.normalized_answer is None:
            abstained += 1
        else:
            available += 1
            correct += answer.normalized_answer == question.expected_answer
    return {"s_select": correct / len(wanted), "correct": correct,
            "available": available, "total": len(wanted),
            "errors": errors, "abstained": abstained,
            "missing": len(wanted) - len(answers)}


def self_selection_scores(
    backbone: Any, *, specs: dict[str, SceneSpec], images: dict[str, str],
    output_path: str | Path, metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Durable per-image prompted scores, with image/question/run resume guards."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row["prompt_id"]
            if prompt_id in existing or prompt_id not in images:
                raise ValueError(f"Duplicate or unexpected evaluation prompt: {prompt_id}")
            if row["metadata"] != metadata:
                raise ValueError("Self-selection evaluation belongs to a different checkpoint/config")
            existing[prompt_id] = row
    rows = []
    with path.open("a", encoding="utf-8") as handle:
        for prompt_id, image_path in sorted(images.items()):
            spec = specs[prompt_id]
            questions = spec_questions(spec)
            question_rows = [asdict(question) for question in questions]
            digest = sha256_json({"prompt": spec.prompt, "questions": question_rows})
            image_digest = rgb_sha256(image_path)
            if prompt_id in existing:
                row = existing[prompt_id]
                if row["question_digest"] != digest or row["rgb_sha256"] != image_digest:
                    raise ValueError(f"Evaluation image/questions changed for {prompt_id}")
                observation = ObservationResult.from_dict(row["observation"])
                rescored = fixed_atomic_score(observation, questions)
                if (observation.rgb_sha256 != image_digest or
                        any(row[key] != value for key, value in rescored.items())):
                    raise ValueError(f"Saved evaluation observation/score mismatch for {prompt_id}")
            else:
                observation = observe_naive(backbone, prompt=spec.prompt,
                                            questions=questions, image_path=image_path)
                if observation.rgb_sha256 != image_digest:
                    raise ValueError(f"Observer returned a different image hash for {prompt_id}")
                row = {"prompt_id": prompt_id, "image_path": image_path,
                       "rgb_sha256": image_digest, "question_digest": digest,
                       "metadata": metadata, "questions": question_rows,
                       "observation": observation.to_dict(),
                       **fixed_atomic_score(observation, questions)}
                encoded = json.dumps(row)
                row = json.loads(encoded)
                handle.write(encoded + "\n")
                handle.flush()
            rows.append(row)
    return rows


def summarize_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean, sem, n = summarize_cycle({str(index): float(row["s_select"])
                                   for index, row in enumerate(rows)})
    total = sum(int(row["total"]) for row in rows)
    available = sum(int(row["available"]) for row in rows)
    return {"mean": mean, "sem": sem, "n": n, "available": available,
            "total": total, "coverage": available / total if total else None,
            **{key: sum(int(row[key]) for row in rows)
               for key in ("correct", "errors", "abstained", "missing")}}


def cycle_scores(
    backbone: Any,
    *,
    specs: dict[str, SceneSpec],
    images: dict[str, str],
) -> dict[str, float]:
    """`log p(prompt | image)` per prompt, under the neutral captioning instruction."""

    return {
        prompt_id: backbone.cycle_consistency_score(image_path, specs[prompt_id].prompt)
        for prompt_id, image_path in sorted(images.items())
    }


def summarize_cycle(scores: Mapping[str, float]) -> tuple[float | None, float | None, int]:
    """Mean, standard error, and count.

    The standard error is not decoration. `estimate_d_star` decides that internal
    consistency is "still rising" by comparing a fitted slope against a
    threshold, and its default threshold is zero -- which a curve of pure
    floating-point noise clears (measured: a perfectly flat internal curve
    yields slope 2e-18 and a confident D* of 75). The only defensible threshold
    is one built from how precisely the curve was measured, so the precision has
    to travel with the mean.
    """

    values = [float(value) for value in scores.values()]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None, 1
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values)), len(values)


# --------------------------------------------------------------------------
# rows, curves, and the estimate they exist for
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointMetrics:
    arm: str
    round_index: int
    step: int
    internal_cycle: float | None
    internal_sem: float | None
    internal_n: int
    external_correct: float | None
    external_n: int
    external_unadjudicated: int
    s_select: float | None = None
    s_select_sem: float | None = None
    s_select_n: int = 0
    s_select_available: int = 0
    s_select_total: int = 0


def write_metrics_csv(path: str | Path, rows: Sequence[CheckpointMetrics]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.arm, item.step)):
            writer.writerow(asdict(row))
    return path


def read_metrics_csv(path: str | Path) -> list[CheckpointMetrics]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(CheckpointMetrics(
                arm=raw["arm"],
                round_index=int(raw["round_index"]),
                step=int(raw["step"]),
                internal_cycle=_optional_float(raw["internal_cycle"]),
                internal_sem=_optional_float(raw["internal_sem"]),
                internal_n=int(raw["internal_n"]),
                external_correct=_optional_float(raw["external_correct"]),
                external_n=int(raw["external_n"]),
                external_unadjudicated=int(raw["external_unadjudicated"]),
                s_select=_optional_float(raw.get("s_select", "")),
                s_select_sem=_optional_float(raw.get("s_select_sem", "")),
                s_select_n=int(raw.get("s_select_n", 0)),
                s_select_available=int(raw.get("s_select_available", 0)),
                s_select_total=int(raw.get("s_select_total", 0)),
            ))
    return rows


def _optional_float(value: str) -> float | None:
    return None if value in ("", "None") else float(value)


def curves(rows: Sequence[CheckpointMetrics], arm: str) -> tuple[list[float], list[float], list[float]]:
    """Steps, internal and external for one arm, with incomplete checkpoints dropped.

    A checkpoint missing either curve is dropped from both rather than carried
    with a gap. `fit_segmented` takes two aligned sequences; letting them fall
    out of alignment would silently pair a step's internal score with a
    different step's external one.
    """

    usable = sorted(
        (row for row in rows
         if row.arm == arm
         and row.internal_cycle is not None
         and row.external_correct is not None),
        key=lambda row: row.step,
    )
    return (
        [float(row.step) for row in usable],
        [float(row.internal_cycle) for row in usable],
        [float(row.external_correct) for row in usable],
    )


@dataclass(frozen=True)
class DivergenceReport:
    arm: str
    checkpoints: int
    min_internal_slope: float
    divergence: DivergenceEstimate
    warning: GradientWarningEstimate | None
    lead: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "checkpoints": self.checkpoints,
            "min_internal_slope": self.min_internal_slope,
            "d_star": self.divergence.d_star,
            "internal_post_slope": self.divergence.internal_post_slope,
            "external_post_slope": self.divergence.external_post_slope,
            "reason": self.divergence.reason,
            "d_g": None if self.warning is None else self.warning.d_g,
            "d_g_reason": None if self.warning is None else self.warning.reason,
            "early_safe": None if self.warning is None else self.warning.early_safe,
            "lead": self.lead,
        }


def internal_noise_slope(rows: Sequence[CheckpointMetrics], arm: str) -> float:
    """The steepest internal slope that measurement noise alone could produce.

    `estimate_d_star` accepts any positive post-breakpoint internal slope as
    evidence that internal consistency is still rising. Its default floor is
    zero, and zero is not a floor: a perfectly flat curve fits at slope 2e-18
    and yields a confident D*. Since D* is *defined* by internal still rising
    while external has stopped, that default turns a null result into a finding.

    The floor used here is the slope of a line that climbs one standard error
    across the entire run. A curve flatter than that has not been shown to rise
    at all, whatever the least-squares fit says.
    """

    usable = sorted((row for row in rows if row.arm == arm), key=lambda row: row.step)
    sems = [row.internal_sem for row in usable if row.internal_sem is not None]
    if len(usable) < 2 or not sems:
        raise ValueError(
            f"Cannot derive a noise floor for arm {arm}: need at least two checkpoints "
            f"carrying internal_sem, found {len(sems)}"
        )
    span = float(usable[-1].step - usable[0].step)
    if span <= 0:
        raise ValueError(f"Arm {arm} has no step span to measure a slope against")
    return float(sorted(sems)[len(sems) // 2]) / span


def divergence_report(
    rows: Sequence[CheckpointMetrics],
    arm: str,
    *,
    gda_free: Sequence[float] | None = None,
    noise_low: float | Sequence[float] | None = None,
    noise_high: float | Sequence[float] | None = None,
    min_internal_slope: float | None = None,
) -> DivergenceReport:
    """D*, and D_g when a gradient curve is supplied alongside its noise floor.

    `Lead = D* - D_g` is the endpoint, and it is only defined when both exist.
    Reporting a lead against a missing D_g -- or against a D_g whose noise floor
    was not measured on the same run -- would turn "we could not estimate this"
    into a number.

    `min_internal_slope` defaults to `internal_noise_slope`. Passing it
    explicitly is for sensitivity analysis; passing zero reproduces the
    estimator's own default and is not a defensible headline number.
    """

    steps, internal, external = curves(rows, arm)
    floor = internal_noise_slope(rows, arm) if min_internal_slope is None else min_internal_slope
    divergence = estimate_d_star(steps, internal, external,
                                 min_positive_internal_slope=floor)
    warning = None
    if gda_free is not None:
        if noise_low is None or noise_high is None:
            raise ValueError("A gradient curve needs its measured noise floor to yield D_g")
        if len(gda_free) != len(steps):
            raise ValueError(
                f"Gradient curve has {len(gda_free)} points, {len(steps)} usable checkpoints"
            )
        warning = estimate_d_g(steps, gda_free, noise_low=noise_low, noise_high=noise_high)
    return DivergenceReport(
        arm=arm,
        checkpoints=len(steps),
        min_internal_slope=floor,
        divergence=divergence,
        warning=warning,
        lead=estimate_lead(divergence.d_star, None if warning is None else warning.d_g),
    )
