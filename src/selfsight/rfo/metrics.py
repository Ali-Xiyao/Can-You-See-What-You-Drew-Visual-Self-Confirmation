"""E1 context and pixel-override metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextTrial:
    trial_id: str
    intended_answer: str
    pixel_answer: str
    rgb_only_answer: str | None
    prompt_only_answer: str | None
    rgb_prompt_answer: str | None
    hard_render_answer: str | None
    counterfactual_answer: str | None


def _mean(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def compute_e1_metrics(trials: Iterable[ContextTrial]) -> dict[str, float]:
    trials = list(trials)
    conflicting = [trial for trial in trials if trial.intended_answer != trial.pixel_answer]
    return {
        "n": float(len(trials)),
        "n_conflicting": float(len(conflicting)),
        "scfr": _mean(trial.rgb_prompt_answer == trial.intended_answer for trial in conflicting),
        "poe": _mean(
            trial.rgb_prompt_answer == trial.intended_answer and trial.rgb_only_answer == trial.pixel_answer
            for trial in conflicting
        ),
        "observation_gain": _mean(
            trial.rgb_only_answer == trial.pixel_answer and trial.prompt_only_answer != trial.pixel_answer
            for trial in trials
        ),
        "hard_render_consistency": _mean(
            trial.rgb_only_answer == trial.hard_render_answer for trial in trials
        ),
        "pixel_sensitive_feedback_preference": _mean(
            trial.counterfactual_answer == trial.pixel_answer for trial in conflicting
        ),
    }
