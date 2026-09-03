"""The v4 paired training loop: two arms, one schedule, one difference.

Both arms start from the same base weights, walk the same prompts in the same
order, and draw candidates from the same latent seeds. The only thing that
differs is which of those candidates each arm trains on: the naive arm keeps the
one it judged best while being told what it was drawing, the RFO arm keeps the
one a blind external observer judged best. If the arms diverge, this file is the
only place the divergence could have entered.

This is a rewrite, not a restoration. `830cfa7` deleted the v3 loop along with
the geometric stack because its data side -- reference renderer, manifest
builder, contour verifier -- was the geometric stack. What is copied here is the
skeleton that survived contact with a shared 3090 over three campaigns:

  * A round is done when `DONE.json` exists. Resume skips finished rounds and
    *renames* an unfinished one rather than deleting it, because the thing you
    most want after a crash is the round that crashed.
  * The two arms live in separate checkpoints and are swapped in and out of the
    card one at a time. Holding two 1.5B models plus optimiser state does not
    fit, and the alternative -- training arms in separate processes -- loses the
    guarantee that they saw identical candidates.
  * If either arm abstains on a prompt, the prompt is dropped from *both*. An
    arm training on a prompt its partner skipped is no longer a paired design.
  * Understanding replay is interleaved at a fixed ratio. Without it the
    backbone's ability to answer questions about images decays over training,
    which would corrupt the naive arm's selections and the internal-consistency
    curve at exactly the same time -- indistinguishable from the effect we are
    looking for.

What is new is the data side. There is no reference image in v4, so replay draws
on the frozen corpus: images the adjudication ladder marked correct, asked the
questions their spec generates. Restricting replay to correct images is what
makes the spec's intended answer also the true answer; on an incorrect image the
two differ and replay would teach the backbone to misread pictures.
"""

from __future__ import annotations

import json
import random
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from selfsight.rfo.selection import select_candidate
from selfsight.schemas import CandidateRecord, SelectionDecision
from selfsight.training.paired import PromptScheduleEntry, _seed_from_parts
from selfsight.v4.observe import observe_naive, observe_rfo
from selfsight.v4.probe import build_pools, spec_questions
from selfsight.v4.spec import SceneSpec

ARMS = ("naive", "rfo_self")
NAMESPACE = "v4-train"


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayExample:
    """One (image, question, answer) triple the backbone must keep getting right."""

    image_path: str
    question: str
    answer: str
    sample_id: str


@dataclass(frozen=True)
class TrainingCorpus:
    specs: dict[str, SceneSpec]
    replay: tuple[ReplayExample, ...]

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.specs))


def load_training_corpus(runs: Sequence[str]) -> TrainingCorpus:
    """Prompts to train on, and adjudicated-correct images to replay.

    Specs are keyed by `spec_id`, not by the pools' `run:spec_id`: the same 228
    specs appear in every batch under different seeds, and training the same
    prompt twice because it was drawn twice would be an accident, not a design.
    """

    pools = build_pools(tuple(runs))
    specs: dict[str, SceneSpec] = {}
    replay: list[ReplayExample] = []
    seen: set[str] = set()
    for pool in pools:
        specs.setdefault(pool.spec.spec_id, pool.spec)
        for candidate in pool.candidates:
            if not candidate.correct or candidate.image_path in seen:
                continue
            seen.add(candidate.image_path)
            for question in spec_questions(pool.spec):
                replay.append(ReplayExample(
                    image_path=candidate.image_path,
                    question=question.text,
                    answer=question.expected_answer,
                    sample_id=f"{pool.run}:{candidate.candidate_id}:{question.question_id}",
                ))
    replay.sort(key=lambda item: item.sample_id)
    return TrainingCorpus(specs=specs, replay=tuple(replay))


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def build_schedule(
    prompt_ids: Sequence[str],
    *,
    rounds: int,
    prompts_per_round: int,
    candidate_k: int,
    seed: int,
    max_epochs: int = 1,
) -> list[PromptScheduleEntry]:
    """The prompt order both arms follow, and the seeds both arms draw from.

    `training.build_paired_schedule` requires `rounds * prompts_per_round`
    distinct prompts. The v4 bank holds 228 specs -- deliberately, since every
    batch re-draws the same specs under fresh seeds -- so any schedule longer
    than that needs a second pass over the bank, and `max_epochs` is how the
    caller says so out loud rather than by accident.

    Each epoch is reshuffled under its own derived seed, and a prompt never
    appears twice within a round. Reuse across rounds is not free: a prompt the
    model has already trained on measures memorisation as well as capability,
    which is exactly the confound D* is about. Whoever raises this above 1 owes
    the write-up a sentence about it.
    """

    unique = list(dict.fromkeys(prompt_ids))
    required = rounds * prompts_per_round
    if prompts_per_round > len(unique):
        raise ValueError(
            f"A round wants {prompts_per_round} distinct prompts, bank holds {len(unique)}"
        )
    if required > len(unique) * max_epochs:
        raise ValueError(
            f"Schedule needs {required} prompt slots; {len(unique)} prompts x "
            f"max_epochs={max_epochs} supplies {len(unique) * max_epochs}. "
            f"Either author more specs or raise max_epochs deliberately."
        )

    order: list[str] = []
    epoch = 0
    while len(order) < required:
        shuffled = list(unique)
        random.Random(_seed_from_parts(seed, "epoch", epoch)).shuffle(shuffled)
        order.extend(shuffled)
        epoch += 1
    order = order[:required]

    entries = []
    for flat_index, prompt_id in enumerate(order):
        round_index = flat_index // prompts_per_round
        entries.append(PromptScheduleEntry(
            round_index=round_index,
            prompt_id=prompt_id,
            candidate_seeds=tuple(
                _seed_from_parts(seed, round_index, prompt_id, candidate_index)
                for candidate_index in range(candidate_k)
            ),
        ))
    for index in range(rounds):
        in_round = [entry.prompt_id for entry in entries if entry.round_index == index]
        if len(set(in_round)) != len(in_round):
            raise ValueError(f"Round {index} repeats a prompt")
    return entries


def round_entries(entries: Sequence[PromptScheduleEntry], round_index: int) -> list[PromptScheduleEntry]:
    return [entry for entry in entries if entry.round_index == round_index]


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------


def pair_decisions(
    entries: Sequence[PromptScheduleEntry],
    decisions: dict[str, Sequence[SelectionDecision]],
) -> dict[str, list[SelectionDecision]]:
    """Keep only the prompts on which every arm actually chose something.

    An abstention is not a missing row to be filled in later; it is a prompt
    where one arm had no opinion. Training the other arm on it anyway would put
    a difference into the comparison that has nothing to do with the criterion.
    """

    by_arm = {
        arm: {decision.prompt_id: decision
              for decision in arm_decisions
              if decision.selected_candidate_id is not None}
        for arm, arm_decisions in decisions.items()
    }
    if not by_arm:
        raise ValueError("No arms to pair")
    keep = set.intersection(*(set(mapping) for mapping in by_arm.values()))
    order = [entry.prompt_id for entry in entries if entry.prompt_id in keep]
    return {arm: [by_arm[arm][prompt_id] for prompt_id in order] for arm in by_arm}


# --------------------------------------------------------------------------
# rounds on disk
# --------------------------------------------------------------------------


def completed_rounds(run_root: str | Path) -> list[int]:
    output = []
    for path in (Path(run_root) / "rounds").glob("round-*"):
        if (path / "DONE.json").is_file():
            try:
                output.append(int(path.name.split("-")[-1]))
            except ValueError:
                continue
    return sorted(output)


def abandon_incomplete(path: str | Path) -> Path | None:
    """Move an unfinished round aside. Never delete it -- it is why the run died."""

    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.abandoned-{stamp}")
    os.replace(path, destination)
    return destination


def write_done(round_dir: str | Path, payload: dict[str, Any]) -> Path:
    """The sentinel, written last and never before the artefacts it vouches for."""

    path = Path(round_dir) / "DONE.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# one arm's round
# --------------------------------------------------------------------------


def generate_and_select(
    *,
    arm: str,
    backbone: Any,
    observer: Any,
    corpus: TrainingCorpus,
    entries: Sequence[PromptScheduleEntry],
    output_dir: str | Path,
    checkpoint_id: str,
) -> tuple[list[CandidateRecord], list[SelectionDecision]]:
    """Draw this round's candidates and pick one per prompt, this arm's way.

    `observer` is used only by the RFO arm and must be `None` for the naive arm:
    an observer sitting unused in the naive path is one refactor away from being
    called there.
    """

    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if (arm == "rfo_self") != (observer is not None):
        raise ValueError("The RFO arm needs a frozen observer; the naive arm must not have one")

    output_dir = Path(output_dir)
    candidates_all: list[CandidateRecord] = []
    decisions: list[SelectionDecision] = []
    for entry in entries:
        spec = corpus.specs[entry.prompt_id]
        questions = spec_questions(spec)
        drawn = backbone.generate_images(
            [spec.prompt] * len(entry.candidate_seeds),
            entry.candidate_seeds,
            output_dir,
            checkpoint_id,
            skip_existing=True,
        )
        candidates = [replace(candidate, prompt_id=entry.prompt_id, scene_id=spec.spec_id)
                      for candidate in drawn]
        observations = {}
        for candidate in candidates:
            if arm == "naive":
                observations[candidate.candidate_id] = observe_naive(
                    backbone,
                    prompt=spec.prompt,
                    questions=questions,
                    image_path=candidate.image_path,
                )
            else:
                observations[candidate.candidate_id] = observe_rfo(
                    observer,
                    namespace=NAMESPACE,
                    prompt_id=entry.prompt_id,
                    candidate_id=candidate.candidate_id,
                    questions=questions,
                    image_path=candidate.image_path,
                )
        first = next(iter(observations.values()))
        decisions.append(select_candidate(
            prompt_id=entry.prompt_id,
            arm=arm,
            candidates=candidates,
            observations=observations,
            questions=questions,
            selector_id=first.observer_id,
            observer_revision=first.observer_revision,
        ))
        candidates_all.extend(candidates)
    return candidates_all, decisions


def replay_indices(*, count: int, cursor: int, total: int) -> list[int]:
    if total <= 0:
        raise ValueError("Understanding replay is enabled but the replay corpus is empty")
    return [(cursor + offset) % total for offset in range(count)]


def train_arm(
    *,
    arm: str,
    backbone: Any,
    optimizer: Any,
    scheduler: Any,
    decisions: Sequence[SelectionDecision],
    candidates: Sequence[CandidateRecord],
    corpus: TrainingCorpus,
    training: dict[str, Any],
    seed: int,
    round_index: int,
) -> dict[str, Any]:
    """One arm, one round: generation loss on its picks, replay woven through."""

    import torch

    from selfsight.backbones.showo2 import Showo2GenerationBatch, Showo2ReplayBatch

    if not decisions:
        raise RuntimeError(f"Arm {arm} has nothing to train on in round {round_index}")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [by_id[str(decision.selected_candidate_id)] for decision in decisions]

    micro_size = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    optimizer_steps = int(training["optimizer_steps_per_round"])
    ratio = Fraction(str(float(training["understanding_replay_ratio"]))).limit_denominator(100)

    t2i_cursor = 0
    replay_cursor = 0
    t2i_losses: list[float] = []
    replay_losses: list[float] = []
    gradient_norms: list[float] = []
    trainable = [parameter for parameter in backbone.model.parameters() if parameter.requires_grad]

    for optimizer_step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accumulation):
            global_micro = optimizer_step * accumulation + micro_step
            use_replay = ratio.numerator > 0 and global_micro % ratio.denominator < ratio.numerator
            batch_seed = _seed_from_parts(seed, round_index, optimizer_step, micro_step,
                                          "replay" if use_replay else "t2i")
            if use_replay:
                picks = [corpus.replay[index] for index in replay_indices(
                    count=micro_size, cursor=replay_cursor, total=len(corpus.replay))]
                replay_cursor += micro_size
                loss = backbone.understanding_replay_loss(Showo2ReplayBatch(
                    images=tuple(item.image_path for item in picks),
                    questions=tuple(item.question for item in picks),
                    answers=tuple(item.answer for item in picks),
                    sample_ids=tuple(item.sample_id for item in picks),
                    latent_seed=batch_seed,
                ))
                replay_losses.append(float(loss.detach().cpu()))
            else:
                picks = [selected[(t2i_cursor + offset) % len(selected)]
                         for offset in range(micro_size)]
                t2i_cursor += micro_size
                loss = backbone.generation_loss(Showo2GenerationBatch(
                    prompts=tuple(corpus.specs[item.prompt_id].prompt for item in picks),
                    images=tuple(item.image_path for item in picks),
                    sample_ids=tuple(item.candidate_id for item in picks),
                    latent_seed=batch_seed,
                ))
                t2i_losses.append(float(loss.detach().cpu()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {arm} loss at round {round_index}")
            (loss / accumulation).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        gradient_norms.append(float(grad_norm.detach().cpu()))
        optimizer.step()
        scheduler.step()

    return {
        "arm": arm,
        "round": round_index,
        "selected_samples": len(selected),
        "optimizer_steps": optimizer_steps,
        "t2i_microbatches": len(t2i_losses),
        "replay_microbatches": len(replay_losses),
        "mean_t2i_loss": sum(t2i_losses) / len(t2i_losses) if t2i_losses else None,
        "mean_replay_loss": sum(replay_losses) / len(replay_losses) if replay_losses else None,
        "mean_gradient_norm_before_clip": sum(gradient_norms) / len(gradient_norms),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
