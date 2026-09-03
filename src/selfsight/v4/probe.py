"""The Gate B probe set: balanced pools and the cycle-consistency atoms.

Gate B asks whether the gradient instrument is sharp enough to carry claim one.
Two of its criteria have never been measured -- the paired bootstrap CI on
`cos(g_naive, g_rfo)`, and whether that cosine degenerates to zero once the
gradients are projected into the LoRA subspace. Both need one LoRA gradient per
prompt per selection criterion, and that needs a fixed probe set of pools where
the criteria can actually disagree.

Three things live here because they are decisions, not plumbing, and each one
has a way of being got quietly wrong:

`spec_questions` builds the questions **from the spec alone**. Every candidate
in a pool must be asked the same thing or the scores are not comparable, and the
v4 corpus questions are built per image from its own detections, so they cannot
be reused here. The expected answer is what the prompt *asked for*, which is
what makes the score a cycle-consistency score: a candidate scores high when the
observer confirms the picture matches the request. It borrows the corpus's
`_place` so the correct option is not always A -- see its docstring for what a
fixed position would do to the Gate B number.

`build_pools` keeps only pools that are balanced under the adjudicated verdict
-- at least one correct and at least one incorrect candidate. A pool where every
candidate is correct cannot separate any two criteria, so including it would
buy a narrower CI without buying any information, which is precisely the way to
pass Gate B while measuring nothing. Pools containing an image that never
reached a verdict are dropped whole, never partly: dropping the one unadjudicated
candidate would change the pool the criteria choose from.

`gold_selection` breaks ties by seed then id, the same rule every other selector
in this project uses, so that "the criteria agreed" never means "they happened
to break a tie the same way".
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selfsight.schemas import AtomicQuestion, QuestionFamily, QuestionFormat
from selfsight.v4.questions import _place
from selfsight.v4.spec import SceneSpec, canonical_noun

UNADJUDICATED = {"pending_human", "unnameable"}
NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass(frozen=True)
class PoolCandidate:
    candidate_id: str
    image_path: str
    sampling_seed: int
    correct: bool


@dataclass(frozen=True)
class Pool:
    prompt_id: str
    run: str
    spec: SceneSpec
    candidates: tuple[PoolCandidate, ...]

    @property
    def balanced(self) -> bool:
        verdicts = {candidate.correct for candidate in self.candidates}
        return len(verdicts) == 2


def _phrase(noun: str, colour: str | None) -> str:
    return f"{colour} {noun}" if colour else noun


def spec_questions(spec: SceneSpec) -> tuple[AtomicQuestion, ...]:
    """The intent, as forced-choice atoms the observer can be asked about.

    One existence atom per requested object, plus a count atom for objects
    requested more than once. The expected answer is always what the *request*
    asked for, which is what makes the resulting score a cycle-consistency
    score: a candidate scores high when the observer confirms the picture
    matches what was asked.

    Two details are load-bearing.

    The correct option is placed in A or B by the same `_place` the v4 corpus
    uses, seeded from the spec so the placement is identical for every candidate
    in a pool and stable across reruns. Left in a fixed position, a model with a
    letter preference would score above chance without looking at the picture,
    every candidate would score alike, all three criteria would fall through to
    the same tie-break, and the Gate B cosine would come out at 1.000 while
    measuring nothing at all.

    `family` is EXISTENCE on every atom, including the counting ones, for the
    reason `v4.questions.to_atomic` gives: the family only chooses the fallback
    vocabulary `normalize_answer` uses when no choice letter is found, and the
    counting fallback rewrites "two" to "2", which would never match an expected
    answer of "two". With EXISTENCE, a reply that names no letter abstains --
    which is the right reading of an unparseable answer to a forced choice.
    """

    rng = random.Random(f"v4-gate-b:{spec.spec_id}")
    questions: list[AtomicQuestion] = []
    for index, obj in enumerate(spec.objects):
        noun = canonical_noun(obj.object)
        phrase = _phrase(noun, obj.color)
        option_a, option_b, gold = _place(rng, "yes", "no")
        questions.append(AtomicQuestion(
            question_id=f"{spec.spec_id}:exists:{index}",
            atom_id=f"{spec.spec_id}:exists:{noun}",
            family=QuestionFamily.EXISTENCE,
            text=(f"Is there a {phrase} in this picture? Answer A or B only.\n"
                  f"A. {option_a}\nB. {option_b}"),
            expected_answer="yes",
            question_format=QuestionFormat.FORCED_CHOICE,
            choices=(option_a, option_b),
            choice_order_seed=ord(gold),
        ))
        if obj.count > 1:
            wanted = NUMBER_WORDS[obj.count]
            option_a, option_b, gold = _place(rng, wanted, NUMBER_WORDS[obj.count - 1])
            questions.append(AtomicQuestion(
                question_id=f"{spec.spec_id}:count:{index}",
                atom_id=f"{spec.spec_id}:count:{noun}",
                family=QuestionFamily.EXISTENCE,
                text=(f"How many {noun}s are in this picture? Answer A or B "
                      f"only.\nA. {option_a}\nB. {option_b}"),
                expected_answer=wanted,
                question_format=QuestionFormat.FORCED_CHOICE,
                choices=(option_a, option_b),
                choice_order_seed=ord(gold),
            ))
    return tuple(questions)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pools(runs: tuple[str, ...]) -> list[Pool]:
    """One pool per spec per run, dropped whole if any candidate lacks a verdict."""
    pools: list[Pool] = []
    for name in runs:
        run = Path(name)
        specs: dict[str, SceneSpec] = {}
        by_spec: dict[str, list[dict[str, Any]]] = {}
        for row in _read_jsonl(run / "manifest.jsonl"):
            spec = SceneSpec.from_dict(row["spec"])
            specs[spec.spec_id] = spec
            by_spec.setdefault(spec.spec_id, []).append(row)
        verdicts = {row["image_path"]: row
                    for row in _read_jsonl(run / "verified.jsonl")}
        for spec_id, rows in sorted(by_spec.items()):
            candidates: list[PoolCandidate] = []
            usable = True
            for row in sorted(rows, key=lambda r: r["candidate_index"]):
                verdict = verdicts.get(row["image_path"])
                if verdict is None or verdict["resolution"] in UNADJUDICATED:
                    usable = False
                    break
                candidates.append(PoolCandidate(
                    candidate_id=f"{spec_id}:{row['candidate_index']}",
                    image_path=row["image_path"],
                    sampling_seed=int(row["seed"]),
                    correct=bool(verdict["image_correct"]),
                ))
            if usable and candidates:
                pools.append(Pool(prompt_id=f"{run.name}:{spec_id}", run=run.name,
                                  spec=specs[spec_id],
                                  candidates=tuple(candidates)))
    return pools


def gold_selection(pool: Pool) -> str:
    """The verifier's pick: a correct candidate, ties by seed then id."""
    return max(pool.candidates,
               key=lambda c: (c.correct, -c.sampling_seed, c.candidate_id)).candidate_id
