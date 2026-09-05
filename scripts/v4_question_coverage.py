"""What the new question construction changes, on the pools it will be used on.

This is the CPU half of the selector-resolution prereg (section 6): the coverage
audit and the arithmetic of the score, neither of which needs the model. What it
cannot tell you is whether resolution actually improves -- that needs the model
to answer the new questions, and it is deliberately not guessed at here.

`runs/v4/gate-b/pools.jsonl` stores the questions each pool was actually asked,
built by the old code. So "before" is read off disk rather than reimplemented,
and "after" is `spec_questions` as it stands now.

The one thing worth being careful about: a candidate's score is
`correct / answered`, so with three questions it can only be 0, 1/3, 2/3 or 1 --
which is exactly the three-valued distribution STATUS 38 measured. More
questions is not a virtue in itself (the prereg warns that easy questions
dilute), but a score with more reachable values can express a difference that a
coarser one rounds away. Both numbers are reported; neither is a threshold.

    envs/core/python.exe scripts/v4_question_coverage.py
"""

from __future__ import annotations

import collections
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfsight.schemas import QuestionFormat  # noqa: E402
from selfsight.v4.probe import spec_questions  # noqa: E402
from selfsight.v4.spec import SceneSpec  # noqa: E402
# The published STATUS 38 figures come out of this module, so the split below
# is the same computation on subsets rather than a second implementation of it.
from v4_selector_resolution import (  # noqa: E402
    load_gate_b_pools,
    pool_candidates,
    resolution,
)

GATE_B = Path("runs/v4/gate-b")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_specs(runs: list[str]) -> dict[str, SceneSpec]:
    specs: dict[str, SceneSpec] = {}
    for name in runs:
        for row in read_jsonl(Path(name) / "manifest.jsonl"):
            spec = SceneSpec.from_dict(row["spec"])
            specs[spec.spec_id] = spec
    return specs


def reachable_scores(n: int) -> int:
    """Distinct values of correct/n, if every question is answered."""
    return len({Fraction(k, n) for k in range(n + 1)}) if n else 0


def main() -> None:
    pools = read_jsonl(GATE_B / "pools.jsonl")
    runs = json.loads((GATE_B / "runs.json").read_text(encoding="utf-8"))["runs"]
    specs = load_specs(runs)

    missing = [p["spec_id"] for p in pools if p["spec_id"] not in specs]
    if missing:
        raise SystemExit(f"{len(missing)} pools have no spec in the source manifests")

    before = collections.Counter()
    after = collections.Counter()
    singletons = 0
    objects = collections.Counter()
    binary_counts = open_counts = 0

    for pool in pools:
        spec = specs[pool["spec_id"]]
        old = pool["questions"]
        new = spec_questions(spec)
        before[len(old)] += 1
        after[len(new)] += 1
        objects[len(spec.objects)] += 1
        singletons += sum(1 for obj in spec.objects if obj.count == 1)
        binary_counts += sum(1 for q in old if ":count:" in q["question_id"])
        open_counts += sum(1 for q in new if ":count:" in q.question_id)

    n = len(pools)
    print(f"{n} pools, from {len(runs)} runs\n")

    print("objects per spec")
    for size, count in sorted(objects.items()):
        print(f"  {size} objects   {count:4d}  {count / n:6.1%}")

    print("\nquestions per pool")
    for label, table in (("before", before), ("after", after)):
        total = sum(size * count for size, count in table.items())
        detail = "  ".join(f"{size}:{count}" for size, count in sorted(table.items()))
        print(f"  {label:6s} mean {total / n:.2f}   {detail}")

    print("\nreachable score values per pool (correct/answered, nothing abstaining)")
    for label, table in (("before", table_of(before)), ("after", table_of(after))):
        detail = "  ".join(f"{values}:{count}" for values, count in sorted(table.items()))
        print(f"  {label:6s} {detail}")

    print("\ncoverage gaps this change closes")
    print(f"  gap 1  singleton objects that had no count question: {singletons}"
          f"  ({singletons / max(1, sum(size * c for size, c in objects.items())):.1%} of all objects)")
    print(f"  gap 2  count atoms binary -> open: {binary_counts} -> {open_counts}")
    print("  gap 3  extra objects nobody asked for: still uncovered -- see the note below")

    forms = collections.Counter(q.question_format for pool in pools
                                for q in spec_questions(specs[pool["spec_id"]]))
    print("\nquestion formats after")
    for form, count in sorted(forms.items(), key=lambda item: item[0].value):
        print(f"  {form.value:14s} {count:5d}")
    assert forms[QuestionFormat.OPEN] == open_counts

    split_by_shape()
    discrimination()

    print("""
Gap 3 is not closed here, on purpose. A spec-determined question about an
object the spec never mentions has gold 0 for every candidate, so it is correct
for a model that recited the prompt and for a model that looked at a clean
picture alike -- it would raise the share of candidates at the ceiling, which is
the number the prereg is trying to bring down. Asking "what do you see" and
grading it against the picture is the right instrument, and that is the
image-gold half of the selector (prereg section 1, part 2), not this one.""")


def shape_of(prompt_id: str) -> str:
    return ("1plus1plus1" if prompt_id.split(":", 1)[0].endswith("1plus1plus1")
            else "2plus1")


def split_by_shape() -> None:
    """STATUS 38's resolution numbers, cut by how many counting questions a pool had.

    The corpus has exactly two spec shapes, 116 pools each. A `2plus1` spec asks
    for one object twice, so the old builder gave it one counting question. A
    `1plus1plus1` spec is three singletons, so it got **no counting question at
    all**: three existence questions, answer yes, on a picture drawn from a
    prompt that asked for exactly those three things.

    The obvious prediction is that the pools with no counting question are the
    degenerate ones. **It is wrong, and it is left here because it is wrong.**
    They are the *less* degenerate half. The prediction ignored a confound in
    its own construction: the two shapes differ in how hard the scene is to
    draw, not only in which questions were asked. `discrimination` below is the
    comparison that holds the pool fixed.

    Same `resolution` function as the published table, same `naive` criterion,
    only the input is split.
    """
    pools = load_gate_b_pools()
    labels = {"2plus1": "2plus1 (1 count question)",
              "1plus1plus1": "1plus1plus1 (0 count questions)"}
    groups: dict[str, list] = {label: [] for label in labels.values()}
    for prompt_id, pool in pools.items():
        groups[labels[shape_of(prompt_id)]].append(pool_candidates(pool))

    print("\nSTATUS 38 resolution, split by how many counting questions the pool had")
    print(f"  {'':34s} {'pools':>6s} {'ceiling':>9s} {'all_tied':>9s} {'unique_top':>11s}")
    for label, subset in groups.items():
        res = resolution(subset)
        print(f"  {label:34s} {res['n_pools']:6d} {res['ceiling']:9.1%}"
              f" {res['all_tied']:9.1%} {res['unique_top']:11.1%}")


def load_answers(arm: str = "naive") -> dict[str, dict[str, list[str]]]:
    """prompt_id -> question_id -> the normalized answer from each candidate."""
    out: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for row in read_jsonl(GATE_B / f"observations.{arm}.jsonl"):
        for answer in row["observation"]["answers"]:
            out[row["prompt_id"]][answer["question_id"]].append(
                answer["normalized_answer"])
    return out


def discrimination() -> None:
    """Which question actually separated candidates, holding the pool fixed.

    A question only contributes to selection if the candidates in a pool answer
    it differently. Within the 2plus1 pools every candidate saw both an
    existence question and the binary counting question, so comparing the two
    there confounds nothing -- same pools, same pictures, same observer.

    The last column is the one that matters for the prereg. `all_tied` is
    exactly "no question varied" (it reproduces to the digit below, so no pool
    is tied by two differences cancelling out), and dropping the counting
    question from the tally moves it by 1.7 points on 2plus1: the binary count
    untied 2 pools out of 116. Whatever the open form is worth, that is the
    figure it replaces.
    """
    answers = load_answers()
    gold = {(pool["prompt_id"], q["question_id"]): q["expected_answer"]
            for pool in read_jsonl(GATE_B / "pools.jsonl") for q in pool["questions"]}

    per_kind: dict[tuple[str, str], list[int]] = collections.defaultdict(
        lambda: [0, 0, 0, 0])
    per_shape: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for prompt_id, questions in answers.items():
        shape = shape_of(prompt_id)
        varies = set()
        for question_id, replies in questions.items():
            kind = "count" if ":count:" in question_id else "exists"
            cell = per_kind[(shape, kind)]
            cell[0] += 1
            cell[1] += len(set(replies)) > 1
            cell[2] += sum(1 for r in replies if r != gold[(prompt_id, question_id)])
            cell[3] += len(replies)
            if len(set(replies)) > 1:
                varies.add(question_id)
        per_shape[shape][0] += 1
        per_shape[shape][1] += not varies
        per_shape[shape][2] += not {q for q in varies if ":count:" not in q}

    print("\nwhich question separated candidates (naive arm, pool held fixed)")
    print(f"  {'pool shape':14s} {'question':8s} {'pool x q':>9s} {'varies':>8s} {'wrong':>8s}")
    for (shape, kind), (n, varied, wrong, replies) in sorted(per_kind.items()):
        print(f"  {shape:14s} {kind:8s} {n:9d} {varied / n:8.1%} {wrong / replies:8.1%}")

    print("\n  what the binary counting question was worth")
    print(f"  {'pool shape':14s} {'all_tied':>10s} {'existence alone':>17s} {'count untied':>14s}")
    for shape, (n, none_varied, no_exists_varied) in sorted(per_shape.items()):
        bought = (no_exists_varied - none_varied) / n
        note = f"{bought:+.1%}" if shape == "2plus1" else "n/a"
        print(f"  {shape:14s} {none_varied / n:10.1%} {no_exists_varied / n:17.1%}"
              f" {note:>14s}")
    print("  (1plus1plus1 has no counting question, so its two columns must agree)")

    tied = sum(s[1] for s in per_shape.values()) / sum(s[0] for s in per_shape.values())
    floor = sum(s[2] for s in per_shape.values()) / sum(s[0] for s in per_shape.values())
    print(f"""
  Read this as the bill the new questions have to pay. all_tied is {tied:.1%}
  today and the prereg registered <= 25%. Existence questions leave {floor:.1%} of
  pools untied on their own, and the new construction does not change them --
  so the count atoms have to untie nearly every one of those pools by
  themselves. That is a demanding target, it was not visible before this audit,
  and it is registered here rather than discovered after the rerun.""")


def table_of(counts: collections.Counter) -> collections.Counter:
    out: collections.Counter = collections.Counter()
    for size, count in counts.items():
        out[reachable_scores(size)] += count
    return out


if __name__ == "__main__":
    main()
