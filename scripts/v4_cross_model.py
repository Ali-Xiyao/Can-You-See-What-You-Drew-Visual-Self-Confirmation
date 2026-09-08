#!/usr/bin/env python
"""E4: is "the description overrides the pixels" a property of one checkpoint?

The cross-sectional result (`review-packets/context-ablation-20260908`) was
measured on Show-o2-1.5B and nothing else. A reviewer's first question is
whether it survives at another scale and in another architecture. This asks the
same questions, about the same images, of three more models:

    showo2_7b       same family, 4.7x the parameters
    showo_v1        the older Show-o, a different tokeniser and a CLIP tower
    janus_pro_1b    a different lab, and a split visual pathway -- the design
                    usually proposed as the cure for this interference

None of them draws anything and none of them trains. They read images
Show-o2-1.5B drew, which is what makes this a comparison of readers.

Three stages, because they fail in different ways and at different costs:

    preflight   two images, one question, per model. Catches the failure that
                would otherwise be invisible: a model answering from the
                question alone because the picture never reached it.
    observe     both conditions, every model. Hours on a card.
    report      the frozen packet's table, plus the registered verdict.

Each model needs its own interpreter -- the three stacks cannot coexist in one
environment -- and each config names its own under `environment`. Run this from
the project root, where `envs/`, `configs/` and `runs/` all resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from selfsight.analysis.context import (
    CONFLICT,
    abstention_pairs,
    conflict_pairs,
    mcnemar,
    wilson,
)
from selfsight.backbones.registry import (
    backbone_interpreter,
    read_backbone_config,
)

# Absolute, from this file rather than from the working directory. As a relative
# path it named whichever checkout the shell happened to be sitting in, which is
# not necessarily the one this script came from.
PIPELINE = str(ROOT / "scripts" / "v4_run_pipeline.py")
DEFAULT_RUNS = ("runs/v4/main-2plus1", "runs/v4/main-1plus1plus1")

# The registered three of E4. Janus-Pro arrived with deviation 1 and is
# reported beside them rather than inside them: "at least two of three" was
# written down before any of this ran, and a fourth model must not be able to
# supply the majority it was not registered for.
REGISTERED = ("showo2_1p5b", "showo2_7b", "showo_v1")
BASELINE = "showo2_1p5b"

MODELS = {
    "showo2_1p5b": None,  # already measured; its rows are answers[.prompted].jsonl
    "showo2_7b": "configs/backbones/showo2_7b.yaml",
    "showo_v1": "configs/backbones/showo_v1.yaml",
    "janus_pro_1b": "configs/backbones/janus_pro_1b.yaml",
}

CONDITIONS = ("image_only", "prompted")


def answer_name(model: str, condition: str) -> str:
    parts = ["answers"]
    if condition == "prompted":
        parts.append("prompted")
    if model != BASELINE:
        parts.append(model)
    return ".".join(parts) + ".jsonl"


def child_env() -> dict[str, str]:
    """The environment for a stage subprocess, pinned to the checkout this file is in.

    Every interpreter under `envs/` has selfsight installed editable against the
    MAIN tree's `src`, so a child that simply imports selfsight gets the main
    tree's copy regardless of which checkout launched it. The parent does not:
    the `sys.path.insert` above binds it to its own. Run this script from a
    worktree and the two disagree -- the parent reads the branch, while
    `v4_run_pipeline.py observe`, which is where every answer is actually
    graded, reads main.

    Nothing announces that. The preflight passes, the worktree's tests pass, and
    the answers come back graded by whichever grader main happens to hold.
    Measured before this was written: with no PYTHONPATH the janus interpreter
    resolves selfsight to the main tree, whose `questions` has no
    `LABELLED_CHOICE`; with PYTHONPATH set to this checkout's `src` it resolves
    here, and it does. The mismatch was reachable, not hypothetical.

    PYTHONPATH rather than reinstalling editable against this checkout, because
    the reinstall is global -- it would move the running experiment's
    interpreter too, and that run's manifest already recorded the digests of
    the sources it started with.

    The four settings after PYTHONPATH are the ones `run_decoupling_pilot.py`
    gives its own stages, and they are here for the same reasons.
    `HF_HUB_OFFLINE` is the load-bearing one: without it an adapter that cannot
    find a snapshot locally reaches for the network and fetches weights, and no
    measurement in this project may acquire a model on its own.
    `CUDA_VISIBLE_DEVICES` is dropped because every stage is told which card to
    use by `--device`, and an inherited mask renumbers the cards underneath
    that flag without saying so.
    """
    env = os.environ.copy()
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{inherited}" if inherited else str(SRC)
    env.update(PYTHONUTF8="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------------ preflight


PREFLIGHT_QUESTION = "What colour is the largest object? Answer with one word."


def stage_preflight(args: argparse.Namespace) -> None:
    """Ask one question about several images and require the answers to differ.

    A model whose image never reaches the language tower answers every question
    identically and fills the run with rows. E4 would then report that the
    effect vanishes on that architecture, which is a publishable-looking claim
    and a false one. Two minutes here, on the CPU if the cards are busy.
    """

    images = args.images or sorted(
        str(path) for path in Path(args.runs[0], "images").glob("*.png")
    )[: args.count]
    if len(images) < 2:
        raise SystemExit("preflight needs at least two images to compare answers")

    failures = []
    for model in args.models:
        config = MODELS[model]
        if config is None:
            print(f"{model}: already measured, nothing to preflight")
            continue
        interpreter = backbone_interpreter(read_backbone_config(config))
        command = [interpreter, "-u", __file__, "_answer_once",
                   "--backbone-config", config, "--device", args.device,
                   "--images", *images]
        print(f"{model}: {interpreter} ({len(images)} images)", flush=True)
        finished = subprocess.run(command, capture_output=True, text=True,
                                  check=False, env=child_env())
        if finished.returncode != 0:
            failures.append(f"{model}: exited {finished.returncode}\n{finished.stderr[-2000:]}")
            continue
        answers = json.loads(finished.stdout.strip().splitlines()[-1])
        for image, answer in zip(images, answers):
            print(f"    {Path(image).name[:44]:<46} {answer!r}")
        if len({answer for answer in answers if answer}) < 2:
            failures.append(f"{model}: the same answer for every image -- "
                            "the picture may not be reaching the model")
        else:
            print("    -> the answer depends on the image\n")

    if failures:
        raise SystemExit("preflight failed:\n" + "\n".join(failures))
    print("all models read the picture")


def stage_answer_once(args: argparse.Namespace) -> None:
    """One question, several images, answers as JSON on the last stdout line.

    A subcommand rather than a function because the caller is a different
    interpreter: this is the only code in the file that imports an adapter.
    """

    from selfsight.backbones.registry import build_observer
    from selfsight.schemas import AtomicQuestion, QuestionFamily

    question = AtomicQuestion(question_id="preflight", atom_id="preflight",
                              family=QuestionFamily.COLOR, text=PREFLIGHT_QUESTION,
                              expected_answer="red")
    observer = build_observer(args.backbone_config, device=args.device)
    answers = [observer.observe_atoms(image, [question]).answers[0].normalized_answer
               for image in args.images]
    print(json.dumps(answers))


# -------------------------------------------------------------------- observe


def stage_observe(args: argparse.Namespace) -> None:
    for model in args.models:
        config = MODELS[model]
        if config is None:
            print(f"{model}: already measured, skipping")
            continue
        interpreter = backbone_interpreter(read_backbone_config(config))
        for run in args.runs:
            for condition in CONDITIONS:
                existing = Path(run) / answer_name(model, condition)
                if existing.exists() and not args.overwrite:
                    print(f"{model} {Path(run).name} {condition}: already answered")
                    continue
                command = [interpreter, "-u", PIPELINE, "observe",
                           "--run", run, "--device", args.device,
                           "--condition", condition,
                           "--backbone-config", config]
                if args.overwrite:
                    command.append("--overwrite")
                print(f"{model} {Path(run).name} {condition}: {' '.join(command)}", flush=True)
                subprocess.run(command, check=True, env=child_env())


# --------------------------------------------------------------------- report


def measure(runs: list[str], model: str) -> dict | None:
    """The frozen packet's two numbers, for one model, pooled over the runs."""

    rows: dict[str, list[dict]] = {}
    for condition in CONDITIONS:
        collected: list[dict] = []
        for run in runs:
            path = Path(run) / answer_name(model, condition)
            if not path.exists():
                return None
            collected.extend(read_jsonl(path))
        rows[condition] = collected

    cells = {}
    for condition, collected in rows.items():
        live = [row for row in collected
                if not row["abstain"] and row["gold_source"] == CONFLICT]
        hits = sum(bool(row["correct"]) for row in live)
        point, low, high = wilson(hits, len(live))
        cells[condition] = {"point": point, "low": low, "high": high, "n": len(live)}

    pairs = conflict_pairs(rows["image_only"], rows["prompted"])
    blind_only, told_only, p_value = mcnemar(pairs, sided=1)

    # Which trials survive to be paired is decided by how the model phrased
    # itself, and the phrasing is not independent of the condition. Janus
    # answers an absence question with "The image does not contain any
    # notebooks", which is correct and matches neither option, so it abstains.
    # If it does that more in one condition than the other, the two marginal
    # accuracies are over different subsets. Two-sided: an imbalance either way
    # is the same problem.
    skipped = abstention_pairs(rows["image_only"], rows["prompted"])
    blind_skipped, told_skipped, skip_p = mcnemar(skipped, sided=2)

    return {
        "model": model,
        "image_only": cells["image_only"],
        "prompted": cells["prompted"],
        "delta": cells["prompted"]["point"] - cells["image_only"]["point"],
        "pairs": len(pairs),
        "blind_only": blind_only,
        "told_only": told_only,
        "p_one_sided": p_value,
        "decisive_trials": len(skipped),
        "blind_only_abstained": blind_skipped,
        "told_only_abstained": told_skipped,
        "p_abstention_imbalance": skip_p,
        # Registered in PREREG E4: prompted below image_only on the decisive
        # trials, one-sided. Nothing about effect size -- E4 asks whether the
        # direction survives, and the size is reported for the reader.
        "replicates": bool(cells["prompted"]["point"] < cells["image_only"]["point"]
                           and p_value < 0.05),
    }


def cell(value: dict) -> str:
    return f"{value['point']:.3f} [{value['low']:.3f},{value['high']:.3f}] n={value['n']}"


def stage_report(args: argparse.Namespace) -> None:
    results = {model: measure(args.runs, model) for model in args.models}

    print("accuracy on the decisive trials (image_differs_from_spec), Wilson 95%")
    print(f"pooled over {', '.join(Path(run).name for run in args.runs)}\n")
    print(f"{'model':<15}{'image_only':<27}{'prompted':<27}{'delta':>7}{'p (1s)':>11}")
    for model, value in results.items():
        if value is None:
            print(f"{model:<15}not measured")
            continue
        print(f"{model:<15}{cell(value['image_only']):<27}{cell(value['prompted']):<27}"
              f"{value['delta']:>+7.3f}{value['p_one_sided']:>11.2e}")

    print("\npaired on (image, question), exact one-sided McNemar\n")
    print(f"{'model':<15}{'pairs':>7}{'blind only':>12}{'told only':>11}{'replicates':>12}")
    for model, value in results.items():
        if value is None:
            continue
        print(f"{model:<15}{value['pairs']:>7}{value['blind_only']:>12}"
              f"{value['told_only']:>11}{value['replicates']!s:>12}")
    print("\n'blind only' = right without the description in context, wrong with it.")

    print("\nwhich trials were answerable at all, paired the same way\n")
    print(f"{'model':<15}{'decisive':>9}{'paired':>8}{'blind only':>12}"
          f"{'told only':>11}{'p (2s)':>11}")
    for model, value in results.items():
        if value is None:
            continue
        print(f"{model:<15}{value['decisive_trials']:>9}{value['pairs']:>8}"
              f"{value['blind_only_abstained']:>12}{value['told_only_abstained']:>11}"
              f"{value['p_abstention_imbalance']:>11.2e}")
    lopsided = [model for model, value in results.items()
                if value is not None and value["p_abstention_imbalance"] < 0.05]
    if lopsided:
        print(f"\n  {', '.join(lopsided)}: declines to answer at different rates in the "
              "two conditions.\n  The paired test above is unaffected -- it only uses "
              "trials answered in both --\n  but the two marginal accuracies are over "
              "subsets the model chose. Report both\n  columns' n and say so; do not "
              "drop the model and do not drop the row.")
    else:
        print("\n  no model declines to answer at a different rate in the two conditions,\n"
              "  so the paired subset is not one the model selected.")

    # The registered verdict, computed over the registered three only.
    measured = [model for model in REGISTERED if results.get(model) is not None]
    replicated = [model for model in measured if results[model]["replicates"]]
    print(f"\nregistered set {REGISTERED}: {len(replicated)}/{len(measured)} replicate "
          f"({', '.join(replicated) or 'none'})")
    if len(measured) < len(REGISTERED):
        print("  VERDICT DEFERRED: "
              f"{', '.join(set(REGISTERED) - set(measured))} not measured yet")
    elif len(replicated) >= 2:
        print("  PREREG E4 met: the generalisation claim in section 6 stands, "
              "labelled as an inference-time replication on frozen negative controls")
    else:
        print("  PREREG E4 not met: write 'appears on the smallest model we tested "
              "and not on the larger ones' and shrink the claim. This is a result, "
              "not a failure -- do not add models until it comes out the other way")

    extra = [model for model in args.models if model not in REGISTERED
             and results.get(model) is not None]
    for model in extra:
        outcome = "replicates" if results[model]["replicates"] else "does not replicate"
        print(f"\nbeyond the registered set: {model} {outcome}. Deviation 1 added it as "
              "a cross-family check; it is reported, not counted toward the 2-of-3.")

    output = Path(args.out or Path(args.runs[0]) / "cross_model.json")
    output.write_text(json.dumps(
        {"runs": args.runs, "registered": list(REGISTERED),
         "results": {k: v for k, v in results.items() if v is not None}}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
        target.add_argument("--models", nargs="+", default=list(MODELS),
                            choices=list(MODELS))

    p = sub.add_parser("preflight", help="does the picture reach each model?")
    common(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--images", nargs="+", default=None)
    p.add_argument("--count", type=int, default=3)
    p.set_defaults(func=stage_preflight)

    o = sub.add_parser("observe", help="both conditions, every model")
    common(o)
    o.add_argument("--device", default="cuda:0")
    o.add_argument("--overwrite", action="store_true")
    o.set_defaults(func=stage_observe)

    r = sub.add_parser("report", help="the table and the registered verdict")
    common(r)
    r.add_argument("--out", default=None,
                   help="where the json goes; default cross_model.json in the first run")
    r.set_defaults(func=stage_report)

    a = sub.add_parser("_answer_once", help=argparse.SUPPRESS)
    a.add_argument("--backbone-config", required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--images", nargs="+", required=True)
    a.set_defaults(func=stage_answer_once)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
