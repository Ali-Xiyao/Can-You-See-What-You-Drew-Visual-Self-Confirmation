# Correction: the run produces 12 checkpoints per arm, not 13

New file. `NOTES.md`, `availability.py` and `output.txt` are left exactly as
they were — the hard rule is that a flawed record stays and the flaw gets
registered beside it.

## What is wrong

`output.txt` costs both re-measurement passes over **26** arm-checkpoints for
the main run and **130** across the five replicates. Both numbers are one
checkpoint per arm too many.

I took the 26 from the pre-registration, which says
`13 checkpoint × 2 臂 = 26 点` (端点 3), and deviation 9.4 scales it to
`5 × 13 × 2`. I did not check it against the supervisor.

## What the supervisor actually does

`scripts/run_decoupling_pilot.py:355` is `for index in range(self.limit)` with
`self.limit == training.rounds == 11`, so rounds 0 through 10. Inside the
loop:

- at `index == 0` only, `evaluate(arm, -1, ...)` measures the saved untrained
  adapter, which is step `(-1 + 1) * 8 = 0`;
- every round then measures its own checkpoint at step `(index + 1) * 8`,
  which runs 8, 16, … 88.

That is **12** evaluated steps per arm — 0, 8, 16, 24, 32, 40, 48, 56, 64, 72,
80, 88 — and 24 arm-checkpoints for the run. Across five replicates, 120.

The evaluations already on disk agree: `step-00000`, `step-00008`,
`step-00016`, `step-00024` after three rounds, which is 1 + 3 rather than
1 + 3 + 1.

## What it changes

Nothing that can go wrong, which is the only reason this is a note and not a
deviation. Both passes get cheaper in proportion:

| pass | as costed (26 / 130) | corrected (24 / 120) |
|---|---|---|
| endpoint 2, blind gap | 0.6 h main, 2.8 h x5 | 0.55 h main, 2.6 h x5 |
| endpoint 3, selection + conflict | 4.1 h main, 20.3 h x5 | 3.8 h main, 18.7 h x5 |
| both, one adapter load | 4.6 h main, 23.1 h x5 | 4.3 h main, 21.3 h x5 |

The per-unit minutes in `output.txt` were measured, not assumed, and they do
not move. Only the multiplier does.

## What it does not change

The pre-registration's 26 and 130 stay as written. No registered criterion
counts checkpoints: endpoint 2 fits a slope through however many checkpoints
exist and deviation 13.1 point 6 requires naming the ones it could not get;
endpoint 3's cluster unit is `(seed, checkpoint)` regardless of how many
checkpoints a seed has. A count that appears only in a cost estimate and a
sentence of motivation is not something to amend a frozen document over.

`src/selfsight/analysis/endpoint3.py` therefore derives the checkpoint list
from disk and hardcodes no count. So does `scripts/v4_e3_blind_observe.py`.

Recorded 2026-09-09, while the main run was at `rfo_gold.step-00024`.
