"""v3.0 primitives: bounded candidate-bank supply and paired gradient inference.

Two measured v2.x failures motivate this package:

* random K=4 pools were informative only 21% of the time, so selection-based
  self-training had almost no variance to exploit (`bank`);
* the disjoint split-half "noise floor" measured a different quantity than the
  paired cross-criterion cosine it was gating, which produced two misleading red
  gates (`paired`).

A third failure is addressed by `selectors` and `observation_ladder`: the Naive
arm asked the atomic question rather than recovering the prompt, so it was the
same function as RFO-Self at step 0 (agreement exactly 1.000).

See `docs/EVIDENCE_LOG.md` sections 4 and 5.
"""

from selfsight.v3.bank import (
    BalancedPool,
    BankCandidate,
    assert_disjoint_seed_domains,
    bank_supply_report,
    build_balanced_pool,
)
from selfsight.v3.observation_ladder import (
    LADDER,
    ObservationCondition,
    contextualized_question,
    descending_rungs,
)
from selfsight.v3.paired import (
    GramMatrices,
    PerPromptGradientStore,
    cosine_from_gram,
    gram_matrices,
    paired_bootstrap_cosine,
    paired_bootstrap_difference,
    sample_noise_diagnostic,
)
from selfsight.v3.selectors import (
    V3_ARMS,
    cycle_scores,
    cycle_selection,
    select_by_score,
    selection_agreement,
)
from selfsight.v3.supply import run_bank_probe
from selfsight.v3.vocabulary import (
    assert_display_vocabulary,
    audit_rows,
    display_text,
    scene_vocabulary_metadata,
)

__all__ = [
    "LADDER",
    "V3_ARMS",
    "BalancedPool",
    "BankCandidate",
    "GramMatrices",
    "ObservationCondition",
    "PerPromptGradientStore",
    "assert_disjoint_seed_domains",
    "assert_display_vocabulary",
    "audit_rows",
    "bank_supply_report",
    "build_balanced_pool",
    "contextualized_question",
    "cosine_from_gram",
    "cycle_scores",
    "cycle_selection",
    "descending_rungs",
    "display_text",
    "gram_matrices",
    "paired_bootstrap_cosine",
    "paired_bootstrap_difference",
    "run_bank_probe",
    "sample_noise_diagnostic",
    "scene_vocabulary_metadata",
    "select_by_score",
    "selection_agreement",
]
