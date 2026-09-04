"""Tests for the pre-declared secondary analyses of 27.

The Mantel-Haenszel estimate is hand-written -- the environments this runs in
have no scipy -- and its output is now quoted in the proposal. So it is pinned
here against two things it must agree with by algebra rather than by my reading
of a formula: the crude odds ratio when there is only one stratum, and the
Yates-corrected Pearson chi-square, which the single-stratum MH chi-square must
equal up to exactly (n-1)/n. A sign error or a swapped row total breaks both.

The Simpson case is the one that says the estimator is doing its job at all: two
strata each with an odds ratio of exactly 1, pooled crudely into 8.56.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_conf_fork_secondary", ROOT / "scripts" / "v4_conf_fork_secondary.py")
sec = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sec)


def table(a: int, n1: int, c: int, n2: int) -> dict[str, tuple[int, int]]:
    return {"prone": (a, n1), "never": (c, n2)}


def yates_chi2(a: int, b: int, c: int, d: int) -> float:
    """Pearson chi-square with the continuity correction, computed directly."""
    n = a + b + c + d
    num = n * (abs(a * d - b * c) - n / 2) ** 2
    return num / ((a + b) * (c + d) * (a + c) * (b + d))


@pytest.mark.parametrize("cells", [(21, 173, 15, 149), (22, 77, 10, 62),
                                   (5, 40, 12, 40), (1, 10, 8, 80)])
def test_one_stratum_reduces_to_the_crude_odds_ratio(cells):
    a, n1, c, n2 = cells
    orat, _, _ = sec.mantel_haenszel([table(a, n1, c, n2)])
    assert orat == pytest.approx((a * (n2 - c)) / ((n1 - a) * c))


@pytest.mark.parametrize("cells", [(21, 173, 15, 149), (22, 77, 10, 62),
                                   (5, 40, 12, 40)])
def test_one_stratum_matches_the_yates_chi_square_up_to_the_n_minus_one_factor(cells):
    a, n1, c, n2 = cells
    b, d, n = n1 - a, n2 - c, n1 + n2
    _, _, two = sec.mantel_haenszel([table(a, n1, c, n2)])
    expected = math.erfc(math.sqrt(yates_chi2(a, b, c, d) * (n - 1) / n / 2))
    assert two == pytest.approx(expected, rel=1e-12)


def test_stratifying_removes_a_confound_the_crude_table_invents():
    """Each stratum has an odds ratio of exactly 1; pooled crudely it is 8.56."""
    low, high = table(1, 10, 8, 80), table(72, 90, 16, 20)
    crude = table(1 + 72, 10 + 90, 8 + 16, 80 + 20)
    crude_or = (73 * 76) / (27 * 24)
    assert crude_or == pytest.approx(8.56, abs=0.01)
    assert sec.mantel_haenszel([crude])[0] == pytest.approx(crude_or)

    orat, one, _ = sec.mantel_haenszel([low, high])
    assert orat == pytest.approx(1.0)
    assert one > 0.05


def test_the_one_sided_p_is_for_prone_being_high():
    """Reversing the two strata must send the one-sided p to the other side."""
    _, high, two = sec.mantel_haenszel([table(30, 100, 10, 100)])
    _, low, two_rev = sec.mantel_haenszel([table(10, 100, 30, 100)])
    assert high < 0.05 < low
    assert high == pytest.approx(1.0 - low)
    assert two == pytest.approx(two_rev)


def test_more_of_the_same_evidence_does_not_move_the_estimate_but_sharpens_it():
    once = table(22, 77, 10, 62)
    or1, p1, _ = sec.mantel_haenszel([once])
    or2, p2, _ = sec.mantel_haenszel([once, dict(once)])
    assert or2 == pytest.approx(or1)
    assert p2 < p1


def test_an_empty_stratum_is_ignored_rather_than_dividing_by_zero():
    live = table(22, 77, 10, 62)
    assert sec.mantel_haenszel([live, table(0, 0, 0, 0)]) == \
        sec.mantel_haenszel([live])


def units(*rows: tuple[str, str, str, bool]) -> list[dict[str, object]]:
    return [{"pair_id": p, "image": i, "cell": c, "stratum": s, "flip": f}
            for p, i, c, s, f in rows]


def test_clustering_draws_exactly_one_pair_per_unit_and_stratum():
    rows = units(
        ("img1:0", "img1", "spec1|pear", "prone", True),
        ("img1:1", "img1", "spec1|pear", "prone", False),
        ("img1:2", "img1", "spec2|cup", "never", True),
        ("img2:0", "img2", "spec1|pear", "prone", False),
    )
    by_image = sec.cluster(rows, "image")
    assert by_image["prone"][1] == 2, "img1 and img2 each contribute one prone pair"
    assert by_image["never"][1] == 1

    by_cell = sec.cluster(rows, "cell")
    assert by_cell["prone"][1] == 1, "both images share one cell, so one draw"
    assert by_cell["never"][1] == 1


def test_clustering_is_reproducible_because_the_seed_is_pre_declared():
    rows = units(*[(f"img1:{k}", "img1", "spec1|pear", "prone", k % 3 == 0)
                   for k in range(12)])
    assert sec.cluster(rows, "image") == sec.cluster(rows, "image")
    assert fork_seed_is_a_string()


def fork_seed_is_a_string() -> bool:
    """random.Random seeds on int and str differ, so the type is load-bearing."""
    return isinstance(sec.fork.CLUSTER_SEED, str)
