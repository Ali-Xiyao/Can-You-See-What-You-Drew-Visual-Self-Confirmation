"""Every registered consequence has a constant, and a driver that prints it.

Four times now a rule in the pre-registration has turned out to have no code
path -- endpoint 2's caliber (deviation 11), endpoints 2 and 3's analysis rules
(13), deviations 7.2 and 7.3 (14), and failure conditions 1 and 2 (15). Each
time the rule was unambiguous and the gap was that nobody had written it down
anywhere a program would run.

The registry below is the fifth instance's tripwire. It is not a substitute for
reading the pre-registration; it pins the part that is mechanically checkable:
the unfavourable outcome of every registered decision has a wording constant,
the constant says where it came from, and the driver that computes the decision
mentions the constant by name rather than restating it.

A restated sentence is the failure this catches. Two copies of "the claim
retreats to inference time" can differ by one clause, and the one in the output
is the one the paper will be written from.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# (module, constant, the driver that must print it, a phrase the constant must
#  keep, the pre-registration source it must cite)
CONSEQUENCES = [
    ("selfsight.analysis.endpoint1", "NOT_DETECTED_CONSEQUENCE",
     "v4_e3_endpoint1.py", "may not claim that BSV improves generation",
     "failure condition 1"),
    ("selfsight.analysis.endpoint2", "FALSIFIED",
     # Not the bare word "falsified": that is also the name of the property the
     # driver branches on, and a phrase that matches an identifier would pass
     # the restatement check for the wrong reason.
     "v4_e3_endpoint2.py", "the mechanism in section 2 is falsified",
     "failure condition 2"),
    ("selfsight.analysis.endpoint3", "DOWNGRADE",
     "v4_e3_endpoint3.py", "not a dose-response", "endpoint 3"),
    ("selfsight.analysis.drift", "NOT_SEPARABLE",
     "v4_e3_drift.py", "retreats to inference time", "deviation 7.2"),
]

# Registered outcomes that are not a verdict about the data but a verdict about
# the instrument: the endpoint did not get measured. Deviation 11.3 and 13.2
# point 7 both make this a reportable result with a required refusal attached,
# so the refusal is checked too -- an unmeasured endpoint quietly backfilled
# from the corpus replay is the specific thing they forbid.
NOT_DONE_REFUSALS = [
    ("v4_e3_endpoint2.py", "substituted_prompted_gap", "deviation 11.3"),
    ("v4_e3_endpoint3.py", "substituted_corpus_replay", "deviation 13.2 point 7"),
]


@pytest.mark.parametrize(("module_name", "constant", "driver", "phrase", "source"),
                         CONSEQUENCES, ids=[row[1] for row in CONSEQUENCES])
def test_the_consequence_exists_and_names_its_source(module_name, constant, driver,
                                                     phrase, source):
    module = importlib.import_module(module_name)
    assert hasattr(module, constant), f"{module_name}.{constant} is gone"
    text = getattr(module, constant)
    assert isinstance(text, str) and text.strip(), f"{constant} is not a sentence"
    assert phrase in text, f"{constant} no longer says {phrase!r}"
    assert source in text, f"{constant} does not cite {source!r}"


@pytest.mark.parametrize(("module_name", "constant", "driver", "phrase", "source"),
                         CONSEQUENCES, ids=[row[1] for row in CONSEQUENCES])
def test_the_driver_prints_the_constant_rather_than_a_copy(module_name, constant,
                                                           driver, phrase, source):
    body = (ROOT / "scripts" / driver).read_text(encoding="utf-8")
    assert constant in body, f"scripts/{driver} never mentions {constant}"
    # The sentence itself must not be inlined anywhere but the module that owns
    # it. A driver holding its own copy is two sentences that can drift.
    assert phrase not in body, (
        f"scripts/{driver} restates {constant} instead of importing it")


@pytest.mark.parametrize(("driver", "field", "source"), NOT_DONE_REFUSALS,
                         ids=[row[0] for row in NOT_DONE_REFUSALS])
def test_not_done_records_the_refusal_to_substitute(driver, field, source):
    body = (ROOT / "scripts" / driver).read_text(encoding="utf-8")
    assert f'"{field}": False' in body, (
        f"scripts/{driver} no longer records {field}=False; {source} makes the "
        f"refusal part of the not-done result, not an omission")


def test_every_analysis_module_with_a_verdict_is_covered():
    """A new endpoint module with a verdict and no entry above fails here.

    The registry is hand-maintained, which is fine as long as forgetting to
    maintain it is noisy. Any module under analysis/ that defines a wording
    constant has to be listed, so adding one and not registering its driver is
    the thing that goes red rather than the thing that goes unnoticed.
    """

    listed = {name for name, _, _, _, _ in CONSEQUENCES}
    found = set()
    for path in sorted((ROOT / "src" / "selfsight" / "analysis").glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module = importlib.import_module(f"selfsight.analysis.{path.stem}")
        for attribute in ("DOWNGRADE", "FALSIFIED", "NOT_SEPARABLE",
                          "NOT_DETECTED_CONSEQUENCE"):
            if hasattr(module, attribute):
                found.add(f"selfsight.analysis.{path.stem}")
    assert found == listed, f"unregistered: {sorted(found - listed)}"
