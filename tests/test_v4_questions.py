"""Tests for the trials that carry the claim.

A v4 trial is only diagnostic of self-confirmation when the correct answer
contradicts the prompt. The first construction produced those from one family
only -- counting -- so 91.5% of the corpus measured nothing about whether the
model defers to its instruction, and the headline number rested on 154 trials
out of 1803. Each family now takes its distractor from the spec's unmet claims
where the image allows one, and these tests pin that behaviour down per family:
what makes a trial diagnostic, and what must stop it being labelled one.
"""

from __future__ import annotations

import random

from selfsight.v4.questions import (
    build_absence,
    build_binding,
    build_counting,
    build_existence,
    build_spatial,
)
from selfsight.v4.spec import SceneSpec, SpecRelation

RNG = lambda: random.Random(3)  # noqa: E731


def _spec(objects, relations=(), prompt="a scene"):
    return SceneSpec.from_dict(
        {
            "spec_id": "s1",
            "prompt": prompt,
            "objects": [
                {"object": o, "color": c, "count": n} for o, c, n in objects
            ],
            "relations": [
                {"subject": s, "relation": r, "object": t} for s, r, t in relations
            ],
        }
    )


def _det(*items):
    out = []
    for entry in items:
        noun, colour, x = entry
        out.append({"object": noun, "color": colour, "center": [x, 100.0]})
    return out


# --------------------------------------------------------------- existence


def test_existence_is_diagnostic_on_a_substitution():
    """Asked for a candle, drew a bottle: the two options separate the hypotheses.

    A model reading the picture says bottle; a model reciting its prompt says
    candle. Nothing else in the family produces that clean an opposition.
    """
    spec = _spec([("apple", "red", 2), ("candle", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("bottle", "green", 300))
    q = build_existence(spec, detections, RNG(), 0)
    assert q.gold_source == "image_differs_from_spec"
    assert {q.option_a, q.option_b} == {"bottle", "candle"}
    assert (q.option_a if q.gold == "A" else q.option_b) == "bottle"


def test_existence_is_not_called_diagnostic_when_both_options_are_in_the_prompt():
    """A missing object alone is not enough.

    Here the drawn option is also in the spec, so a prompt-reciter sees both
    names in its own instruction and is at chance rather than wrong. Counting
    this as diagnostic would pad the set with trials that cannot separate the two
    hypotheses, which is the failure this whole change is fixing.
    """
    spec = _spec([("apple", "red", 2), ("candle", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("apple", "red", 300))
    q = build_existence(spec, detections, RNG(), 0)
    assert q.gold_source == "image"
    assert {q.option_a, q.option_b} == {"apple", "candle"}


def test_existence_falls_back_to_the_pool_when_the_spec_was_met():
    spec = _spec([("apple", "red", 2), ("mug", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("mug", "white", 300))
    q = build_existence(spec, detections, RNG(), 0)
    assert q.gold_source == "spec_matches_image"
    assert q.metadata["absent_requested"] is False


# ----------------------------------------------------------------- absence


def test_absence_names_a_requested_object_the_image_does_not_contain():
    """The cleanest opposition in the set: the spec says yes, the pixels say no."""
    spec = _spec([("apple", "red", 2), ("candle", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("bottle", "green", 300))
    q = build_absence(spec, detections, RNG(), 0)
    assert q.gold_source == "image_differs_from_spec"
    assert q.metadata["absent"] == "candle"
    assert (q.option_a if q.gold == "A" else q.option_b) == "no candles"


def test_absence_falls_back_to_the_pool_when_everything_requested_was_drawn():
    spec = _spec([("apple", "red", 2), ("mug", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("mug", "white", 300))
    q = build_absence(spec, detections, RNG(), 0)
    assert q.gold_source == "spec_matches_image"
    assert q.metadata["absent"] not in {"apple", "mug"}


# ----------------------------------------------------------------- binding


def test_binding_offers_the_requested_colour_against_the_drawn_one():
    spec = _spec([("pear", "green", 2), ("mug", "white", 1)])
    detections = _det(("pear", "red", 60), ("pear", "red", 140),
                      ("mug", "white", 300))
    q = build_binding(spec, detections, RNG(), 0)
    assert q.gold_source == "image_differs_from_spec"
    assert {q.option_a, q.option_b} == {"red", "green"}
    assert (q.option_a if q.gold == "A" else q.option_b) == "red"
    assert "pears" in q.prompt_text  # two of them were drawn


def test_binding_skips_a_category_drawn_in_two_colours():
    """One noun, two colours, no single correct answer.

    Scoring either colour as the gold would mark a true reply wrong half the
    time, so the question is not asked at all -- the same principle as the
    spatial near-tie.
    """
    spec = _spec([("pear", "green", 2)])
    detections = _det(("pear", "red", 60), ("pear", "yellow", 140))
    assert build_binding(spec, detections, RNG(), 0) is None


def test_binding_distractor_stays_plausible_for_the_category():
    """A purple banana would be answerable from the noun alone."""
    spec = _spec([("banana", "yellow", 2), ("mug", "white", 1)])
    detections = _det(("banana", "yellow", 60), ("banana", "yellow", 140),
                      ("mug", "white", 300))
    for seed in range(25):
        q = build_binding(spec, detections, random.Random(seed), 0)
        if q.metadata["object"] != "banana":
            continue
        assert q.gold_source == "spec_matches_image"
        assert {q.option_a, q.option_b} <= {"yellow", "green"}


# ----------------------------------------------------------------- spatial


def test_spatial_is_diagnostic_when_a_requested_relation_was_violated():
    spec = _spec(
        [("mug", "white", 1), ("apple", "red", 1), ("book", "blue", 1)],
        relations=[("mug", "left_of", "apple")],
    )
    detections = _det(("mug", "white", 400), ("apple", "red", 100),
                      ("book", "blue", 250))
    q = build_spatial(spec, detections, RNG(), 0)
    assert q.gold_source == "image_differs_from_spec"
    assert (q.option_a if q.gold == "A" else q.option_b) == "right"


def test_spatial_honoured_relation_is_not_diagnostic():
    spec = _spec(
        [("mug", "white", 1), ("apple", "red", 1), ("book", "blue", 1)],
        relations=[("mug", "left_of", "apple")],
    )
    detections = _det(("mug", "white", 100), ("apple", "red", 400),
                      ("book", "blue", 250))
    q = build_spatial(spec, detections, RNG(), 0)
    assert q.gold_source == "spec_matches_image"


def test_spatial_without_a_relation_reads_the_image_and_claims_nothing_more():
    """The shipped corpus is relation-free, so this is the common case.

    The trial is still worth asking -- it measures whether the model can read its
    own layout -- but it is not evidence about deferring to the prompt, and the
    tag has to say so.
    """
    spec = _spec([("mug", "white", 1), ("apple", "red", 1), ("book", "blue", 1)])
    detections = _det(("mug", "white", 100), ("apple", "red", 400),
                      ("book", "blue", 250))
    q = build_spatial(spec, detections, RNG(), 0)
    assert q.gold_source == "image"


def test_spatial_near_tie_is_skipped_rather_than_coin_flipped():
    spec = _spec([("mug", "white", 1), ("apple", "red", 1)])
    detections = _det(("mug", "white", 200), ("apple", "red", 210))
    assert build_spatial(spec, detections, RNG(), 0) is None


# ---------------------------------------------------------------- counting


def test_counting_offers_the_requested_count_against_the_drawn_one():
    spec = _spec([("pear", "green", 2), ("mug", "white", 1)])
    detections = _det(("pear", "green", 60), ("pear", "green", 140),
                      ("pear", "green", 220), ("mug", "white", 400))
    q = build_counting(spec, detections, RNG(), 0)
    if q.metadata["object"] == "pear":
        assert q.gold_source == "image_differs_from_spec"
        assert {q.option_a, q.option_b} == {"three", "two"}
        assert (q.option_a if q.gold == "A" else q.option_b) == "three"


# ------------------------------------------------------------------ shared


def test_every_family_can_reach_a_diagnostic_trial():
    """The regression this file exists for.

    Four of the five families produced zero diagnostic trials over 468 real
    images. Spatial is the exception and stays one by choice: it needs a
    requested relation, and the corpus does not carry them.
    """
    spec = _spec(
        [("pear", "green", 2), ("candle", "white", 1)],
        relations=[("pear", "left_of", "candle")],
    )
    detections = _det(("pear", "red", 60), ("pear", "red", 140),
                      ("pear", "red", 220), ("bottle", "green", 300))
    builders = (build_existence, build_absence, build_binding, build_counting)
    sources = {b.__name__: b(spec, detections, RNG(), 0).gold_source for b in builders}
    assert set(sources.values()) == {"image_differs_from_spec"}, sources


def test_the_correct_option_is_split_evenly_between_the_letters():
    """A position bias must not be scoreable as picture reading."""
    spec = _spec([("apple", "red", 2), ("candle", "white", 1)])
    detections = _det(("apple", "red", 60), ("apple", "red", 140),
                      ("bottle", "green", 300))
    golds = [build_absence(spec, detections, random.Random(s), 0).gold
             for s in range(400)]
    assert 0.4 < golds.count("A") / len(golds) < 0.6


def test_relations_survive_the_spec_round_trip():
    """build_spatial reads spec.relations, so a dropped field would silently
    turn every spatial trial non-diagnostic rather than fail."""
    spec = _spec([("mug", "white", 1), ("apple", "red", 1), ("book", "blue", 1)],
                 relations=[("mug", "left_of", "apple")])
    assert spec.relations == (SpecRelation("mug", "left_of", "apple"),)
    assert SceneSpec.from_dict(spec.to_dict()).relations == spec.relations
