"""What happens to an image whose contents cannot be named.

The reviewer reported two failure modes the ladder had no word for: a generator
that fuses two objects into one body (nothing in the frame is a mug, and nothing
is a pear either), and a picture that cannot be read at all. Both used to be
forced into a noun, which put a fabricated object into the gold list.

Both are the same case at different scales and use one word. An unreadable image
briefly had its own -- `unusable` -- which took it out of p's denominator instead
of counting it as a miss. These tests pin down that it does not: p is the share
of generations that drew what was asked, an unreadable one demonstrably did not,
and exempting it deletes the generator's worst output from its own score.

What such an image cannot do is supply a trial. A question needs a defensible
answer, and "how many pears did you draw" has none when one candidate is half a
pear. So: inside p, outside the trials, and reported on its own line.
"""

from __future__ import annotations

from typing import Any

from selfsight.v4.spec import SceneSpec, has_unnameable, image_correct
from selfsight.v4.verifier import Resolution, ladder_summary, verify


class _Fixed:
    def __init__(self, detector_id: str, detections: list[dict[str, Any]]):
        self.detector_id = detector_id
        self._detections = detections

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return [dict(d) for d in self._detections]

    def detect_crop(self, image_path: str, bbox: Any) -> list[dict[str, Any]]:
        return []


def _spec(objects, spec_id="s1"):
    return SceneSpec.from_dict({
        "spec_id": spec_id,
        "prompt": "a scene",
        "objects": [{"object": o, "color": c, "count": n} for o, c, n in objects],
    })


def _obj(noun, colour, x=100.0):
    return {"object": noun, "color": colour,
            "bbox": [x, 100.0, x + 50, 150.0], "center": [x + 25, 125.0]}


def test_an_unnameable_object_is_detected_by_its_noun():
    assert has_unnameable([_obj("mug", "blue"), {"object": "unnameable"}])
    assert not has_unnameable([_obj("mug", "blue")])
    assert not has_unnameable([])


def test_unnameable_survives_the_plural_and_case_folding_in_canonical_noun():
    """Reviewers type quickly. "Unnameable" and "unnameables" are the same word,
    and the noun canonicaliser is what the flag is built on."""
    assert has_unnameable([{"object": "Unnameable"}])
    assert has_unnameable([{"object": "unnameables"}])


def test_a_whole_image_that_cannot_be_read_is_the_same_word():
    """The reviewer writes the one word and enumerates nothing else.

    A list that is only `unnameable` says "there is something here and it is not
    any object", which is exactly the claim, and it needs no second vocabulary.
    """
    assert has_unnameable([{"object": "unnameable", "color": ""}])


def test_an_unnameable_object_makes_the_image_a_miss_rather_than_a_match():
    """It counts as an object, not as empty space.

    Left out of the list, a fused mug-and-pear would score the image as if that
    region held nothing, and a one-mug spec drawn as one fused blob would come
    out as a match.
    """
    spec = _spec([("mug", "blue", 1)])
    assert image_correct(spec, [_obj("mug", "blue")]) is True
    assert image_correct(spec, [{"object": "unnameable", "color": ""}]) is False
    assert image_correct(
        spec, [_obj("mug", "blue"), {"object": "unnameable", "color": ""}]) is False


def test_an_unreadable_image_is_scored_a_miss_not_excused():
    """The case the user pushed back on: this is a generation error too.

    Taking it out of the denominator would have raised p by removing exactly the
    generations that failed hardest.
    """
    spec = _spec([("mug", "blue", 2)])
    detector = _Fixed("a", [_obj("mug", "blue", 10)])
    result = verify("img.png", spec, detector, _Fixed("b", []),
                    human_labels={"img.png": [{"object": "unnameable", "color": ""}]})
    assert result.resolution is Resolution.HUMAN
    assert result.image_correct is False


def test_a_human_label_of_unnameable_still_produces_a_verdict():
    """Contrast with the case above: this image is scored, and scored wrong."""
    spec = _spec([("mug", "blue", 2)])
    labels = {"img.png": [_obj("mug", "blue", 10), {"object": "unnameable"}]}
    result = verify("img.png", spec, _Fixed("a", []), _Fixed("b", []),
                    human_labels=labels)
    assert result.resolution is Resolution.HUMAN
    assert result.image_correct is False
    assert has_unnameable(list(result.detections))


def test_p_counts_the_unreadable_image_as_a_failure_and_reports_it_separately():
    """One correct, one wrong objects, one unreadable: p is 1/3, not 1/2.

    Both failures are in p because both are the generator missing its spec, and
    the unnameable count is on its own line because "drew the wrong objects" and
    "drew non-objects" are different failures that p sums into one number.
    """
    spec = _spec([("mug", "blue", 1)])
    good = _Fixed("a", [_obj("mug", "blue")])
    bad = _Fixed("a", [_obj("pear", "green")])
    results = [
        verify("a.png", spec, good, _Fixed("b", [_obj("mug", "blue")])),
        verify("b.png", spec, bad, _Fixed("b", [_obj("pear", "green")])),
        verify("c.png", spec, good, _Fixed("b", []),
               human_labels={"c.png": [{"object": "unnameable"}]}),
    ]
    summary = ladder_summary(results)
    assert summary["n"] == 3
    assert summary["unnameable"] == 1
    assert summary["p"] == 1 / 3
