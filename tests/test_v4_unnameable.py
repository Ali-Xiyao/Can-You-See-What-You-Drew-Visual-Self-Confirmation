"""What happens to an image whose contents cannot be named.

The reviewer reported two failure modes the ladder had no word for: a generator
that fuses two objects into one body (nothing in the frame is a mug, and nothing
is a pear either), and a picture that cannot be read at all. Both used to be
forced into a noun, which put a fabricated object into the gold list.

They are not the same case and must not collapse into one. `unnameable` names one
bad object among readable ones: the verdict survives, because whatever that thing
is it is not what the spec asked for, so the image is a miss and belongs in p.
`unusable` says nothing in the frame can be read, so there is no verdict to have
and the image leaves the denominator instead of counting as a miss.

Both are barred from supplying trials. A question needs a defensible answer, and
"how many pears did you draw" has none when one candidate is half a pear.
"""

from __future__ import annotations

from typing import Any

from selfsight.v4.spec import (SceneSpec, has_unnameable, image_correct,
                               is_unusable)
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


def test_unusable_is_a_separate_word_from_unnameable():
    """The stronger claim must not be reachable by writing the weaker one."""
    assert is_unusable([{"object": "unusable"}])
    assert not is_unusable([{"object": "unnameable"}])
    assert not has_unnameable([{"object": "unusable"}])


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


def test_a_human_label_of_unusable_leaves_the_image_without_a_verdict():
    spec = _spec([("mug", "blue", 2)])
    detector = _Fixed("a", [_obj("mug", "blue", 10)])
    result = verify("img.png", spec, detector, _Fixed("b", []),
                    human_labels={"img.png": [{"object": "unusable", "color": ""}]})
    assert result.resolution is Resolution.UNUSABLE
    assert result.image_correct is None
    assert result.report == {}


def test_a_human_label_of_unnameable_still_produces_a_verdict():
    """Contrast with the case above: this image is scored, and scored wrong."""
    spec = _spec([("mug", "blue", 2)])
    labels = {"img.png": [_obj("mug", "blue", 10), {"object": "unnameable"}]}
    result = verify("img.png", spec, _Fixed("a", []), _Fixed("b", []),
                    human_labels=labels)
    assert result.resolution is Resolution.HUMAN
    assert result.image_correct is False
    assert has_unnameable(list(result.detections))


def test_p_is_computed_over_the_images_that_have_a_verdict():
    """One unusable image among three must not be counted as a failure.

    Two of the three scored images are correct, so p is 1/2 over the verdicts and
    would be 1/3 if the unusable one were folded in as wrong -- a 17-point move
    invented by a picture nobody could read.
    """
    spec = _spec([("mug", "blue", 1)])
    good = _Fixed("a", [_obj("mug", "blue")])
    bad = _Fixed("a", [_obj("pear", "green")])
    results = [
        verify("a.png", spec, good, _Fixed("b", [_obj("mug", "blue")])),
        verify("b.png", spec, bad, _Fixed("b", [_obj("pear", "green")])),
        verify("c.png", spec, good, _Fixed("b", []),
               human_labels={"c.png": [{"object": "unusable"}]}),
    ]
    summary = ladder_summary(results)
    assert summary["n"] == 3
    assert summary["n_with_verdict"] == 2
    assert summary["unusable"] == 1
    assert summary["p"] == 0.5
