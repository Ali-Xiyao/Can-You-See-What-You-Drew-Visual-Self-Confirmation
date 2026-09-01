"""Scene specs for verifiable natural scenes (v4).

A verifiable natural scene is one whose prompt is the natural-language form of an
exact spec: every item can be ticked off against the image by a person in
seconds. That property, not the visual style, is what makes the corpus usable --
it is what lets the verifier itself be audited, which STATUS 11 showed is not
optional (the shipped geometric verifier agreed with human labels only 0.708 of
the time while silently driving every number in 10).

Two changes from v3's geometric scenes:

* Objects are real categories with plausible colours, so the images are ordinary
  photographs rather than shape tests.
* Correctness is one rule for every family -- the detected (object, colour)
  multiset must equal the spec's -- instead of a per-family gold whose minimal
  claims could be satisfied by drawing less. All three v3 families leaked through
  that: 15% / 37.4% / 44.8% of their correct verdicts came from omission.

Absolute size is deliberately absent. SIZE_BOUNDARY_PX = 95 was calibrated on the
reference renderer; in a perspective render an object's pixel extent depends on
its distance from the camera, so a foreground small apple can outspan a
background large bowl. STATUS already excluded absolute size as a family; it
survived inside the gold only by oversight. Relative size within one image stays
well defined and is still available as a question.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

Multiset = collections.Counter


@dataclass(frozen=True)
class SpecObject:
    """One requested group of identical objects."""

    object: str
    color: str | None
    count: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SpecObject:
        colour = value.get("color")
        return cls(
            object=str(value["object"]).strip().lower(),
            color=str(colour).strip().lower() if colour else None,
            count=int(value["count"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"object": self.object, "color": self.color, "count": self.count}


@dataclass(frozen=True)
class SpecRelation:
    """An optional image-plane relation between two spec objects.

    Relations are anchored to the image plane (further left in the picture)
    rather than to a viewer-relative frame, because in a perspective render
    "to its left" has no single referent.
    """

    subject: str
    relation: str
    object: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SpecRelation:
        return cls(
            subject=str(value["subject"]).strip().lower(),
            relation=str(value["relation"]).strip().lower(),
            object=str(value["object"]).strip().lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }


@dataclass(frozen=True)
class SceneSpec:
    """The exact requirement a prompt encodes.

    `objects` is mandatory: it carries the difficulty knob (measured to be the
    dominant failure mode -- 54% of v3 images had the wrong object count, which
    the old per-family gold could not see) and it defines `image_correct`, which
    the whole selection mechanism is built on.

    `relations` is optional. A relation that is present is also required of the
    generator and therefore costs p; a relation that is absent can still be asked
    about, because the gold answer comes from the detected boxes rather than from
    the spec. Keeping some prompts relation-free buys spatial questions for free.
    """

    spec_id: str
    prompt: str
    objects: tuple[SpecObject, ...]
    relations: tuple[SpecRelation, ...] = ()
    surface: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneSpec:
        return cls(
            spec_id=str(value["spec_id"]),
            prompt=str(value["prompt"]),
            objects=tuple(SpecObject.from_dict(o) for o in value["objects"]),
            relations=tuple(
                SpecRelation.from_dict(r) for r in value.get("relations", ())
            ),
            surface=str(value.get("surface", "")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "prompt": self.prompt,
            "objects": [o.to_dict() for o in self.objects],
            "relations": [r.to_dict() for r in self.relations],
            "surface": self.surface,
            "metadata": dict(self.metadata),
        }

    @property
    def n_items(self) -> int:
        """Number of distinct requested groups -- the difficulty knob."""
        return len(self.objects)

    @property
    def n_objects(self) -> int:
        """Total requested objects, counting duplicates."""
        return sum(o.count for o in self.objects)

    def multiset(self) -> Multiset:
        """The (object, colour) multiset this spec requires.

        Colour is part of the key when the spec names one. A spec item with no
        colour contributes an uncoloured key, which only matches a detection
        whose colour was likewise not asserted -- so an unspecified colour is
        never silently treated as correct.
        """
        counter: Multiset = collections.Counter()
        for item in self.objects:
            counter[(item.object, item.color)] += item.count
        return counter


def detected_multiset(detections: list[dict[str, Any]]) -> Multiset:
    """The (object, colour) multiset a verifier reported for an image."""
    counter: Multiset = collections.Counter()
    for item in detections:
        colour = item.get("color")
        counter[
            (
                str(item["object"]).strip().lower(),
                str(colour).strip().lower() if colour else None,
            )
        ] += 1
    return counter


def image_correct(spec: SceneSpec, detections: list[dict[str, Any]]) -> bool:
    """Did the generator draw the spec?

    One rule for every family, and no negative claim anywhere, so the draw-less-
    to-score-higher loophole is impossible by construction rather than patched.
    """
    return spec.multiset() == detected_multiset(detections)


def match_report(spec: SceneSpec, detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Where a mismatch came from, for the capability-floor table.

    The levels are nested, so the drop between two rows is what that attribute
    cost on its own. On v3's corpus this decomposition was the finding: count
    0.459 -> object 0.277 -> colour 0.252, i.e. the object count was the dominant
    failure and colour was nearly free.
    """
    want, got = spec.multiset(), detected_multiset(detections)
    want_objects = collections.Counter(key[0] for key in want.elements())
    got_objects = collections.Counter(key[0] for key in got.elements())
    missing = want - got
    extra = got - want
    return {
        "count_ok": sum(want.values()) == sum(got.values()),
        "objects_ok": want_objects == got_objects,
        "objects_and_color_ok": want == got,
        "missing": [
            {"object": key[0], "color": key[1], "n": n}
            for key, n in sorted(missing.items(), key=lambda kv: str(kv[0]))
        ],
        "extra": [
            {"object": key[0], "color": key[1], "n": n}
            for key, n in sorted(extra.items(), key=lambda kv: str(kv[0]))
        ],
        "n_requested": sum(want.values()),
        "n_detected": sum(got.values()),
    }
