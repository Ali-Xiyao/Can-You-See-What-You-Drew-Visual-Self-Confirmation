"""Image-correctness verification for verifiable natural scenes (v4).

Exactly one step in the v4 pipeline is judged by a model: reading the pixels.
Everything downstream -- comparing multisets, looking the gold answer up, marking
an A/B reply right or wrong -- is deterministic arithmetic and stays that way.
Concentrating the uncertainty in one place is the point, because that one place
can then be audited, and STATUS 11 is the record of what happens when it is not:
the geometric detector it replaced agreed with human labels on only 70.8% of
verdicts while every number in 10 rested on it.

Measured on 72 blind human-labelled images:

    verifier                precision  recall     F1   exact-set
    geometric (v3)              0.750   0.757  0.754       0.486
    Qwen3-VL-8B-Instruct        0.981   0.990  0.986       0.944
    InternVL3.5-8B              0.985   0.971  0.978       0.903

Two models, one escalation ladder
---------------------------------
The two are read together rather than one being taken on trust, but agreement is
used as a *flag*, not as a filter. Filtering to the agreeing subset looks
harmless and is not: the images the two disagree on are the occluded, blurred,
ambiguous ones, so dropping them quietly resamples the corpus toward easy scenes,
inflates p, and -- because clutter is not evenly distributed across families --
compares different populations family to family. Those hard images are also the
only place a selection experiment has any headroom.

So every image is resolved instead:

    1. the two report the same object list -> accept
    2. the lists differ but the verdict is
       the same either way                 -> accept the verdict; ask no
                                              question about a disputed object
    3. the verdict itself differs          -> re-query just the disputed object,
                                              cropped and enlarged
    4. still differs                       -> human adjudication

Level 2 is the one that makes the ladder affordable, and it is not a relaxation.
What this verifier outputs is `image_correct`, and on a 468-image run the two
models produced the same object list on 49.6% of images but the same *verdict*
on 86.9%. The gap is disagreements that cannot change the answer: asked for two
blue mugs, one model sees three mugs and a saucer and the other sees three mugs,
and the image is wrong on either reading. Sending those to a person buys nothing
about the measured quantity -- it spent 121 of 157 adjudications on images whose
verdict was never in doubt.

Both agreement numbers are reported, because they answer different questions:
0.496 is how well two strong detectors agree about what is in a generated image,
which is a fact about these images worth stating; 0.869 is how reliable the gold
standard is for what it is used for.

Where the verdict agrees but the lists do not, the settled list is the agreed
core -- the objects both models saw -- and the disputed object is dropped rather
than attributed to either. Questions are then never asked about an object whose
existence two strong detectors disagree on, which would score the model against a
coin flip. The verdict is *not* recomputed from that core: two models can call an
image wrong for different reasons, and the intersection of their lists can match
the spec when neither of them did.

Level 3 works because the remaining disagreements are about one object, not the
whole image. Cropping to it supplies more pixels, which is new information; a
third model would only be another correlated vote.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from selfsight.v4.spec import (
    SceneSpec,
    canonical_noun,
    detected_multiset,
    image_correct,
    match_report,
)


class Resolution(str, Enum):
    """How an image's object list was settled."""

    AGREED = "agreed"
    AGREED_VERDICT = "agreed_verdict"
    RESOLVED_BY_CROP = "resolved_by_crop"
    HUMAN = "human"
    PENDING_HUMAN = "pending_human"


@dataclass(frozen=True)
class VerificationResult:
    image_path: str
    spec_id: str
    detections: tuple[dict[str, Any], ...]
    image_correct: bool
    resolution: Resolution
    verifier_agreement: bool
    disputed: tuple[dict[str, Any], ...] = ()
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "spec_id": self.spec_id,
            "detections": [dict(d) for d in self.detections],
            "image_correct": self.image_correct,
            "resolution": self.resolution.value,
            "verifier_agreement": self.verifier_agreement,
            "disputed": [dict(d) for d in self.disputed],
            "report": dict(self.report),
        }


class Detector(Protocol):
    """A frozen external observer that reports objects from pixels alone.

    It never sees the prompt, the spec, or the gold answer -- only the image.
    """

    detector_id: str

    def detect(self, image_path: str) -> list[dict[str, Any]]: ...

    def detect_crop(
        self, image_path: str, bbox: tuple[float, float, float, float]
    ) -> list[dict[str, Any]]: ...


def _key(item: dict[str, Any]) -> tuple[str, str | None]:
    colour = item.get("color")
    return (
        canonical_noun(item["object"]),
        str(colour).strip().lower() if colour else None,
    )


def disputed_keys(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[tuple[str, str | None]]:
    """The (object, colour) keys the two detectors do not agree on."""
    a, b = detected_multiset(primary), detected_multiset(secondary)
    return sorted(set((a - b) | (b - a)), key=str)


def _bbox_of(items: list[dict[str, Any]], key: tuple[str, str | None]) -> Any:
    for item in items:
        if _key(item) == key:
            box = item.get("bbox") or item.get("box")
            if box and len(box) == 4:
                return tuple(float(v) for v in box)
    return None


def _pad(bbox: tuple[float, ...], factor: float = 0.35) -> tuple[float, ...]:
    """Widen a box before cropping, so context around the object survives."""
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * factor, (y1 - y0) * factor
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def verify(
    image_path: str,
    spec: SceneSpec,
    primary: Detector,
    secondary: Detector | None = None,
    *,
    human_labels: dict[str, list[dict[str, Any]]] | None = None,
) -> VerificationResult:
    """Settle an image's object list, then score it against the spec.

    With no secondary detector this degrades to single-model scoring and reports
    `verifier_agreement=True` vacuously, which keeps the ladder optional for
    cheap sweeps without silently pretending two models were consulted.
    """
    detections = primary.detect(image_path)

    if human_labels is not None and image_path in human_labels:
        settled = human_labels[image_path]
        return VerificationResult(
            image_path=image_path,
            spec_id=spec.spec_id,
            detections=tuple(settled),
            image_correct=image_correct(spec, settled),
            resolution=Resolution.HUMAN,
            verifier_agreement=False,
            report=match_report(spec, settled),
        )

    if secondary is None:
        return VerificationResult(
            image_path=image_path,
            spec_id=spec.spec_id,
            detections=tuple(detections),
            image_correct=image_correct(spec, detections),
            resolution=Resolution.AGREED,
            verifier_agreement=True,
            report=match_report(spec, detections),
        )

    other = secondary.detect(image_path)
    disputes = disputed_keys(detections, other)
    if not disputes:
        return VerificationResult(
            image_path=image_path,
            spec_id=spec.spec_id,
            detections=tuple(detections),
            image_correct=image_correct(spec, detections),
            resolution=Resolution.AGREED,
            verifier_agreement=True,
            report=match_report(spec, detections),
        )

    # Level 2: the lists differ, but do they differ in a way that changes the
    # answer? If both readings give the same verdict, the verdict stands and no
    # human is needed. The settled list is the agreed core, so nothing downstream
    # asks a question about an object only one model saw; the verdict is taken
    # from the two models rather than recomputed from that core, because two
    # models can call an image wrong for different reasons and the intersection
    # of their lists can then match the spec when neither of them did.
    verdict = image_correct(spec, detections)
    if verdict == image_correct(spec, other):
        core = [item for item in detections if _key(item) not in set(disputes)]
        return VerificationResult(
            image_path=image_path,
            spec_id=spec.spec_id,
            detections=tuple(core),
            image_correct=verdict,
            resolution=Resolution.AGREED_VERDICT,
            verifier_agreement=False,
            disputed=tuple({"object": k[0], "color": k[1]} for k in disputes),
            report=match_report(spec, detections),
        )

    # Level 3: re-ask about the disputed object only, with more pixels on it.
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for key in disputes:
        bbox = _bbox_of(detections, key) or _bbox_of(other, key)
        if bbox is None:
            unresolved.append({"object": key[0], "color": key[1], "reason": "no_bbox"})
            continue
        recheck = primary.detect_crop(image_path, _pad(bbox))
        hits = [item for item in recheck if _key(item) == key]
        if hits:
            resolved.append(hits[0])
        else:
            unresolved.append({"object": key[0], "color": key[1],
                               "reason": "not_confirmed_in_crop"})

    if unresolved:
        # Level 4. The list is left as the primary's so downstream code has
        # something well formed, but the row is flagged and must not enter the
        # main analysis until a human has settled it.
        return VerificationResult(
            image_path=image_path,
            spec_id=spec.spec_id,
            detections=tuple(detections),
            image_correct=image_correct(spec, detections),
            resolution=Resolution.PENDING_HUMAN,
            verifier_agreement=False,
            disputed=tuple(unresolved),
            report=match_report(spec, detections),
        )

    # Every dispute was settled by the crop: keep the agreed core plus what the
    # crop confirmed.
    agreed_core = [
        item for item in detections if _key(item) not in set(disputes)
    ]
    settled = agreed_core + resolved
    return VerificationResult(
        image_path=image_path,
        spec_id=spec.spec_id,
        detections=tuple(settled),
        image_correct=image_correct(spec, settled),
        resolution=Resolution.RESOLVED_BY_CROP,
        verifier_agreement=False,
        disputed=tuple({"object": k[0], "color": k[1]} for k in disputes),
        report=match_report(spec, settled),
    )


def ladder_summary(results: list[VerificationResult]) -> dict[str, Any]:
    """The coverage line the paper reports for the gold standard.

    Reporting how each image was settled is a strength, not a caveat: it says the
    labels were managed rather than taken raw from one model.
    """
    counts = collections.Counter(r.resolution.value for r in results)
    n = len(results) or 1
    return {
        "n": len(results),
        "by_resolution": dict(counts),
        # Two agreement numbers, reported separately on purpose. The first is
        # how often the two detectors returned the same object list; the second
        # is how often they reached the same verdict, which is what the gold
        # standard is actually used for.
        "list_agreement_rate": counts[Resolution.AGREED.value] / n,
        "verdict_agreement_rate": (
            counts[Resolution.AGREED.value] + counts[Resolution.AGREED_VERDICT.value]
        ) / n,
        "agreement_rate": counts[Resolution.AGREED.value] / n,
        "crop_resolved_rate": counts[Resolution.RESOLVED_BY_CROP.value] / n,
        "human_rate": (
            counts[Resolution.HUMAN.value] + counts[Resolution.PENDING_HUMAN.value]
        ) / n,
        "pending_human": counts[Resolution.PENDING_HUMAN.value],
        "p": sum(r.image_correct for r in results) / n,
    }
