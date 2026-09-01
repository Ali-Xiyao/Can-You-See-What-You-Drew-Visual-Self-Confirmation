"""Tier B: matched image pairs differing in exactly one atomic fact.

The natural pool can only ask "what did you draw" about mistakes the generator
happened to make, and it does not make the mistakes evenly: over 912 images
colour came out right 1495 times against 88 wrong, so the binding family yielded
72 diagnostic trials and spatial yielded none. Waiting for more of them is not a
plan -- the shortage is a property of the generator, and more corpus scales the
count without changing the proportion.

Here the difference is manufactured, so the truth is known from the edit rather
than from a detector, and the yield is whatever we choose it to be.

Two edits, and they do NOT measure the same thing:

`flip`     mirrors the whole image. Every left/right relation reverses exactly,
           with no synthesis and nothing to verify. But the corpus specs carry
           no spatial relations, so both members of a flip pair fit the prompt
           equally well and there is no deference to measure. A flip pair asks
           the more basic question: does the answer track the pixels at all? A
           model reciting the prompt answers both members identically, because
           nothing it is reciting from has changed.

`recolour` rotates the hue of one object inside its detection box. The original
           was externally verified as matching the spec, so after the edit the
           picture contradicts the prompt on exactly one atom and agrees with it
           on everything else. That is a binding diagnostic where "what did it
           actually draw" is not an inference -- we changed it.

Hue rotation rather than a flat fill, because a flat fill destroys the shading,
specular highlights and texture that tell the observer it is looking at a mug.
Rotating H while keeping S and V preserves all of that and changes only the
name of the colour.

The edit is deliberately crude and the acceptance is strict: an edit is kept
only if both detectors report the target in its new colour and every other
(object, colour) unchanged. Building a careful editor and trusting it would put
the burden on code nobody checks; a rough editor behind the same two-detector
gate the main pool uses puts the burden on the same instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

# Reference hues in [0, 1). Brown is orange at low value, so it can be a source
# (rotate away from it) but never a target: no hue rotation alone produces
# brown from a bright colour, it would need V pulled down too, and that is a
# second edit.
COLOUR_HUES: dict[str, float] = {
    "red": 0.000,
    "brown": 0.055,
    "orange": 0.075,
    "yellow": 0.145,
    "green": 0.330,
    "blue": 0.600,
    "purple": 0.790,
    "pink": 0.930,
}
TARGET_COLOURS = frozenset(COLOUR_HUES) - {"brown"}

# Mask thresholds. S and V floors drop the white highlights, the shadow under
# the object and the surface it sits on, all of which share no colour name with
# it. The hue window is wide because a detector's "red" apple spans roughly
# 0.95-0.05 once shading is included.
HUE_WINDOW = 0.075
MIN_SATURATION = 0.20
MIN_VALUE = 0.12
# An edit covering almost none of the box did not recolour the object, and one
# covering almost all of it recoloured the box rather than the object.
MIN_MASK_SHARE = 0.06
MAX_MASK_SHARE = 0.85
# A detection box is tight around its object, so the object owns the middle of
# it. A mask that covers the border and not the middle has selected the
# background showing around the object -- and when the background is a green
# wall behind a green pear, it is exactly as saturated and exactly the right
# hue, so nothing in the hue window or the connected-component step rules it
# out. Two of the first eight sample edits failed this way: the pear was left
# untouched and a rotated rectangle of wall appeared behind it.
MIN_CENTRE_COVERAGE = 0.30
CENTRE_FRACTION = 0.5
PAD_FRACTION = 0.15


@dataclass(frozen=True)
class RecolourPlan:
    image_path: str
    noun: str
    bbox: tuple[float, float, float, float]
    source_colour: str
    target_colour: str
    colour_constrained: bool
    """True when the noun's colour is semantically constrained (a lemon is not
    blue). Recorded rather than excluded: a prompted model answering "green" for
    a recoloured pear could be deferring to the prompt or leaning on a prior
    about pears, and those are separable only if the constrained and free nouns
    are tagged and compared."""


def hue_distance(first: float, second: float) -> float:
    """Circular distance on the hue wheel, where 0.98 and 0.02 are close."""
    gap = abs(first - second) % 1.0
    return min(gap, 1.0 - gap)


def _hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV on a float array in [0, 1]."""
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    span = maximum - minimum
    hue = np.zeros_like(maximum)
    safe = span > 1e-9
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    is_red = safe & (maximum == red)
    is_green = safe & (maximum == green) & ~is_red
    is_blue = safe & ~is_red & ~is_green
    with np.errstate(invalid="ignore", divide="ignore"):
        hue[is_red] = ((green - blue)[is_red] / span[is_red]) % 6.0
        hue[is_green] = ((blue - red)[is_green] / span[is_green]) + 2.0
        hue[is_blue] = ((red - green)[is_blue] / span[is_blue]) + 4.0
    hue /= 6.0
    saturation = np.where(maximum > 1e-9, span / np.maximum(maximum, 1e-9), 0.0)
    return np.stack([hue % 1.0, saturation, maximum], axis=-1)


def _rgb_array(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(hue * 6.0)
    f = hue * 6.0 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - f * saturation)
    t = value * (1.0 - (1.0 - f) * saturation)
    i = (i % 6).astype(int)[..., None]
    out = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [
            np.stack([value, t, p], axis=-1),
            np.stack([q, value, p], axis=-1),
            np.stack([p, value, t], axis=-1),
            np.stack([p, q, value], axis=-1),
            np.stack([t, p, value], axis=-1),
            np.stack([value, p, q], axis=-1),
        ],
    )
    return out


def _object_components(mask: np.ndarray, keep: int = 2) -> np.ndarray:
    """Keep the blobs that are the object, drop the ones that are the scene.

    Two things put same-coloured pixels inside a detection box that do not
    belong to the object: the surface it stands on, and the wall behind it. A
    green pear against a green wall is the hard case -- the wall is inside the
    hue window, is fully saturated, and forms a *bigger* blob than the pear, so
    neither the hue test nor "take the largest" excludes it. Two of the first
    eight sample edits came out with the pear untouched and a rotated rectangle
    of wall behind it.

    What separates them is not size but extent: the mask is computed on a box
    padded beyond the detection, and the object cannot reach the padded border
    because its own box was tight around it. The background can and does. So any
    component touching the padded border is scene, whatever its size.

    Of what remains, the largest two are kept -- two rather than one because a
    handle or a rim is often disconnected from the body -- and a second blob far
    smaller than the first is speckle rather than a handle.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    if count == 0:
        return mask
    border = set(labelled[0, :]) | set(labelled[-1, :])
    border |= set(labelled[:, 0]) | set(labelled[:, -1])
    border.discard(0)
    sizes = ndimage.sum(mask, labelled, range(1, count + 1))
    inner = [index for index in range(count) if (index + 1) not in border]
    if not inner:
        # Everything touches the edge. Either the object runs off the image or
        # the whole box is one colour; neither is an edit we can trust.
        return np.zeros_like(mask)
    inner.sort(key=lambda index: sizes[index], reverse=True)
    chosen = inner[:keep]
    if len(chosen) > 1 and sizes[chosen[1]] < 0.15 * sizes[chosen[0]]:
        chosen = chosen[:1]
    return np.isin(labelled, [index + 1 for index in chosen])


def centre_coverage(mask: np.ndarray, fraction: float = CENTRE_FRACTION) -> float:
    """Share of the box's central region the mask covers.

    Low means the mask found the surround rather than the object.
    """
    height, width = mask.shape
    margin_y = int(height * (1 - fraction) / 2)
    margin_x = int(width * (1 - fraction) / 2)
    centre = mask[margin_y:height - margin_y, margin_x:width - margin_x]
    return float(centre.mean()) if centre.size else 0.0


def recolour(
    image: np.ndarray,
    bbox: Sequence[float],
    source_colour: str,
    target_colour: str,
) -> tuple[np.ndarray, float, float]:
    """Rotate the hue of `source_colour` pixels inside `bbox` to `target_colour`.

    Returns the edited image, the share of the box changed, and the share of the
    box's centre changed. The caller decides whether those are credible; this
    function does not silently refuse, because a refusal that looks like a
    successful edit is the worst outcome available.
    """
    if source_colour not in COLOUR_HUES or target_colour not in COLOUR_HUES:
        raise ValueError(f"not a rotatable colour: {source_colour}->{target_colour}")
    height, width = image.shape[:2]
    bx0, by0, bx1, by1 = (int(round(v)) for v in bbox)
    # The mask is built on a padded box so that the background, which continues
    # past the object's own box, reaches the border of the patch and can be
    # told apart from the object, which cannot.
    pad_x = max(4, int(PAD_FRACTION * (bx1 - bx0)))
    pad_y = max(4, int(PAD_FRACTION * (by1 - by0)))
    x0, y0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    x1, y1 = min(width, bx1 + pad_x), min(height, by1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return image, 0.0, 0.0

    patch = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    hsv = _hsv_array(patch)
    source_hue = COLOUR_HUES[source_colour]
    mask = (
        (np.abs(((hsv[..., 0] - source_hue + 0.5) % 1.0) - 0.5) <= HUE_WINDOW)
        & (hsv[..., 1] >= MIN_SATURATION)
        & (hsv[..., 2] >= MIN_VALUE)
    )
    if not mask.any():
        return image, 0.0, 0.0
    mask = _object_components(mask)
    if not mask.any():
        return image, 0.0, 0.0
    # Reported against the detection box, not the padded patch, so the numbers
    # mean the same thing they did before the padding was introduced.
    inner = mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    share = float(inner.mean()) if inner.size else 0.0
    centre = centre_coverage(inner) if inner.size else 0.0
    if share == 0.0:
        return image, 0.0, 0.0

    delta = (COLOUR_HUES[target_colour] - source_hue) % 1.0
    hsv[..., 0] = np.where(mask, (hsv[..., 0] + delta) % 1.0, hsv[..., 0])
    edited_patch = np.clip(_rgb_array(hsv) * 255.0, 0, 255).astype(np.uint8)
    out = image.copy()
    out[y0:y1, x0:x1] = np.where(mask[..., None], edited_patch, image[y0:y1, x0:x1])
    return out, share, centre


def flip(image: np.ndarray) -> np.ndarray:
    """Mirror left-to-right. Exact, lossless, and nothing to verify."""
    return image[:, ::-1].copy()


def target_colours(
    noun: str,
    source_colour: str,
    other_colours: Iterable[str],
    plausible: Iterable[str] | None,
) -> list[str]:
    """Colours this object could be recoloured to without wrecking the trial.

    Excluded: the colour it already is; any colour another object in the scene
    is wearing, which would make "what colour is the mug" ambiguous to a reader
    who mis-attributes; and anything outside the noun's plausible set when it
    has one, so a recoloured lemon is still a lemon and the model is not being
    asked about an object no photograph could contain.
    """
    allowed = set(TARGET_COLOURS)
    if plausible is not None:
        allowed &= set(plausible)
    allowed -= {source_colour}
    allowed -= set(other_colours)
    # A target within the hue window of the source is not a visible change.
    allowed = {
        colour for colour in allowed
        if hue_distance(COLOUR_HUES[colour], COLOUR_HUES[source_colour])
        > 2 * HUE_WINDOW
    }
    return sorted(allowed)


def recolour_question(
    noun: str, drawn: str, edited: str, count: int, gold_first: bool
) -> dict[str, Any]:
    """The binding 2AFC, worded exactly as the natural pool words it.

    Same wording for both members of the pair and for both conditions, so a gap
    between them cannot be read as a wording effect.
    """
    verb = "is" if count == 1 else "are"
    plural = noun if count == 1 else f"{noun}s"
    option_a, option_b = (drawn, edited) if gold_first else (edited, drawn)
    return {
        "family": "binding",
        "question": (
            f"What colour {verb} the {plural} in this picture? "
            f"Answer A or B only.\nA. {option_a}\nB. {option_b}"
        ),
        "option_a": option_a,
        "option_b": option_b,
        "gold_original": "A" if gold_first else "B",
        "gold_edited": "B" if gold_first else "A",
    }


def flip_question(
    subject: str, other: str, subject_left: bool, gold_first: bool
) -> dict[str, Any]:
    """The spatial 2AFC. Gold on the mirrored member is the opposite, always."""
    truth = "left" if subject_left else "right"
    other_side = "right" if subject_left else "left"
    option_a, option_b = (truth, other_side) if gold_first else (other_side, truth)
    return {
        "family": "spatial",
        "question": (
            f"In this picture, is the {subject} further to the left or further "
            f"to the right than the {other}? Answer A or B only.\n"
            f"A. {option_a}\nB. {option_b}"
        ),
        "option_a": option_a,
        "option_b": option_b,
        "gold_original": "A" if gold_first else "B",
        "gold_edited": "B" if gold_first else "A",
    }


def edit_accepted(
    plan: RecolourPlan,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Did the edit change exactly the one thing it was supposed to change?

    Compares the two detector readings as (noun, colour) multisets. The target
    must have moved to its new colour and every other pair must be untouched.
    Anything else -- an object that lost its colour, a new object appearing
    because the recoloured region now reads as something else, the target
    unchanged because the mask missed it -- is a rejection, and the reason is
    recorded so the failure modes can be counted rather than guessed at.
    """
    from selfsight.v4.spec import canonical_noun, countable

    def pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return sorted(
            (canonical_noun(row.get("object", "")),
             str(row.get("color", "")).strip().lower())
            for row in countable(rows)
        )

    got_before, got_after = pairs(before), pairs(after)
    expected = sorted(
        [(plan.noun, plan.target_colour) if pair == (plan.noun, plan.source_colour)
         else pair for pair in got_before]
    )
    if got_before == got_after:
        return False, "unchanged"
    if (plan.noun, plan.source_colour) not in got_before:
        return False, "target_absent_before"
    if got_after == expected:
        return True, "ok"
    if (plan.noun, plan.target_colour) not in got_after:
        return False, "target_not_recoloured"
    return False, "collateral_change"
