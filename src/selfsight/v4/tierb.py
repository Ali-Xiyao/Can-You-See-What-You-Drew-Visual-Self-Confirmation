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
from typing import Any, Callable, Iterable, Sequence

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

ACHROMATIC: dict[str, tuple[float, float, float]] = {
    # colour: (max saturation, min value, max value)
    "white": (0.18, 0.70, 1.01),
    "silver": (0.16, 0.45, 0.92),
    "grey": (0.16, 0.22, 0.72),
    "gray": (0.16, 0.22, 0.72),
    "black": (0.45, 0.00, 0.22),
}
"""Objects with no hue, segmented by saturation and value instead.

`COLOUR_HUES` cannot describe these and the first deletion pass fell back to
protecting a neighbour's whole bounding box whenever it met one. That box
overlaps the target in a still life, so it carved a hole out of the deletion and
left a stub of the removed object standing in it -- half a red bottle on the
white block it stood on, a green ghost between two black buckets. Segmenting
them properly is what makes the fallback rare instead of routine.

Black is allowed more saturation than the others because a black object in a
coloured scene picks up the surround; at 0.16 it lost its own edges.
"""

DELETE_BOX_PAD = 0.12
"""How far the deletion box reaches past the detection, as a share of its side.

Wider than it looks like it needs to be, because parts of an object routinely
fall outside its detection box: a candle's flame, a bottle's neck highlight, the
shadow it casts. At 0.05 those survived the deletion and left a flame burning in
mid-air. The neighbours are protected by pixel mask, so widening the box costs
coverage against them rather than safety.
"""

DELETE_PROTECT_GROW = 2
"""Pixels a neighbour's mask grows before it is subtracted from the target's box.

Covers the neighbour's antialiased edge. Without it the deletion nicks a one
pixel outline off whatever it was standing next to.
"""

MAX_DELETE_SHARE = 0.20
"""Largest share of the frame the hole may cover.

Not a taste threshold -- it is where the filler stops working. These are close
up still lifes and one object routinely owns a third of the frame; the eight
pilot deletions came out clean at 0.16 and 0.17, smeared at 0.67, and at 0.90
the inpainter replaced a dinner plate with a dark wall and left the spoons
floating on it. Cropping a window around the hole was tried first and changed
nothing, because at these hole sizes the window is the whole image anyway.

At 0.20, 292 of 690 candidates survive across 188 images, which is supply
enough. The cap is applied before the detectors so the GPU is not spent on
edits already known to be unusable.
"""

MIN_DELETE_CENTRE = 0.80
"""How much of the target's own box the deletion must reach.

Low means a neighbour's protected pixels sit in the middle of the target, so
the target cannot be removed without taking the neighbour with it -- a spoon
lying on the plate being deleted. Rejecting is right: the alternative is an
image with half an object still in it.
"""

DELETE_GROW = 3
"""Pixels the deletion mask grows past the object.

The antialiased rim and the contact shadow are not the object's own colour, so
they survive a pixel-exact mask and read as a halo where something used to be --
the most visible tell that the image was edited, and edit artifacts are one of
the three explanations this experiment exists to separate.
"""


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


@dataclass(frozen=True)
class DeletePlan:
    image_path: str
    noun: str
    colour: str
    bbox: tuple[float, float, float, float]
    sham: bool = False
    """A null edit: the same filler over the same area of background, with the
    object list left alone. Carries the artifact without the fact."""


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


def object_mask(
    image: np.ndarray,
    bbox: Sequence[float],
    colour: str,
) -> tuple[np.ndarray, float, float]:
    """The object's own pixels inside `bbox`, not the box.

    Returns a full-image boolean mask, the share of the detection box it covers,
    and the share of that box's centre. Both edits that need to know where an
    object *is* -- rotating its hue, and removing it -- go through here, so the
    two cannot drift apart in what they consider "the object".

    Extracted from `recolour`, which is why the padding logic reads the way it
    does: the mask is built on a box padded beyond the detection so that the
    background, which continues past the object, reaches the patch border and
    can be told apart from the object, which cannot. See `_object_components`.
    """
    if colour not in COLOUR_HUES and colour not in ACHROMATIC:
        raise ValueError(f"not a segmentable colour: {colour}")
    height, width = image.shape[:2]
    bx0, by0, bx1, by1 = (int(round(v)) for v in bbox)
    pad_x = max(4, int(PAD_FRACTION * (bx1 - bx0)))
    pad_y = max(4, int(PAD_FRACTION * (by1 - by0)))
    x0, y0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    x1, y1 = min(width, bx1 + pad_x), min(height, by1 + pad_y)
    empty = np.zeros((height, width), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return empty, 0.0, 0.0

    patch = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    hsv = _hsv_array(patch)
    if colour in COLOUR_HUES:
        hue = COLOUR_HUES[colour]
        mask = (
            (np.abs(((hsv[..., 0] - hue + 0.5) % 1.0) - 0.5) <= HUE_WINDOW)
            & (hsv[..., 1] >= MIN_SATURATION)
            & (hsv[..., 2] >= MIN_VALUE)
        )
    else:
        max_saturation, min_value, max_value = ACHROMATIC[colour]
        mask = (
            (hsv[..., 1] <= max_saturation)
            & (hsv[..., 2] >= min_value)
            & (hsv[..., 2] <= max_value)
        )
    if not mask.any():
        return empty, 0.0, 0.0
    mask = _object_components(mask)
    if not mask.any():
        return empty, 0.0, 0.0
    # Reported against the detection box, not the padded patch, so the numbers
    # mean the same thing they did before the padding was introduced.
    inner = mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    share = float(inner.mean()) if inner.size else 0.0
    centre = centre_coverage(inner) if inner.size else 0.0
    full = empty.copy()
    full[y0:y1, x0:x1] = mask
    return full, share, centre


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
    if target_colour not in COLOUR_HUES:
        raise ValueError(f"not a rotatable colour: {source_colour}->{target_colour}")
    mask, share, centre = object_mask(image, bbox, source_colour)
    if share == 0.0:
        return image, 0.0, 0.0

    height, width = image.shape[:2]
    delta = (COLOUR_HUES[target_colour] - COLOUR_HUES[source_colour]) % 1.0
    hsv = _hsv_array(image.astype(np.float32) / 255.0)
    hsv[..., 0] = np.where(mask, (hsv[..., 0] + delta) % 1.0, hsv[..., 0])
    edited = np.clip(_rgb_array(hsv) * 255.0, 0, 255).astype(np.uint8)
    out = np.where(mask[..., None], edited, image)
    return out.astype(np.uint8), share, centre


def deletion_mask(
    image: np.ndarray,
    bbox: Sequence[float],
    colour: str,
    others: Iterable[tuple[Sequence[float], str]] = (),
    box_pad: float = DELETE_BOX_PAD,
    protect_grow: int = DELETE_PROTECT_GROW,
) -> tuple[np.ndarray, float]:
    """Everything inside the target's box except the neighbours' own pixels.

    Two masks were tried first and both failed, in opposite directions, on the
    same eight pilot images -- these are compact still lifes and the objects
    touch:

    * The **bounding box** takes the neighbour with it. A bottle came out sliced
      flat and a red book vanished along with the candle beside it, because the
      candle's box covered them. Four of eight edits were unusable.
    * The **object's own colour mask** (`object_mask`) leaves the object's
      remains behind: the stem, the specular highlight, the contact shadow and
      any second-coloured part are not the body's hue, so an apple was removed
      and left a white blob with a stem floating over it, and a carrot was
      removed from under its own green top.

    The union of the two failure sets is most of the corpus, so neither is a
    filter that could be tightened. What is wanted is the box -- so nothing of
    the target survives -- minus whatever inside it belongs to something else,
    which is exactly what `object_mask` finds for the neighbours. The target's
    own pixels are added back afterwards in case it shares a hue with a
    neighbour whose mask spilled onto it.

    Returns the mask and the share of the box's centre it covers. A low centre
    share means a neighbour sits in the middle of the target's box and the
    deletion cannot be made cleanly; the caller rejects on it.
    """
    from scipy import ndimage

    height, width = image.shape[:2]
    bx0, by0, bx1, by1 = (int(round(v)) for v in bbox)
    pad_x = max(3, int(box_pad * (bx1 - bx0)))
    pad_y = max(3, int(box_pad * (by1 - by0)))
    x0, y0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    x1, y1 = min(width, bx1 + pad_x), min(height, by1 + pad_y)
    box = np.zeros((height, width), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return box, 0.0
    box[y0:y1, x0:x1] = True

    protect = np.zeros((height, width), dtype=bool)
    for other_bbox, other_colour in others:
        mask, share = np.zeros((height, width), dtype=bool), 0.0
        if other_colour in COLOUR_HUES or other_colour in ACHROMATIC:
            mask, share, _ = object_mask(image, other_bbox, other_colour)
        if share > 0.0:
            protect |= mask
        else:
            # Either the colour is not one this module can segment at all, or
            # `object_mask` refused, which it does when the neighbour reaches
            # the border of its own padded patch. Refusing is right for editing
            # an object and wrong for protecting one: a banana lying across a
            # plate got no mask and was deleted along with the plate. So fall
            # back to the neighbour's whole box, which costs deletion coverage
            # and is rejected later on centre coverage. Losing a candidate is
            # the safe error; losing a bystander is not.
            ox0, oy0, ox1, oy1 = (int(round(v)) for v in other_bbox)
            protect[max(0, oy0):oy1, max(0, ox0):ox1] = True
    if protect.any() and protect_grow > 0:
        protect = ndimage.binary_dilation(protect, iterations=protect_grow)

    # The target's own pixels are added back in case a neighbour's mask spilled
    # onto them.
    if colour in COLOUR_HUES or colour in ACHROMATIC:
        own, own_share, _ = object_mask(image, bbox, colour)
    else:
        own = np.zeros((height, width), dtype=bool)
    mask = (box & ~protect) | (own & box)
    inner = mask[by0:by1, bx0:bx1]
    centre = centre_coverage(inner) if inner.size else 0.0
    return mask, centre


def delete(
    image: np.ndarray,
    bbox: Sequence[float],
    colour: str,
    inpaint: Callable[[np.ndarray, np.ndarray], np.ndarray],
    others: Iterable[tuple[Sequence[float], str]] = (),
    grow: int = DELETE_GROW,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Remove the object at `bbox` and let `inpaint` fill what it stood on.

    `inpaint` is passed in rather than imported so this module keeps no opinion
    about which filler is used and needs no runtime for it; the acceptance gate,
    not the filler's reputation, is what licenses the label.

    Returns the edited image, the mask, and the mask's centre coverage.
    """
    from scipy import ndimage

    mask, centre = deletion_mask(image, bbox, colour, others)
    if not mask.any():
        return image, mask, 0.0
    if grow > 0:
        # Only outward, and only where nothing else was protected: the rim and
        # the contact shadow read as a halo where something used to be, which is
        # the most visible tell that the image was edited -- and edit artifacts
        # are one of the three explanations this experiment exists to separate.
        mask = ndimage.binary_dilation(mask, iterations=grow)
    filled = inpaint(image, mask)
    # Only masked pixels move. The rest of the image is bit-identical, so
    # "nothing else changed" is true by construction and the detectors are
    # asked only whether the object is gone.
    out = np.where(mask[..., None], filled, image)
    return out.astype(np.uint8), mask, centre


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


def delete_question(noun: str, colour: str, gold_first: bool) -> dict[str, Any]:
    """The existence 2AFC for a deletion pair.

    The natural pool words existence as a contrast between two nouns, one drawn
    and one not. That form cannot carry a deletion pair: the deleted noun is
    present in one member and absent in the other, so whichever second noun is
    offered, one of the two members has either both options present or neither.

    A yes/no about the deleted object is the form that stays answerable on both
    members and changes its answer on exactly the fact that was edited. Yes/no
    invites acquiescence, and the wording does nothing to prevent it -- what
    does is that the comparison is *paired* and within-image: a constant lean
    toward "yes" shifts both members equally and cancels in the difference.
    Which of yes and no is option A is randomised per pair as well, so the lean
    cannot align with a position preference.
    """
    truth, lie = "yes", "no"
    option_a, option_b = (truth, lie) if gold_first else (lie, truth)
    # Agrees with whatever word actually comes next, which is the colour.
    article = "an" if (colour or noun)[:1] in "aeiou" else "a"
    return {
        "family": "existence",
        "question": (
            f"Is there {article} {colour} {noun} in this picture? "
            f"Answer A or B only.\nA. {option_a}\nB. {option_b}"
        ),
        "option_a": option_a,
        "option_b": option_b,
        "gold_original": "A" if gold_first else "B",
        "gold_edited": "B" if gold_first else "A",
    }


def sham_box(
    shape: tuple[int, int],
    boxes: Iterable[Sequence[float]],
    area: float,
    rng: Any,
    pad: float = 0.06,
    tries: int = 400,
) -> tuple[int, int, int, int] | None:
    """A square of `area` pixels on the background, touching no detection.

    The null edit. It runs the same filler over the same amount of image and
    leaves the object list alone, so whatever the inpainter does to an image --
    the softened texture, the seam, the faint halo -- appears on both arms while
    only the deletion arm changes a fact. Without it a difference between the
    members of a deletion pair could be the missing object or could be that one
    member has been through a network and the other has not, and those are two
    of the three explanations this experiment exists to separate.

    Matched on area rather than on shape, because area is what governs how badly
    the filler struggles; matching the outline as well would require putting the
    hole where the object was, which is the thing being controlled for.

    A square is tried first and then progressively longer rectangles of the same
    area. These are tight still lifes and the clear background is usually a band
    above or beside the objects, not a patch: insisting on a square lost the
    control on two thirds of the images, which would have left the sham arm too
    small to rule anything out. The shapes are tried most-square first, so the
    stretched ones are used only where nothing rounder fits.
    """
    height, width = shape
    blocked = []
    for box in boxes:
        bx0, by0, bx1, by1 = (float(v) for v in box)
        px, py = pad * (bx1 - bx0), pad * (by1 - by0)
        blocked.append((bx0 - px, by0 - py, bx1 + px, by1 + py))

    for ratio in (1.0, 1.5, 2.0, 3.0, 4.0):
        for long_side_is_x in (True, False):
            long_side = int(round((area * ratio) ** 0.5))
            short_side = max(1, int(round(area / max(1, long_side))))
            w, h = ((long_side, short_side) if long_side_is_x
                    else (short_side, long_side))
            if min(w, h) < 8 or w >= width or h >= height:
                continue
            for _ in range(tries):
                x0 = rng.randrange(0, width - w)
                y0 = rng.randrange(0, height - h)
                x1, y1 = x0 + w, y0 + h
                if all(x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1
                       for bx0, by0, bx1, by1 in blocked):
                    return x0, y0, x1, y1
            if ratio == 1.0:
                break
    return None


def delete_accepted(
    plan: "DeletePlan",
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Did the deletion remove exactly the one object and nothing else?

    The failure the filler actually commits is not leaving the object behind --
    the mask is the whole box, so it cannot -- but inventing a replacement. In
    the pilot a red mug came out as a beige egg and a white bowl beside the
    target lost its right half. Both are collateral changes and both are caught
    here, which is why the editor is allowed to be crude.

    For a sham the expectation is the opposite: the list must not move at all.
    """
    from selfsight.v4.spec import canonical_noun, countable

    def pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return sorted(
            (canonical_noun(row.get("object", "")),
             str(row.get("color", "")).strip().lower())
            for row in countable(rows)
        )

    got_before, got_after = pairs(before), pairs(after)
    target = (plan.noun, plan.colour)
    if plan.sham:
        if got_after == got_before:
            return True, "ok"
        return False, "sham_changed_the_list"
    if target not in got_before:
        return False, "target_absent_before"
    expected = sorted(got_before)
    expected.remove(target)
    if got_after == got_before:
        return False, "unchanged"
    if target in got_after:
        return False, "target_still_there"
    if got_after == expected:
        return True, "ok"
    return False, "collateral_change"


def delete_confirmed(plan: "DeletePlan", after: list[dict[str, Any]]) -> bool:
    """Did this detector see the edited fact alone? See `edit_confirmed`."""
    from selfsight.v4.spec import canonical_noun, countable

    seen = {
        (canonical_noun(row.get("object", "")),
         str(row.get("color", "")).strip().lower())
        for row in countable(after)
    }
    target = (plan.noun, plan.colour)
    return target in seen if plan.sham else target not in seen


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


def edit_confirmed(
    plan: RecolourPlan, after: list[dict[str, Any]]
) -> bool:
    """Did this detector see the edited fact, ignoring the rest of the list?

    The strict gate above asks whether the whole list moved by exactly one pair,
    and on the first run it threw away 19 of 80 recolours because the two
    detectors disagreed about some *other* object -- an object neither of them
    was asked about and neither edit touched.

    That is the same mistake the main pipeline already made and corrected: two
    strong detectors disagree about the object list roughly half the time and
    about the verdict rarely, so escalating on the list spends the budget on
    images whose answer was never in doubt (STATUS 19). The fact this gate is
    protecting is "the mug is now red", and that is what this function checks.

    Used together with the strict gate, not instead of it: an edit is a trial
    when both detectors confirm the edited fact AND at least one of them sees
    nothing else changed. Pre-existing disagreement about an untouched object is
    noise; both detectors reporting a second change is not.
    """
    from selfsight.v4.spec import canonical_noun, countable

    seen = {
        (canonical_noun(row.get("object", "")),
         str(row.get("color", "")).strip().lower())
        for row in countable(after)
    }
    return ((plan.noun, plan.target_colour) in seen
            and (plan.noun, plan.source_colour) not in seen)
