"""LLM-authored scene specs for v4.

v3 built prompts from templates over a fixed vocabulary of shapes and colours.
That gave exact control and reproducibility but produced images nobody would ever
ask a generator for, which is a real external-validity problem: the claim is
about generation, and the stimuli were shape tests.

Here an LLM writes the scenes instead, and the one thing it must do is emit the
prompt and the machine-readable spec *together*. That is not an extra stage, it
is the output format -- without it there is no spec, and without a spec there is
no difficulty knob, no notion of image_correct for the selection mechanism to
operate on, and no way to build the trials where what was drawn differs from what
was asked.

The known risks of letting a model author the corpus are not mitigated with
machinery, they are measured: `distribution_report` reports the realised counts,
colours and item widths so a skew can be seen rather than assumed away. The
prompt asks for a target distribution; whether it was honoured is checked, and
reported, afterwards.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Iterable

from selfsight.v4.spec import SceneSpec, SpecObject

RELATIONS = ("left_of", "right_of")

TOTAL_OBJECTS = 3
"""How many things a scene may contain in total, summed over its entries.

Measured, not chosen. 48 prompts x 4 candidates through Show-o2-1.5B-HQ, scored
by Qwen3-VL-8B under the scene-match rule, gives a cliff rather than a slope:

    objects in scene    n    p       C(p) at K=4
    2                   8    0.500   0.438
    3                  12    0.417   0.468
    4                  52    0.154   0.334
    5                  20    0.000   0
    6 or more         100    0.010   ~0

The selection ceiling C(p) = (1-(1-p)^4) - p needs p roughly in [0.13, 0.70], so
five objects or more leaves the experiment with no headroom at all: every
candidate is wrong and there is nothing to select between. Three is the widest
scene that stays in the band.

The per-entry counts matter separately. Asked for one of something the generator
drew the right number 75.4% of the time; asked for two, 31.2%; asked for three,
21.3%. Counts above two are therefore excluded -- not because a miscount is
uninteresting, but because at three the miss is near-certain and the trial stops
discriminating. A count of two is kept precisely because it misses often enough
to supply the trials where the drawn count differs from the requested one, which
is where `gold_source="image_differs_from_spec"` comes from.
"""

INSTRUCTION = """You are writing test scenes for an image-generation benchmark.

Write {n} scenes. Each scene is a photograph of ordinary objects resting on a
surface. Return ONLY a JSON array; no commentary.

Each element must be exactly:
{{"prompt": "...", "objects": [{{"object": "...", "color": "...", "count": N}}],
  "relations": [], "surface": "..."}}

Hard requirements:
- "prompt" must state every object, its colour and its count, and nothing else
  that could be judged right or wrong. It is the plain-English form of "objects".
- Use concrete, countable, nameable categories. Draw widely from this list and do
  not lean on the first few: apple, banana, lemon, pear, orange, tomato, carrot,
  egg, mug, cup, bowl, plate, spoon, fork, book, notebook, candle, bottle, jar,
  box, hat, ball, brush, clock, key, sock, shell, stone, flower, leaf. Never use
  vague nouns like "items", "decor", "some things".
- Do not repeat the same pair of object categories in more than a few scenes.
- Colours must be plain colour words and must be plausible for that object. Never
  write a blue banana or a black apple: the generator's object prior fights the
  prompt and the scene then measures the wrong thing. Fruit and vegetables may
  only take colours they actually occur in. Manufactured objects (mug, book,
  candle, bowl, plate, hat, notebook, bottle, box) may take any colour.
- Never describe mood, style, lighting or quality ("cosy", "rustic", "beautiful").
  Nothing in the prompt may be unjudgeable.
- counts must be between 1 and 2, and the counts in one scene must sum to
  exactly 3. So a scene is either three entries of one, or two entries where one
  asks for two. Nothing wider.
- {items} distinct object entries per scene.
- Vary the surface: wooden table, marble counter, white cloth, metal tray, desk.

Target distribution across the {n} scenes, spread them evenly:
- colours: do not let any one colour exceed a quarter of all entries.

{relations_clause}"""

RELATIONS_ON = """Also: in about half the scenes, add ONE relation to
"relations" as {"subject": "<object>", "relation": "left_of", "object":
"<other object>"} and state it in the prompt using picture-plane wording such as
"with the mug further left in the picture than the pear". Leave "relations" empty
for the other half."""

RELATIONS_OFF = 'Always leave "relations" empty.'


def build_instruction(n: int, items: int, with_relations: bool) -> str:
    return INSTRUCTION.format(
        n=n,
        items=items,
        relations_clause=RELATIONS_ON if with_relations else RELATIONS_OFF,
    )


COLOR_WORDS = frozenset({
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "grey", "gray", "silver", "gold", "beige",
})
"""Colour words a spec may use.

"clear" and "transparent" are excluded on purpose, and they did turn up: three
entries of the first clean corpus asked for clear jars. A detector asked for an
object's dominant colour will never answer "clear" -- transparency is the absence
of a colour, not one of them -- so such an entry is a guaranteed mismatch that
says nothing about the generator. Silver and gold stay, because a detector does
return them for metal.
"""

PLAUSIBLE_COLORS: dict[str, frozenset[str]] = {
    "apple": frozenset({"red", "green", "yellow"}),
    "banana": frozenset({"yellow", "green"}),
    "pear": frozenset({"green", "yellow", "brown"}),
    "orange": frozenset({"orange"}),
    "lemon": frozenset({"yellow", "green"}),
    "tomato": frozenset({"red", "green"}),
    "carrot": frozenset({"orange"}),
    "egg": frozenset({"white", "brown"}),
    "bread": frozenset({"brown", "white"}),
    "strawberry": frozenset({"red"}),
    "grape": frozenset({"green", "purple", "black"}),
    "peach": frozenset({"orange", "pink", "yellow"}),
    "lime": frozenset({"green"}),
}
"""Colours a natural kind may plausibly be, enforced rather than merely asked for.

Only natural kinds are listed. A mug, book or candle genuinely comes in any
colour, so constraining those would remove variation the corpus needs; an apple
does not, and asking for a blue one changes what the trial measures.

This is enforced in `_check` because asking was tried and did not work. The
instruction has said "never write a blue banana" from the first version, and the
authored corpus still came back 35% implausible on apples alone -- blue x10,
black x4, brown x3, white x2 out of 55. The generator's object prior fights a
prompt like that, so the image fails for a reason unrelated to the capability
under test, and it fails by a different amount for each object, which quietly
turns the object identity into a confound of the difficulty tiers.

A category absent from this table is unconstrained. That is deliberate: the list
should grow when a natural kind actually turns up in a corpus, not be guessed at
in advance.
"""


class SpecRejected(ValueError):
    """A generated scene that cannot serve as a spec."""


def _check(scene: dict[str, Any], expect_items: int | None) -> None:
    prompt = str(scene.get("prompt", "")).strip()
    objects = scene.get("objects")
    if not prompt:
        raise SpecRejected("empty prompt")
    if not isinstance(objects, list) or not objects:
        raise SpecRejected("no objects")
    if expect_items is not None and len(objects) != expect_items:
        raise SpecRejected(f"expected {expect_items} entries, got {len(objects)}")
    total = sum(item.get("count") or 0 for item in objects if isinstance(item, dict))
    if total != TOTAL_OBJECTS:
        raise SpecRejected(f"scene holds {total} objects, want {TOTAL_OBJECTS}")
    lowered = prompt.lower()
    seen = set()
    for item in objects:
        noun = str(item.get("object", "")).strip().lower()
        count = item.get("count")
        if not noun:
            raise SpecRejected("object with no name")
        if noun in seen:
            raise SpecRejected(f"duplicate entry for {noun}")
        seen.add(noun)
        if not isinstance(count, int) or not 1 <= count <= 2:
            raise SpecRejected(f"count out of range for {noun}: {count!r}")
        # The defining property of a verifiable scene: the prompt must actually
        # say what the spec claims. A spec the prompt does not state cannot be
        # ticked off against the image, which is the whole point.
        if noun not in lowered and noun.rstrip("s") not in lowered:
            raise SpecRejected(f"prompt does not mention {noun}")
        colour = item.get("color")
        if colour and str(colour).strip().lower() not in lowered:
            raise SpecRejected(f"prompt does not mention colour {colour}")
        if colour and str(colour).strip().lower() not in COLOR_WORDS:
            raise SpecRejected(f"not a colour word: {colour}")
        allowed = PLAUSIBLE_COLORS.get(noun) or PLAUSIBLE_COLORS.get(noun.rstrip("s"))
        if allowed and colour and str(colour).strip().lower() not in allowed:
            raise SpecRejected(f"implausible colour: {colour} {noun}")
    for relation in scene.get("relations") or ():
        names = {str(relation.get("subject", "")).lower(),
                 str(relation.get("object", "")).lower()}
        if not names <= seen:
            raise SpecRejected("relation refers to an object not in the spec")
        if str(relation.get("relation")) not in RELATIONS:
            raise SpecRejected(f"unsupported relation {relation.get('relation')!r}")


def parse_scenes(
    text: str, *, prefix: str, expect_items: int | None = None
) -> tuple[list[SceneSpec], list[dict[str, Any]]]:
    """Turn one model reply into specs, keeping the rejects for inspection."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [], [{"reason": "no JSON array", "raw": text[:200]}]
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], [{"reason": f"json error: {exc.msg}", "raw": text[:200]}]

    accepted: list[SceneSpec] = []
    rejected: list[dict[str, Any]] = []
    for index, scene in enumerate(raw):
        if not isinstance(scene, dict):
            rejected.append({"reason": "not an object", "index": index})
            continue
        try:
            _check(scene, expect_items)
        except SpecRejected as exc:
            rejected.append({"reason": str(exc), "index": index,
                             "prompt": str(scene.get("prompt", ""))[:120]})
            continue
        accepted.append(
            SceneSpec.from_dict({
                "spec_id": f"{prefix}-{len(accepted):04d}",
                "prompt": str(scene["prompt"]).strip(),
                "objects": scene["objects"],
                "relations": scene.get("relations") or [],
                "surface": str(scene.get("surface", "")),
                "metadata": {"author": "llm", "source_index": index},
            })
        )
    return accepted, rejected


def generate(
    ask: Callable[[str], str],
    *,
    total: int,
    items: int,
    with_relations: bool,
    prefix: str,
    batch: int = 20,
    max_batches: int = 40,
) -> tuple[list[SceneSpec], list[dict[str, Any]]]:
    """Collect `total` accepted specs, batching until the quota is met."""
    accepted: list[SceneSpec] = []
    rejected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for _ in range(max_batches):
        if len(accepted) >= total:
            break
        want = min(batch, total - len(accepted))
        got, bad = parse_scenes(
            ask(build_instruction(want, items, with_relations)),
            prefix=prefix,
            expect_items=items,
        )
        rejected.extend(bad)
        for spec in got:
            key = spec.prompt.strip().lower()
            if key in seen_prompts:
                rejected.append({"reason": "duplicate prompt", "prompt": spec.prompt})
                continue
            seen_prompts.add(key)
            accepted.append(spec)
    renumbered = [
        SceneSpec.from_dict({**spec.to_dict(), "spec_id": f"{prefix}-{i:04d}"})
        for i, spec in enumerate(accepted[:total])
    ]
    return renumbered, rejected


def distribution_report(specs: Iterable[SceneSpec]) -> dict[str, Any]:
    """The realised distribution, so a skew is visible rather than assumed away.

    An LLM asked for balance will still over-produce its favourite counts and
    colours. Nothing here corrects that; it measures it, and the number goes in
    the paper alongside the corpus.
    """
    specs = list(specs)
    counts: Counter = Counter()
    colours: Counter = Counter()
    objects: Counter = Counter()
    widths: Counter = Counter()
    with_relations = 0
    for spec in specs:
        widths[spec.n_items] += 1
        with_relations += bool(spec.relations)
        for item in spec.objects:
            counts[item.count] += 1
            objects[item.object] += 1
            if item.color:
                colours[item.color] += 1
    entries = sum(counts.values()) or 1
    top_colour = colours.most_common(1)
    return {
        "n_specs": len(specs),
        "n_entries": entries,
        "items_per_spec": dict(sorted(widths.items())),
        "count_distribution": dict(sorted(counts.items())),
        "objects_per_scene_mean": (
            sum(s.n_objects for s in specs) / len(specs) if specs else 0.0
        ),
        "distinct_objects": len(objects),
        "top_objects": objects.most_common(8),
        "distinct_colors": len(colours),
        "top_colors": colours.most_common(8),
        "max_color_share": (top_colour[0][1] / entries) if top_colour else 0.0,
        "with_relations": with_relations,
    }


def spec_object_from_dict(value: dict[str, Any]) -> SpecObject:
    return SpecObject.from_dict(value)
