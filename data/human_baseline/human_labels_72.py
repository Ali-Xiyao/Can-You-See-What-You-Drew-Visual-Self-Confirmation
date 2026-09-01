"""My own labels for the 72 blind candidates, written before seeing any verifier
output on them.

Recorded from `blind/blind_*.png`, which carried no boxes, no verdicts and no
prompt. Size follows the same rule the geometric verifier uses so the two are
comparable: LARGE when the object's longest side reaches the 95/512 scale bar
drawn on each tile, SMALL otherwise. `?` marks a size I judged to be within about
10% of that bar -- those are excluded from the size-accuracy figure rather than
being counted as either answer.

Shapes are named by what the object *is*, not by how it is drawn: a rendered cube
is a square, a rendered diamond is a square, a 3D prism seen end-on is a square.
That is the same convention the reference renders use.
"""

LABELS: dict[int, list[str]] = {
    0:  ["green triangle S", "green triangle S", "yellow square S", "blue circle L"],
    1:  ["green triangle L", "red circle S", "green square L"],
    2:  ["blue square L", "yellow triangle S?"],
    3:  ["red circle S", "green triangle L?", "red circle S"],
    4:  ["red triangle L", "green square S"],
    5:  ["yellow square L", "blue square L", "green circle S"],
    6:  ["red square L", "green circle S", "yellow square S"],
    7:  ["green square L", "red circle L"],
    8:  ["yellow triangle L", "red square L", "red square L", "yellow circle S"],
    9:  ["blue circle L", "yellow square L"],
    10: ["blue circle L", "green square L?"],
    11: ["red triangle L", "green square L"],
    12: ["blue circle L", "green circle S?", "yellow square L"],   # 2nd is teal
    13: ["yellow square L", "blue square S", "red circle S"],
    14: ["green circle S", "green circle S", "red square L?", "green square L"],
    15: ["red square L", "red square L", "yellow circle S", "red square L",
         "blue triangle L"],
    16: ["blue triangle L", "yellow circle S", "red square S"],
    17: ["yellow circle L", "blue square L?"],
    18: ["blue circle L", "blue square L", "blue circle S", "yellow circle S",
         "yellow circle S"],
    19: ["red square L", "yellow circle L", "blue square L"],
    20: ["green circle S", "yellow square L", "green square L"],
    21: ["red square S", "green square S", "blue circle L"],
    22: ["yellow circle S", "green square L", "blue square S"],
    23: ["green triangle L", "green circle S", "blue circle S", "blue circle S"],
    24: ["red square L", "blue circle L", "red triangle L"],
    25: ["yellow triangle L", "yellow square L"],
    26: ["red square L", "red circle S", "red triangle S"],
    27: ["red triangle L", "green triangle S", "green circle L", "green circle L"],
    28: ["blue square L", "red circle S", "red circle S"],
    29: ["blue triangle L", "blue circle S"],
    30: ["blue circle L", "green square L"],
    31: ["blue triangle L", "red circle S"],
    32: ["blue triangle L", "blue square L"],
    33: ["green square L", "blue square S", "yellow circle S"],
    34: ["green square L", "blue square L"],
    35: ["blue triangle L"],
    36: ["yellow triangle L?", "blue square L"],
    37: ["green circle L", "blue triangle L", "red square L?"],
    38: ["blue triangle L", "red square S"],
    39: ["green circle L", "green square S", "blue triangle L", "green square S"],
    40: ["blue circle L", "yellow triangle L"],
    41: ["yellow circle L", "red circle S"],
    42: ["blue circle L", "green square L"],
    43: ["yellow triangle L", "yellow circle L"],
    44: ["red square L?", "green circle L", "yellow square L"],
    45: ["green square L", "blue circle S?", "blue square L"],
    46: ["yellow circle L", "green square S", "blue triangle L"],
    47: ["green square L?", "green square S", "green circle L", "green circle L"],
    48: ["yellow square L", "blue circle S", "blue triangle S", "blue circle S"],
    49: ["red square L", "yellow square S", "blue circle S"],
    50: ["red circle L", "blue square L"],
    51: ["yellow circle L", "green square L", "red square L", "red square L?"],
    52: ["yellow triangle L", "blue circle L", "blue circle S"],
    53: ["blue circle L", "blue square S"],
    54: ["red triangle L", "yellow square S", "blue triangle S", "blue circle S"],
    55: ["yellow triangle L", "red square L", "yellow circle S"],
    56: ["blue square L", "yellow circle L"],
    57: ["blue triangle L", "yellow triangle S", "blue triangle S",
         "yellow triangle S"],
    58: ["green square L", "blue circle L", "red triangle L?"],
    59: ["green square L", "red circle S", "green triangle S", "green square L"],
    60: ["green square L", "green circle L"],   # 2nd is the cast shadow ellipse
    61: ["red square S", "yellow circle L"],
    62: ["green circle S", "green square L?", "red square L", "green square L"],
    63: ["red circle L", "yellow circle S", "blue square S"],
    64: ["blue circle L", "yellow triangle L", "blue square L"],
    65: ["blue circle L", "red triangle L", "blue square L"],
    66: ["green circle L", "blue square L"],
    67: ["blue circle L", "green square L", "red triangle L"],
    68: ["blue circle S", "yellow square L"],
    69: ["red circle L", "yellow square S", "green triangle S"],
    70: ["blue circle L", "yellow circle S", "yellow square L"],
    71: ["green circle L", "blue square L", "blue square L", "blue square L"],
}

# Tiles where an honest label is not available, recorded so they can be excluded
# rather than silently resolved in whichever direction favours a verifier.
NOTES = {
    12: "second object is teal, which is not in the four-colour palette",
    47: "fourth object is the large green ground-shadow ellipse, not a drawn object",
    60: "second object is the cast shadow under the cube, not a drawn object",
    15: "the blue shape is an arrow/pentagon, closest to triangle but not clean",
    33: "the blue band is part of the green slab's texture, not a separate object",
}


def parse(entry: str) -> dict:
    shape_color, size = entry.rsplit(" ", 1)
    color, shape = shape_color.split(" ")
    return {
        "color": color,
        "shape": shape,
        "size": {"L": "large", "S": "small"}[size[0]],
        "size_borderline": size.endswith("?"),
    }


def as_records() -> dict[int, list[dict]]:
    return {index: [parse(item) for item in items] for index, items in LABELS.items()}


if __name__ == "__main__":
    import json

    records = as_records()
    print(json.dumps({"labels": records, "notes": NOTES}, indent=1))
