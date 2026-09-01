"""Frozen external detectors for verifiable natural scenes.

These implement `selfsight.v4.verifier.Detector`. Each one sees the image and
nothing else -- not the prompt it was generated from, not the spec, not the gold
answer. That is the whole basis on which their output can be used to score the
generator: a detector that knew what was asked for would be marking its own
homework, and the failure this project studies is precisely a model agreeing
with the prompt against the pixels.

Two are used, and the reason is the shape of their errors rather than their
scores. On 72 blind human-labelled images Qwen3-VL-8B and InternVL3.5-8B reach
F1 0.986 and 0.978, but Qwen skews toward inventing objects (3 cases against 1)
and InternVL toward missing them (4 against 1). Opposite skews that overlap on a
single error are what makes a pair worth having; two models that fail the same
way would agree confidently and be wrong together.

Qwen3-VL-4B is deliberately not offered. Its F1 of 0.943 is respectable, but 12
of its errors are missing objects, and a detector that misses an object cannot
be told apart from a generator that did not draw one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image

QWEN3VL_8B = r"H:\Xiyao_Wang\001_models\Qwen3-VL-8B-Instruct"
INTERNVL35_8B = r"H:\Xiyao_Wang\001_models\InternVL3_5-8B"

INSTRUCTION = """List every distinct physical object in this photograph.

Rules:
- Name each object with a single common noun: apple, mug, book, candle, hat.
- Count each object separately. Three apples are three entries, not one.
- Give each object's dominant colour as a plain colour word.
- Do NOT list the surface the objects rest on (table, tray, counter, cloth),
  the background, the wall, shadows, or reflections.
- If an object is partly hidden behind another, still list it.

Return ONLY a JSON array, no other text. Each element must be:
{"object": <noun>, "color": <colour>, "box": [x0, y0, x1, y1]}

where the box is that object's bounding box, with (0,0) at the top-left of the
image. If there are no objects, return []."""

CROP_INSTRUCTION = """This is a close crop of one part of a photograph.

List every distinct physical object you can see in it, using the same format: a
single common noun, a plain colour word, and a bounding box in this crop.

Do NOT list the surface, background, shadows or reflections.

Return ONLY a JSON array of {"object": <noun>, "color": <colour>,
"box": [x0, y0, x1, y1]}. If you see no object, return []."""


def _rescale(box: Any, width: int, height: int) -> list[float] | None:
    """Put a returned box into this image's pixel frame.

    Qwen returns boxes in a normalised 0-1000 frame whatever the instruction
    asks for. That is detected from the values rather than assumed, so a model
    which does honour the pixel request is not silently rescaled into nonsense.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if max(values) <= 1.05:
        return [values[0] * width, values[1] * height,
                values[2] * width, values[3] * height]
    if max(values) > max(width, height) * 1.05:
        return [values[0] / 1000 * width, values[1] / 1000 * height,
                values[2] / 1000 * width, values[3] / 1000 * height]
    return values


def parse_reply(text: str, width: int, height: int) -> tuple[list[dict[str, Any]], str | None]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [], "no_json_array"
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"json_error:{exc.msg}"
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        noun = str(item.get("object") or item.get("name") or "").strip().lower()
        if not noun:
            continue
        colour = item.get("color") or item.get("colour")
        box = _rescale(item.get("box") or item.get("bbox"), width, height)
        entry: dict[str, Any] = {
            "object": noun,
            "color": str(colour).strip().lower() if colour else None,
        }
        if box:
            entry["bbox"] = box
            entry["center"] = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        out.append(entry)
    return out, None


class VlmDetector:
    """Wraps an `(image, instruction) -> reply` callable as a `Detector`."""

    def __init__(self, detector_id: str, run: Callable[[Image.Image, str], str]):
        self.detector_id = detector_id
        self._run = run

    def _ask(self, image: Image.Image, instruction: str) -> list[dict[str, Any]]:
        reply = self._run(image, instruction)
        detections, error = parse_reply(reply, image.width, image.height)
        if error:
            # An unparseable reply is not an empty scene. Returning [] here would
            # be scored as "the model drew nothing", which is a wrong answer
            # invented by the harness rather than observed in the pixels.
            raise ValueError(f"{self.detector_id}: {error}: {reply[:200]!r}")
        return detections

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        with Image.open(image_path) as handle:
            return self._ask(handle.convert("RGB"), INSTRUCTION)

    def detect_crop(
        self, image_path: str, bbox: tuple[float, float, float, float]
    ) -> list[dict[str, Any]]:
        """Re-read one region, enlarged. This is the ladder's second level.

        The crop is upsampled to 448px on its short side: the point of the step
        is to put more pixels on the disputed object than the whole-image pass
        had, and handing back a 40x40 patch would just repeat the first question
        with less context.
        """
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            x0 = max(0, int(bbox[0]))
            y0 = max(0, int(bbox[1]))
            x1 = min(image.width, int(bbox[2]))
            y1 = min(image.height, int(bbox[3]))
            if x1 <= x0 or y1 <= y0:
                return []
            crop = image.crop((x0, y0, x1, y1))
        short = min(crop.width, crop.height)
        if short < 448:
            factor = 448 / short
            crop = crop.resize(
                (int(crop.width * factor), int(crop.height * factor)), Image.LANCZOS
            )
        found = self._ask(crop, CROP_INSTRUCTION)
        for item in found:  # boxes are in crop coordinates; the caller has none
            item.pop("bbox", None)
            item.pop("center", None)
        return found


def load_qwen3vl(path: str = QWEN3VL_8B, device: str = "cuda:0") -> VlmDetector:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    ).eval()

    def run(image: Image.Image, instruction: str) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instruction}]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

    return VlmDetector("qwen3vl-8b", run)


def load_internvl(path: str = INTERNVL35_8B, device: str = "cuda:0") -> VlmDetector:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).eval().to(device)

    # A single 448 tile. The corpus is 512x512, so dynamic tiling would only add
    # crops of a scene that already fits, and crops invite double counting -- the
    # one error mode indistinguishable from the generator drawing an extra object.
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def run(image: Image.Image, instruction: str) -> str:
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(device)
        return model.chat(
            tokenizer, pixel_values, "<image>\n" + instruction,
            dict(max_new_tokens=512, do_sample=False),
        )

    return VlmDetector("internvl3.5-8b", run)


LOADERS: dict[str, Callable[..., VlmDetector]] = {
    "qwen3vl": load_qwen3vl,
    "internvl": load_internvl,
}


def load(name: str, device: str = "cuda:0") -> VlmDetector:
    if name not in LOADERS:
        raise KeyError(f"unknown detector {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](device=device)


def cached_detections(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read a detections jsonl into `image_path -> detections`.

    Rows without a `detections` key are skipped rather than stored empty. A
    to-do list and a results file have the same shape here, and merging the
    former silently blanked real results once already.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "detections" in row:
            out[str(row["image_path"])] = list(row["detections"])
    return out
