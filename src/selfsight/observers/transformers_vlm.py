"""Pinned SmolVLM, Qwen2/2.5-VL, Qwen3-VL, and InternVL observer backends."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from selfsight.models import locked_model, snapshot_path
from selfsight.observers.base import BaseObserver
from selfsight.schemas import AtomicQuestion


def _locked_local_path(model_id: str, revision: str) -> Path:
    expected = str(locked_model(model_id)["revision"])
    if revision != expected:
        raise ValueError(f"Observer revision mismatch for {model_id}: {revision} != {expected}")
    return snapshot_path(model_id)


def directory_model_digest(root: str | Path) -> str:
    """Content binding for a weight directory that is not in HF-cache layout.

    Some observer checkpoints are plain (e.g. ModelScope) directories outside
    `SELFSIGHT_MODEL_ROOT` and therefore have no HF revision to pin. Hashing 17GB
    of shards on every load is not viable, so bind the exact weight set by the
    index, the config, and every shard's (name, size). Changing any shard changes
    its size or the index, and either changes this digest.
    """

    root = Path(root).resolve()
    payload: dict[str, object] = {}
    for name in ("config.json", "model.safetensors.index.json"):
        candidate = root / name
        if candidate.is_file():
            payload[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    payload["shards"] = sorted(
        (item.name, item.stat().st_size) for item in root.glob("*.safetensors")
    )
    if not payload["shards"]:
        raise FileNotFoundError(f"No safetensors shards under {root}")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _external_local_path(model_id: str, revision: str) -> Path:
    """Resolve a directory-sourced observer and verify its content digest.

    `revision` carries `dir-sha256:<digest>` for these models. The directory is
    read from `SELFSIGHT_EXTERNAL_MODEL_DIR` so the absolute path is not baked
    into a committed config.
    """

    prefix = "dir-sha256:"
    if not revision.startswith(prefix):
        raise ValueError(
            f"Directory-sourced observer {model_id} needs a '{prefix}<digest>' revision"
        )
    configured = os.environ.get("SELFSIGHT_EXTERNAL_MODEL_DIR")
    if not configured:
        raise RuntimeError(
            f"{model_id} is a directory-sourced observer; set SELFSIGHT_EXTERNAL_MODEL_DIR "
            "to its weight directory"
        )
    root = Path(configured).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SELFSIGHT_EXTERNAL_MODEL_DIR does not exist: {root}")
    actual = directory_model_digest(root)
    expected = revision[len(prefix) :]
    if actual != expected:
        raise ValueError(
            f"Observer weight digest mismatch for {model_id} at {root}: "
            f"{actual} != {expected}"
        )
    return root


def _resolve_observer_path(model_id: str, revision: str) -> Path:
    if revision.startswith("dir-sha256:"):
        return _external_local_path(model_id, revision)
    return _locked_local_path(model_id, revision)


class SmolVLMObserver(BaseObserver):
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.observer_id = model_id
        self.revision = revision
        self.device = device
        model_path = _resolve_observer_path(model_id, revision)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa",
        ).to(device).eval()

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        import torch

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        answers = []
        for question in questions:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question.text}]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
            prefix = inputs["input_ids"].shape[1]
            answers.append(self.processor.batch_decode(generated[:, prefix:], skip_special_tokens=True)[0])
        return answers


class Qwen2VLObserver(BaseObserver):
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor

        if "Qwen2.5" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass

        self.observer_id = model_id
        self.revision = revision
        self.device = device
        model_path = _resolve_observer_path(model_id, revision)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = ModelClass.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        import torch

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        answers = []
        for question in questions:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question.text}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            answers.append(self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        return answers


class InternVLObserver(BaseObserver):
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.observer_id = model_id
        self.revision = revision
        self.device = device
        self.torch = torch
        model_path = _resolve_observer_path(model_id, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)

    def _pixels(self, image: Image.Image):
        from torchvision.transforms import InterpolationMode, v2

        transform = v2.Compose(
            [
                v2.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(self.torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return transform(image).unsqueeze(0).to(dtype=self.torch.bfloat16, device=self.device)

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        with Image.open(image_path) as opened:
            pixels = self._pixels(opened.convert("RGB"))
        generation_config = {"max_new_tokens": 16, "do_sample": False}
        return [
            self.model.chat(self.tokenizer, pixels, question.text, generation_config)
            for question in questions
        ]


class Qwen3VLObserver(BaseObserver):
    """Qwen3-VL. Instruct variants only.

    Thinking variants are deliberately unsupported here: the observer must emit a
    short atomic answer that `normalize_answer` can parse, and a reasoning trace
    both breaks that and inflates run-to-run variance against the >=90% repeat
    agreement requirement.
    """

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        # Reject before importing or loading anything.
        if "thinking" in model_id.lower():
            raise ValueError(
                f"Refusing a Thinking observer ({model_id}): atomic answers must be short "
                "and repeatable. Use an Instruct variant."
            )
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.observer_id = model_id
        self.revision = revision
        self.device = device
        model_path = _resolve_observer_path(model_id, revision)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        import torch

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        answers = []
        for question in questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question.text},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
            trimmed = [
                output[len(source) :]
                for source, output in zip(inputs.input_ids, generated, strict=False)
            ]
            answers.append(
                self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            )
        return answers


def create_transformers_observer(name: str, model_id: str, revision: str, device: str) -> BaseObserver:
    if name == "smolvlm":
        return SmolVLMObserver(model_id, revision, device)
    if name == "qwen2vl":
        return Qwen2VLObserver(model_id, revision, device)
    if name == "qwen3vl":
        return Qwen3VLObserver(model_id, revision, device)
    if name == "internvl":
        return InternVLObserver(model_id, revision, device)
    raise ValueError(f"Unsupported transformer observer backend: {name}")
