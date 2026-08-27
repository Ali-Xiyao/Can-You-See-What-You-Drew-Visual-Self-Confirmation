"""Pinned SmolVLM, Qwen2-VL, and InternVL observer backends."""

from __future__ import annotations

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


class SmolVLMObserver(BaseObserver):
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.observer_id = model_id
        self.revision = revision
        self.device = device
        model_path = _locked_local_path(model_id, revision)
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
        model_path = _locked_local_path(model_id, revision)
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
        model_path = _locked_local_path(model_id, revision)
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


def create_transformers_observer(name: str, model_id: str, revision: str, device: str) -> BaseObserver:
    if name == "smolvlm":
        return SmolVLMObserver(model_id, revision, device)
    if name == "qwen2vl":
        return Qwen2VLObserver(model_id, revision, device)
    if name == "internvl":
        return InternVLObserver(model_id, revision, device)
    raise ValueError(f"Unsupported transformer observer backend: {name}")
