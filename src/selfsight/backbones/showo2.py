"""Show-o2 adapter for the v2.2 Joint Generate–Observe Readiness gate."""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml
from PIL import Image

from selfsight.adapters import GradientResult, ModelAdapter
from selfsight.backbones.base import (
    BackboneCapabilities,
    BackboneIdentity,
    LoraTargetAudit,
    ResourceReport,
)
from selfsight.data.questions import normalize_answer
from selfsight.models import load_model_lock, locked_model, repository_path, snapshot_path
from selfsight.rfo.isolation import hard_render
from selfsight.schemas import (
    AtomicObservation,
    AtomicQuestion,
    CandidateRecord,
    ObservationResult,
)
from selfsight.training.checkpoint import load_checkpoint, save_checkpoint
from selfsight.training.gradients import collect_lora_gradient, collect_lora_gradient_accumulated
from selfsight.utils.cuda import cuda_device_index
from selfsight.utils.hashing import rgb_sha256, sha256_json
from selfsight.utils.jsonl import atomic_write_json


@dataclass(frozen=True)
class Showo2GenerationBatch:
    prompts: tuple[str, ...]
    images: tuple[str | Path | Image.Image, ...]
    sample_ids: tuple[str, ...]
    latent_seed: int = 20260828


@dataclass(frozen=True)
class Showo2ReplayBatch:
    images: tuple[str | Path | Image.Image, ...]
    questions: tuple[str, ...]
    answers: tuple[str, ...]
    sample_ids: tuple[str, ...]
    latent_seed: int = 20260828


class Showo2Adapter(ModelAdapter):
    """Lazy, local-only wrapper around the locked official Show-o2 implementation."""

    def __init__(
        self,
        *,
        backbone_config: str | Path = "configs/backbones/showo2_1p5b.yaml",
        lock_path: str | Path = "configs/models.lock.yaml",
        device: str | None = None,
        dtype: str = "bf16",
        lazy: bool = True,
        max_new_tokens: int = 32,
    ) -> None:
        config_path = Path(backbone_config).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Invalid Show-o2 backbone config: {config_path}")
        self.backbone_config_path = config_path
        self.backbone_config = raw
        self.lock_path = Path(lock_path).resolve()
        self.model_id = str(raw["backbone_id"])
        self.revision = str(raw["revision"])
        lock_revision = str(locked_model(self.model_id, self.lock_path)["revision"])
        if self.revision != lock_revision:
            raise ValueError(
                f"Backbone config/lock revision mismatch: {self.revision} != {lock_revision}"
            )
        profile = raw["official_profile"]
        self.native_resolution = int(profile["resolution"])
        self.mmu_resolution = int(profile.get("mmu_resolution", self.native_resolution))
        self.generation_steps = int(profile["generation_steps"])
        self.guidance_scale = float(profile["guidance_scale"])
        self.device_spec = str(device or raw["hardware"]["generator_device"])
        if dtype not in {"bf16", "float32"}:
            raise ValueError(f"Unsupported Show-o2 dtype: {dtype}")
        self.dtype_name = dtype
        self.max_new_tokens = int(max_new_tokens)
        self._loaded = False
        self._lora_attached = False
        self._runtime: dict[str, Any] = {}
        self.model: Any = None
        self.vae_model: Any = None
        self.tokenizer: Any = None
        self.token_ids: dict[str, int] = {}
        self.device: Any = None
        self.dtype: Any = None
        if not lazy:
            self._load()

    @property
    def identity(self) -> BackboneIdentity:
        source = self.backbone_config["source"]
        return BackboneIdentity(
            model_id=self.model_id,
            revision=self.revision,
            source_repository=str(source["repository_id"]),
            source_revision=str(source["revision"]),
            implementation="showo2_qwen2_5_wan21",
            native_resolution=self.native_resolution,
        )

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(True, True, True, True, True)

    @staticmethod
    def _package_root(module: Any) -> Path | None:
        file_name = getattr(module, "__file__", None)
        return Path(file_name).resolve() if file_name else None

    def _assert_import_origin(self, package: str, source_root: Path) -> None:
        existing = sys.modules.get(package)
        if existing is None:
            return
        origin = self._package_root(existing)
        if origin is None or not origin.is_relative_to(source_root):
            raise RuntimeError(
                f"Python package collision for {package!r}: {origin}. "
                "Run Show-o2 in its dedicated environment/process."
            )

    @staticmethod
    def _assert_materialized(module: Any, label: str) -> None:
        meta = [
            f"parameter:{name}"
            for name, value in module.named_parameters()
            if bool(getattr(value, "is_meta", False))
        ]
        meta.extend(
            f"buffer:{name}"
            for name, value in module.named_buffers()
            if bool(getattr(value, "is_meta", False))
        )
        if meta:
            preview = ", ".join(meta[:8])
            raise RuntimeError(f"{label} retained meta tensors after weight loading: {preview}")

    @staticmethod
    def _ensure_legacy_shard_index(snapshot: Path, torch_module: Any) -> Path | None:
        """Create the Diffusers index omitted by the official Show-o2 7B snapshot."""

        if (snapshot / "pytorch_model.bin").is_file():
            return None
        pattern = re.compile(r"pytorch_model-(\d{5})-of-(\d{5})\.bin")
        shards = sorted(
            path
            for path in snapshot.glob("pytorch_model-*-of-*.bin")
            if pattern.fullmatch(path.name)
        )
        if not shards:
            return None
        totals = {int(pattern.fullmatch(path.name).group(2)) for path in shards}
        if len(totals) != 1 or totals.pop() != len(shards):
            raise RuntimeError(f"Incomplete Show-o2 checkpoint shard set: {snapshot}")

        # The upstream loader overrides WEIGHTS_NAME with pytorch_model.bin but keeps
        # Diffusers' index constant, so this exact filename is required.
        index_path = snapshot / "diffusion_pytorch_model.bin.index.json"
        shard_names = {path.name for path in shards}
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
            if isinstance(weight_map, Mapping) and set(weight_map.values()) == shard_names:
                return index_path
            raise RuntimeError(f"Invalid derived Show-o2 shard index: {index_path}")

        weight_map: dict[str, str] = {}
        total_size = 0
        for shard in shards:
            state_dict = torch_module.load(
                shard,
                map_location="meta",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(state_dict, Mapping):
                raise TypeError(f"Show-o2 shard is not a state dict: {shard}")
            for name, value in state_dict.items():
                key = str(name)
                if key in weight_map:
                    raise RuntimeError(f"Duplicate Show-o2 tensor across shards: {key}")
                if not torch_module.is_tensor(value):
                    raise TypeError(f"Non-tensor Show-o2 state entry: {key}")
                weight_map[key] = shard.name
                total_size += int(value.numel()) * int(value.element_size())
            del state_dict
        atomic_write_json(
            index_path,
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        return index_path

    def _load(self) -> None:
        if self._loaded:
            return
        import torch

        self.device = torch.device(self.device_spec)
        self.dtype = torch.bfloat16 if self.dtype_name == "bf16" else torch.float32
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Show-o2 readiness requires a CUDA device")
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"BF16 is not supported on {self.device}")

        source_repository = str(self.backbone_config["source"]["repository_id"])
        source_root = repository_path(
            source_repository,
            lock_path=self.lock_path,
        ) / str(self.backbone_config["source"]["subtree"])
        if not (source_root / "models" / "modeling_showo2_qwen2_5.py").is_file():
            raise FileNotFoundError(f"Locked Show-o2 source subtree is missing: {source_root}")
        self._assert_import_origin("models", source_root)
        self._assert_import_origin("transport", source_root)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
        from models.misc import get_text_tokenizer, prepare_gen_input
        from transport import Sampler, create_transport

        showo_path = snapshot_path(self.model_id, lock_path=self.lock_path)
        self._ensure_legacy_shard_index(showo_path, torch)
        profile = self.backbone_config["official_profile"]
        t2i_tokens = int(profile["t2i_image_tokens_with_time"])
        mmu_tokens = int(profile["mmu_image_tokens_with_time"])
        latent_height = int(profile["latent_height"])
        latent_width = int(profile["latent_width"])
        mmu_latent_height = int(profile.get("mmu_latent_height", latent_height))
        mmu_latent_width = int(profile.get("mmu_latent_width", latent_width))
        if t2i_tokens - 1 != latent_height * latent_width:
            raise RuntimeError("Show-o2 T2I token/latent geometry mismatch")
        if mmu_tokens - 1 != mmu_latent_height * mmu_latent_width:
            raise RuntimeError("Show-o2 MMU token/latent geometry mismatch")
        language_base_id = str(profile["language_base_id"])
        qwen_path = snapshot_path(language_base_id, lock_path=self.lock_path)
        siglip_path = snapshot_path("google/siglip-so400m-patch14-384", lock_path=self.lock_path)
        wan_path = snapshot_path(
            "Wan-AI/Wan2.1-T2V-14B",
            lock_path=self.lock_path,
            require_complete=False,
        )
        vae_path = wan_path / "Wan2.1_VAE.pth"
        if not vae_path.is_file():
            raise FileNotFoundError(f"Locked Wan VAE is missing: {vae_path}")

        tokenizer, token_ids = get_text_tokenizer(
            str(qwen_path),
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name="qwen2_5",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"for .*: copying from a non-meta parameter.*",
                category=UserWarning,
                module=r"torch\.nn\.modules\.module",
            )
            model = Showo2Qwen2_5.from_pretrained(
                str(showo_path),
                llm_model_path=str(qwen_path),
                clip_pretrained_model_path=str(siglip_path),
                use_safetensors=False,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map={"": str(self.device)},
                torch_dtype=self.dtype,
            )
        native_position_tokens = int(model.position_embedding.weight.shape[0])
        if native_position_tokens != mmu_tokens - 1:
            raise RuntimeError(
                "Show-o2 checkpoint position embedding is incompatible with MMU geometry: "
                f"{native_position_tokens} != {mmu_tokens - 1}"
            )
        if int(model.image_position_ids.shape[-1]) != t2i_tokens - 1:
            model.image_position_ids = torch.arange(t2i_tokens - 1).expand((1, -1))
        model.config.image_latent_height = latent_height
        model.config.image_latent_width = latent_width
        self._assert_materialized(model, "Show-o2")
        model = model.to(self.device, dtype=self.dtype)
        model.requires_grad_(False).eval()
        vae_model = WanVAE(vae_pth=str(vae_path), dtype=self.dtype, device=self.device)
        self._assert_materialized(vae_model.model, "WanVAE")

        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.token_ids = {key: int(value) for key, value in token_ids.items()}
        self.num_t2i_image_tokens = t2i_tokens
        self.num_mmu_image_tokens = mmu_tokens
        self.max_seq_len = int(profile["max_sequence_length"])
        self.max_text_len = self.max_seq_len - self.num_t2i_image_tokens - 4
        self.image_latent_dim = int(profile["image_latent_dim"])
        self.patch_size = int(profile["patch_size"])
        self.latent_height = latent_height
        self.latent_width = latent_width
        transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=None,
            sample_eps=None,
            snr_type="lognorm",
            do_shift=True,
            seq_len=self.num_t2i_image_tokens,
        )
        self.transport = transport
        self.sampler = Sampler(transport)
        self._runtime = {
            "omni_attn_mask_naive": omni_attn_mask_naive,
            "prepare_gen_input": prepare_gen_input,
        }
        self._loaded = True

    def _fork_devices(self) -> list[int]:
        if self.device.type != "cuda":
            return []
        return [self.device.index if self.device.index is not None else 0]

    def _generate_one(self, prompt: str, seed: int) -> Image.Image:
        self._load()
        import torch

        prepare_gen_input = self._runtime["prepare_gen_input"]
        omni_attn_mask_naive = self._runtime["omni_attn_mask_naive"]
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed(int(seed))
            text, null_text, positions, null_positions = prepare_gen_input(
                [prompt],
                self.tokenizer,
                self.num_t2i_image_tokens,
                self.token_ids["bos_id"],
                self.token_ids["eos_id"],
                self.token_ids["boi_id"],
                self.token_ids["eoi_id"],
                int(self.tokenizer.pad_token_id),
                self.token_ids["img_pad_id"],
                self.max_text_len,
                self.device,
            )
            latent = torch.randn(
                (
                    1,
                    self.image_latent_dim,
                    self.latent_height * self.patch_size,
                    self.latent_width * self.patch_size,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            if self.guidance_scale > 0:
                latent = torch.cat([latent, latent], dim=0)
                text = torch.cat([text, null_text], dim=0)
                positions = torch.cat([positions, null_positions], dim=0)
            attention = omni_attn_mask_naive(
                text.size(0), self.max_seq_len, positions, self.device
            ).to(self.dtype)
            model_kwargs = {
                "text_tokens": text,
                "attention_mask": attention,
                "modality_positions": positions,
                "output_hidden_states": True,
                "max_seq_len": self.max_seq_len,
                "guidance_scale": self.guidance_scale,
            }
            sample_fn = self.sampler.sample_ode(
                sampling_method="euler",
                num_steps=self.generation_steps,
                atol=1e-6,
                rtol=1e-3,
                reverse=False,
                time_shifting_factor=3.0,
            )
            with torch.inference_mode():
                sample = sample_fn(latent, self.model.t2i_generate, **model_kwargs)[-1]
                if self.guidance_scale > 0:
                    sample = torch.chunk(sample, 2)[0]
                decoded = self.vae_model.batch_decode(sample.unsqueeze(2)).squeeze(2)
            array = (
                torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)[0]
                .permute(1, 2, 0)
                .float()
                .cpu()
                .numpy()
            )
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")

    def generate_images(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        output_dir: str | Path,
        checkpoint_id: str,
    ) -> list[CandidateRecord]:
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal length")
        self._load()
        output_dir = Path(output_dir).resolve()
        records = []
        was_training = self.model.training
        self.model.eval()
        try:
            for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
                prompt_id = sha256_json({"benchmark": "2.2", "prompt": prompt})[:20]
                candidate_id = f"{checkpoint_id}-{prompt_id}-{int(seed)}-{index}"
                image_path = output_dir / f"{candidate_id}.png"
                evidence = hard_render(self._generate_one(prompt, int(seed)), image_path)
                records.append(
                    CandidateRecord(
                        candidate_id=candidate_id,
                        prompt_id=prompt_id,
                        scene_id=prompt_id,
                        sampling_seed=int(seed),
                        image_path=str(image_path),
                        rgb_sha256=str(evidence["rgb_sha256"]),
                        generator_id=self.model_id,
                        generator_revision=self.revision,
                        checkpoint_id=checkpoint_id,
                        metadata={
                            "prompt": prompt,
                            "native_resolution": self.native_resolution,
                            "generation_steps": self.generation_steps,
                            "guidance_scale": self.guidance_scale,
                            "source_revision": self.identity.source_revision,
                        },
                    )
                )
        finally:
            self.model.train(was_training)
        return records

    def _image_tensor(
        self, image_or_path: Image.Image | str | Path, *, resolution: int | None = None
    ):
        self._load()
        from torchvision import transforms

        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            with Image.open(image_or_path) as opened:
                image = opened.convert("RGB")
        target_resolution = int(resolution or self.native_resolution)
        transform = transforms.Compose(
            (
                transforms.Resize(
                    target_resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop((target_resolution, target_resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
        )
        return transform(image)

    def _encode_images(
        self,
        images: Sequence[Image.Image | str | Path],
        *,
        seed: int,
        resolution: int | None = None,
    ):
        self._load()
        import torch

        pixels = torch.stack(
            [self._image_tensor(image, resolution=resolution) for image in images]
        ).to(self.device, dtype=self.dtype)
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed(int(seed))
            with torch.no_grad():
                return self.vae_model.sample(pixels.unsqueeze(2)).squeeze(2).to(self.dtype)

    def encode_image_targets(self, images: Sequence[Image.Image | str | Path]) -> Any:
        return self._encode_images(images, seed=20260828)

    def _image_embeddings(self, image: Image.Image, *, seed: int):
        import torch

        latent = self._encode_images((image,), seed=seed, resolution=self.mmu_resolution)
        with torch.inference_mode():
            semantic = self.model.image_embedder_und(latent)
            generation = self.model.image_embedder_gen(latent)
            position_ids = torch.arange(semantic.shape[1], device=self.device).expand((1, -1))
            if int(semantic.shape[1]) != int(self.model.position_embedding.weight.shape[0]):
                raise RuntimeError("Show-o2 MMU image tokens do not match position embedding")
            semantic = semantic + self.model.position_embedding(position_ids)
            semantic = self.model.und_trans(semantic)["last_hidden_state"]
            return self.model.fusion_proj(torch.cat([semantic, generation], dim=-1))

    def _observe_one(self, image_embeds: Any, question: str) -> str:
        import torch

        omni_attn_mask_naive = self._runtime["omni_attn_mask_naive"]
        system_ids = self.tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
        )["input_ids"]
        role_user = self.tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        role_assistant = self.tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)[
            "input_ids"
        ]
        question_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        text_a = torch.tensor(
            [self.token_ids["bos_id"], *system_ids, *role_user],
            device=self.device,
        )[None]
        text_b = torch.tensor(
            [self.token_ids["boi_id"], self.token_ids["eoi_id"], *question_ids, *role_assistant],
            device=self.device,
        )[None]
        embeds_a = self.model.showo.model.embed_tokens(text_a)
        embeds_b = self.model.showo.model.embed_tokens(text_b)
        time = self.model.time_embed(torch.tensor([[1.0]], device=self.device), embeds_a.dtype)
        if hasattr(self.model, "time_embed_proj"):
            time = self.model.time_embed_proj(time)
        inputs = torch.cat(
            (embeds_a, embeds_b[:, :1], time, image_embeds, embeds_b[:, 1:]), dim=1
        ).to(self.dtype)
        positions = torch.tensor(
            [int(text_a.shape[1]) + 2, self.num_mmu_image_tokens],
            device=self.device,
        )[None, None]
        attention = omni_attn_mask_naive(
            B=inputs.size(0),
            LEN=inputs.size(1),
            modalities=positions,
            device=self.device,
            inverted=True,
        ).to(inputs.dtype)
        with torch.inference_mode():
            output = self.model.mmu_generate(
                input_embeds=inputs,
                attention_mask=attention,
                top_k=1,
                max_new_tokens=self.max_new_tokens,
                eos_token=self.tokenizer.eos_token_id,
            )
        if not output:
            return ""
        tokens = torch.stack(output).reshape(1, -1)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def observe_atoms(
        self,
        image_path: str | Path,
        questions: Sequence[AtomicQuestion],
    ) -> ObservationResult:
        self._load()
        started = datetime.now(timezone.utc).isoformat()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image_hash = rgb_sha256(image)
        latent_seed = int(image_hash[:16], 16) % (2**31)
        was_training = self.model.training
        self.model.eval()
        answers = []
        try:
            image_embeds = self._image_embeddings(image, seed=latent_seed)
            for question in questions:
                question_started = perf_counter()
                raw = self._observe_one(image_embeds, question.text)
                normalized = normalize_answer(raw, question)
                answers.append(
                    AtomicObservation(
                        question_id=question.question_id,
                        raw_answer=raw,
                        normalized_answer=normalized,
                        abstain=normalized is None,
                        latency_ms=(perf_counter() - question_started) * 1000.0,
                    )
                )
        finally:
            self.model.train(was_training)
        return ObservationResult(
            request_id=sha256_json(
                {"rgb_sha256": image_hash, "questions": [item.question_id for item in questions]}
            ),
            observer_id=self.model_id,
            observer_revision=self.revision,
            rgb_sha256=image_hash,
            answers=tuple(answers),
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def discover_lora_targets(self) -> LoraTargetAudit:
        self._load()
        import torch

        linear = tuple(
            name
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear)
        )
        suffixes = Counter(name.rsplit(".", 1)[-1] for name in linear)
        shared = tuple(name for name in linear if name.startswith("showo.model.layers."))
        generation = tuple(
            name
            for name in linear
            if name.startswith(("diffusion_head_a.", "diffusion_head_b.", "image_embedder_gen."))
        )
        forbidden = (
            "showo.model.embed_tokens",
            "showo.lm_head",
            "image_embedder_und",
            "und_trans",
            "position_embedding",
            "fusion_proj",
        )
        return LoraTargetAudit(
            model_id=self.model_id,
            revision=self.revision,
            linear_modules=linear,
            suffix_counts=dict(sorted(suffixes.items())),
            shared_transformer_candidates=shared,
            generation_head_candidates=generation,
            frozen_or_forbidden=forbidden,
        )

    def attach_lora(
        self,
        *,
        target_modules: Sequence[str],
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> Mapping[str, Any]:
        self._load()
        if self._lora_attached:
            raise RuntimeError("Show-o2 LoRA is already attached")
        audit = self.discover_lora_targets()
        allowed = set(audit.shared_transformer_candidates)
        targets = tuple(str(item) for item in target_modules)
        if not targets:
            raise ValueError("Show-o2 LoRA targets must come from a saved module-tree audit")
        invalid = sorted(set(targets).difference(allowed))
        if invalid:
            raise ValueError(f"Unaudited or non-shared Show-o2 LoRA targets: {invalid[:20]}")
        peft_targets = tuple(name.removeprefix("showo.") for name in targets)
        from peft import LoraConfig, TaskType, inject_adapter_in_model

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(peft_targets),
        )
        # The upstream multimodal wrapper directly calls
        # ``self.showo.model.embed_tokens`` in several forward paths. Wrapping the
        # language model in PeftModelForCausalLM adds another ``model`` level and
        # breaks that fixed contract. In-place injection preserves the original
        # Qwen2ForCausalLM structure while replacing only the audited Linear layers.
        self.model.showo = inject_adapter_in_model(lora_config, self.model.showo)
        if not hasattr(self.model.showo.model, "embed_tokens"):
            raise RuntimeError("In-place LoRA injection changed the Show-o2 embedding contract")
        if gradient_checkpointing and hasattr(
            self.model.showo, "gradient_checkpointing_enable"
        ):
            # Reentrant checkpointing requires a grad-bearing embedding input. The upstream
            # Show-o2 forward mutates that embedding tensor in-place while inserting image/time
            # tokens, which is illegal for a leaf that requires grad. Non-reentrant checkpointing
            # records the graph without that input requirement and remains compatible with LoRA.
            self.model.showo.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        trainable_names = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        invalid_trainables = [name for name in trainable_names if "lora_" not in name.lower()]
        if invalid_trainables:
            raise RuntimeError(
                f"Non-LoRA Show-o2 parameters became trainable: {invalid_trainables[:20]}"
            )
        self.model.train()
        self._lora_attached = True
        return {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": targets,
            "peft_relative_target_modules": peft_targets,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "trainable_names": trainable_names,
        }

    def generation_loss(self, batch: Showo2GenerationBatch):
        self._load()
        import torch

        size = len(batch.prompts)
        if not (size == len(batch.images) == len(batch.sample_ids)) or size == 0:
            raise ValueError("Show-o2 generation batch fields must be non-empty and equal length")
        prepare_gen_input = self._runtime["prepare_gen_input"]
        omni_attn_mask_naive = self._runtime["omni_attn_mask_naive"]
        clean = self._encode_images(batch.images, seed=batch.latent_seed)
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(int(batch.latent_seed))
            torch.cuda.manual_seed(int(batch.latent_seed))
            t, noise, target = self.transport.sample(clean)
            t, noised, velocity = self.transport.path_sampler.plan(t, noise, target)
        text, _, positions, _ = prepare_gen_input(
            list(batch.prompts),
            self.tokenizer,
            self.num_t2i_image_tokens,
            self.token_ids["bos_id"],
            self.token_ids["eos_id"],
            self.token_ids["boi_id"],
            self.token_ids["eoi_id"],
            int(self.tokenizer.pad_token_id),
            self.token_ids["img_pad_id"],
            self.max_text_len,
            self.device,
        )
        image_masks = (text == self.token_ids["img_pad_id"]).to(self.dtype)
        attention = omni_attn_mask_naive(text.size(0), text.size(1), positions, self.device).to(
            self.dtype
        )
        # LoRA dropout is active during training. Bind its random mask to the batch seed so
        # same-candidate gradient repeats are an exact control and criterion comparisons share
        # the same stochastic path. The surrounding fork prevents diagnostics from perturbing the
        # caller's global CUDA RNG state.
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(int(batch.latent_seed))
            torch.cuda.manual_seed(int(batch.latent_seed))
            _, loss_flow = self.model(
                text_tokens=text,
                image_latents=noised,
                t=t.to(self.dtype),
                attention_mask=attention,
                image_masks=image_masks,
                image_labels=velocity,
                modality_positions=positions,
                output_hidden_states=True,
                max_seq_len=text.size(1),
                device=self.device,
            )
        return loss_flow

    def understanding_replay_loss(self, batch: Showo2ReplayBatch):
        self._load()
        import torch

        size = len(batch.images)
        if (
            not (size == len(batch.questions) == len(batch.answers) == len(batch.sample_ids))
            or size == 0
        ):
            raise ValueError("Show-o2 replay fields must be non-empty and equal length")
        latents = self._encode_images(batch.images, seed=batch.latent_seed)
        pad_id = int(self.tokenizer.pad_token_id)
        rows = []
        labels = []
        positions = []
        for question, answer in zip(batch.questions, batch.answers):
            prefix = self.tokenizer(f"Question: {question}\nAnswer:", add_special_tokens=False)[
                "input_ids"
            ]
            answer_ids = self.tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
            row = [
                self.token_ids["bos_id"],
                self.token_ids["boi_id"],
                *([self.token_ids["img_pad_id"]] * self.num_mmu_image_tokens),
                self.token_ids["eoi_id"],
                *prefix,
                *answer_ids,
                self.token_ids["eos_id"],
            ]
            if len(row) > self.max_seq_len:
                raise ValueError("Show-o2 replay text exceeds the registered sequence length")
            answer_start = 2 + self.num_mmu_image_tokens + 1 + len(prefix)
            label = [-100] * len(row)
            label[answer_start:] = [*answer_ids, self.token_ids["eos_id"]]
            rows.append(row + [pad_id] * (self.max_seq_len - len(row)))
            labels.append(label + [-100] * (self.max_seq_len - len(label)))
            positions.append(((2, self.num_mmu_image_tokens),))
        text = torch.tensor(rows, dtype=torch.long, device=self.device)
        text_labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        modality_positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        attention = self._runtime["omni_attn_mask_naive"](
            text.size(0), text.size(1), modality_positions, self.device
        ).to(self.dtype)
        t = torch.ones((size,), device=self.device, dtype=self.dtype)
        _, loss_ntp = self.model(
            text_tokens=text,
            image_latents=latents,
            t=t,
            attention_mask=attention,
            text_labels=text_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text.size(1),
            device=self.device,
        )
        return loss_ntp

    def compute_lora_gradient(
        self,
        batch: Showo2GenerationBatch,
        criterion: str,
    ) -> GradientResult:
        self._load()
        if not self._lora_attached:
            raise RuntimeError("Attach audited Show-o2 LoRA targets before collecting gradients")
        snapshot = collect_lora_gradient(
            self.model,
            lambda: self.generation_loss(batch),
            criterion=criterion,
            sample_ids=batch.sample_ids,
        )
        return GradientResult(
            criterion=criterion,
            vector=snapshot.vector,
            per_block=snapshot.per_block,
            loss=snapshot.loss,
            sample_ids=snapshot.sample_ids,
        )

    def collect_gradient(self, batch: Showo2GenerationBatch, criterion: str) -> GradientResult:
        return self.compute_lora_gradient(batch, criterion)

    def compute_lora_gradient_accumulated(
        self,
        batches: Sequence[Showo2GenerationBatch],
        criterion: str,
    ) -> GradientResult:
        """Collect the mean LoRA gradient over deterministic Show-o2 microbatches."""

        snapshot = collect_lora_gradient_accumulated(
            self.model,
            [lambda batch=batch: self.generation_loss(batch) for batch in batches],
            criterion=criterion,
            sample_ids=(sample_id for batch in batches for sample_id in batch.sample_ids),
        )
        return GradientResult(
            criterion=criterion,
            vector=snapshot.vector,
            per_block=snapshot.per_block,
            loss=snapshot.loss,
            sample_ids=snapshot.sample_ids,
        )

    def save_adapter(self, destination: str | Path, **state: Any) -> Path:
        required = {
            "optimizer",
            "scheduler",
            "config_digest",
            "config_values",
            "step",
            "round_index",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Missing Show-o2 checkpoint state: {missing}")
        return save_checkpoint(
            destination,
            model=self.model,
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            config_digest=str(state["config_digest"]),
            config_values=dict(state["config_values"]),
            step=int(state["step"]),
            round_index=int(state["round_index"]),
            metadata=dict(state.get("metadata", {})),
        )

    def load_adapter(self, source: str | Path, **state: Any) -> Mapping[str, Any]:
        required = {"optimizer", "scheduler", "expected_config_digest"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Missing Show-o2 restore state: {missing}")
        return load_checkpoint(
            source,
            model=self.model,
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            expected_config_digest=str(state["expected_config_digest"]),
        )

    def resource_report(self) -> ResourceReport:
        if not self._loaded:
            return ResourceReport(
                device=self.device_spec,
                dtype=self.dtype_name,
                loaded=False,
                total_parameters=None,
                trainable_parameters=None,
                allocated_gpu_bytes=None,
                reserved_gpu_bytes=None,
            )
        import torch

        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return ResourceReport(
            device=str(self.device),
            dtype=self.dtype_name,
            loaded=True,
            total_parameters=total,
            trainable_parameters=trainable,
            allocated_gpu_bytes=int(torch.cuda.memory_allocated(self.device)),
            reserved_gpu_bytes=int(torch.cuda.memory_reserved(cuda_device_index(self.device))),
        )

    def dependency_revisions(self) -> dict[str, str]:
        lock = load_model_lock(self.lock_path)
        dependency_ids = tuple(str(item) for item in self.backbone_config["dependencies"])
        by_id = {str(item["id"]): str(item["revision"]) for item in lock["models"]}
        resolved = {model_id: by_id[model_id] for model_id in dependency_ids}
        expected = {
            str(key): str(value) for key, value in self.backbone_config["dependencies"].items()
        }
        if resolved != expected:
            raise RuntimeError("Show-o2 backbone dependency config/model lock mismatch")
        return resolved


__all__ = ["Showo2Adapter", "Showo2GenerationBatch", "Showo2ReplayBatch"]
