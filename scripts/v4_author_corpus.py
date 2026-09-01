"""Author the v4 scene corpus with a local LLM.

v3 built its prompts from templates over a fixed shape/colour vocabulary. That
gave exact control at the cost of external validity: the claim is about image
generation, and the stimuli were shape tests nobody would ever ask a generator
for. Here the scenes are written instead, and every one is checked against
`selfsight.v4.tasks` before it is kept -- the prompt must state exactly what the
spec claims, or there is nothing to verify the image against.

The model is used only as an author. It never scores an image; that is the
verifier's job and a different model does it.

    envs/observer/python.exe scripts/v4_author_corpus.py \
        --total 240 --out data/v4/corpus.jsonl --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from selfsight.v4.tasks import distribution_report, generate

DEFAULT_MODEL = r"H:\Xiyao_Wang\001_models\Qwen3-VL-8B-Instruct"


def build_asker(model_path: str, device: str, temperature: float):
    """Return a `str -> str` callable, which is all `tasks.generate` needs."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        chat = processor.apply_chat_template
    except Exception:  # a text-only checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        chat = tokenizer.apply_chat_template
    model.eval()

    def ask(prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = chat(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                # Sampling on purpose. Greedy decoding returns the same handful of
                # scenes batch after batch, and the duplicate filter then starves
                # the quota rather than filling it.
                do_sample=temperature > 0,
                temperature=temperature or None,
                top_p=0.95,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply

    return ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=240)
    parser.add_argument("--items", type=int, default=2,
                        help="distinct object entries per scene; 2 gives 2+1, 3 gives 1+1+1")
    parser.add_argument("--with-relations", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--prefix", default="v4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    ask = build_asker(args.model, args.device, args.temperature)
    specs, rejected = generate(
        ask,
        total=args.total,
        items=args.items,
        with_relations=args.with_relations,
        prefix=args.prefix,
        batch=args.batch,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")

    report = distribution_report(specs)
    report["rejected"] = len(rejected)
    report["reject_reasons"] = {}
    for item in rejected:
        key = str(item.get("reason", "?")).split(":")[0]
        report["reject_reasons"][key] = report["reject_reasons"].get(key, 0) + 1
    report["elapsed_s"] = round(time.time() - started, 1)
    report["author_model"] = args.model
    report["temperature"] = args.temperature

    side = args.out.with_suffix(".report.json")
    side.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
