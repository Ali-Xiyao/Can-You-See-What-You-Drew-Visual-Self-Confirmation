"""Long-lived JSONL observer subprocess entry point."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from selfsight.observers.protocol import decode_request, encode_result, execute_request
from selfsight.utils.jsonl import atomic_write_json


def _load_backend(name: str, model_id: str | None, revision: str | None, device: str):
    if name == "mock":
        from selfsight.observers.mock import MockPixelObserver

        return MockPixelObserver()
    if not model_id or not revision:
        raise ValueError("A pinned --model-id and --revision are required for a real observer")
    if name in {"showo", "showo_discrete", "janus"}:
        from selfsight.observers.unified import (
            DiscreteShowoBlindObserver,
            JanusProObserver,
            ShowoBlindObserver,
        )

        backends = {
            "showo": ShowoBlindObserver,
            "showo_discrete": DiscreteShowoBlindObserver,
            "janus": JanusProObserver,
        }
        backend = backends[name]
        return backend(model_id, revision, device)
    from selfsight.observers.transformers_vlm import create_transformers_observer

    return create_transformers_observer(name, model_id, revision, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "mock",
            "showo",
            "showo_discrete",
            "janus",
            "smolvlm",
            "qwen2vl",
            "internvl",
        ),
        required=True,
    )
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ready-report", type=Path)
    args = parser.parse_args()
    load_started = perf_counter()
    # Third-party remote-code models sometimes print informational text to stdout.
    # Stdout is reserved exclusively for the JSONL protocol, so quarantine it.
    with redirect_stdout(sys.stderr):
        observer = _load_backend(args.backend, args.model_id, args.revision, args.device)
    if args.ready_report:
        report = {
            "schema_version": 1,
            "backend": args.backend,
            "observer_id": observer.observer_id,
            "observer_revision": observer.revision,
            "device": args.device,
            "load_seconds": perf_counter() - load_started,
            "ready_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            import torch

            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                index = torch.device(args.device).index or 0
                report.update(
                    {
                        "gpu_name": torch.cuda.get_device_name(index),
                        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(index)),
                    }
                )
        except ImportError:
            pass
        atomic_write_json(args.ready_report, report)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = decode_request(line)
            with redirect_stdout(sys.stderr):
                result = execute_request(observer, request)
            response = {"ok": True, "result": json.loads(encode_result(result))}
        except Exception as exc:  # noqa: BLE001 - JSONL boundary must return backend failures.
            response = {
                "ok": False,
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
