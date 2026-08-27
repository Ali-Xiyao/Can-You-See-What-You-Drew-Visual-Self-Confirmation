"""Small CUDA compatibility helpers shared by Windows and Linux runners."""

from __future__ import annotations

from typing import Any


def cuda_device_index(device: Any) -> int:
    """Return the integer CUDA index expected by low-level memory-stat APIs."""

    if isinstance(device, bool):
        raise TypeError(f"Invalid CUDA device: {device!r}")
    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"Invalid CUDA device index: {device}")
        return device
    spec = str(device).strip().lower()
    if spec == "cuda":
        import torch

        return int(torch.cuda.current_device())
    if spec.startswith("cuda:"):
        raw_index = spec.partition(":")[2]
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device!r}") from exc
        if index < 0:
            raise ValueError(f"Invalid CUDA device index: {index}")
        return index
    raise ValueError(f"Expected a CUDA device, got: {device!r}")


def reset_cuda_peak_memory_stats(device: Any) -> int:
    """Initialize the selected context, reset its peak counter, and return its index."""

    import torch

    index = cuda_device_index(device)
    torch.cuda.set_device(index)
    torch.empty(0, device=f"cuda:{index}")
    torch.cuda.reset_peak_memory_stats(index)
    return index
