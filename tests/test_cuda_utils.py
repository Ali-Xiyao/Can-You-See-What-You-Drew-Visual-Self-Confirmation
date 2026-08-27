from __future__ import annotations

import pytest

from selfsight.utils.cuda import cuda_device_index


def test_cuda_device_index_normalizes_explicit_devices() -> None:
    assert cuda_device_index(0) == 0
    assert cuda_device_index("cuda:0") == 0
    assert cuda_device_index("CUDA:1") == 1


@pytest.mark.parametrize("device", [-1, "cpu", "cuda:-1", "cuda:nope"])
def test_cuda_device_index_rejects_invalid_values(device: object) -> None:
    with pytest.raises(ValueError):
        cuda_device_index(device)


def test_cuda_device_index_rejects_boolean_as_type_error() -> None:
    with pytest.raises(TypeError):
        cuda_device_index(True)
