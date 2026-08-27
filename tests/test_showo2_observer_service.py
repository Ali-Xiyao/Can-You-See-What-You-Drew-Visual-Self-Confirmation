from __future__ import annotations

import pytest

from selfsight.observers.service import _load_backend


def test_showo2_service_requires_explicit_backbone_config_before_loading() -> None:
    with pytest.raises(ValueError, match="backbone-config"):
        _load_backend(
            "showo2",
            "showlab/show-o2-1.5B",
            "locked",
            "cuda:1",
        )
