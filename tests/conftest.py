from __future__ import annotations

from pathlib import Path

import pytest

from selfsight.data.generator import build_splits


@pytest.fixture(scope="session")
def registered_splits():
    return build_splits(20260827)


@pytest.fixture(autouse=True)
def h_drive_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    roots = {
        "SELFSIGHT_CACHE_ROOT": "H:\\selfsight-cache",
        "SELFSIGHT_DATA_ROOT": "H:\\selfsight-data",
        "SELFSIGHT_RUN_ROOT": "H:\\selfsight-runs",
        "SELFSIGHT_MODEL_ROOT": "H:\\selfsight-models",
        "SELFSIGHT_ENV_ROOT": "H:\\selfsight-envs",
        "SELFSIGHT_TMP_ROOT": "H:\\selfsight-tmp",
    }
    for name, value in roots.items():
        monkeypatch.setenv(name, value)
