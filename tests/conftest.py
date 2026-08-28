from __future__ import annotations

from pathlib import Path

import pytest

from selfsight.data.generator import build_splits


@pytest.fixture(scope="session")
def registered_splits():
    return build_splits(20260827)


@pytest.fixture(autouse=True)
def h_drive_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_root = tmp_path / "project"
    roots = {
        "SELFSIGHT_PROJECT_ROOT": str(project_root),
        "SELFSIGHT_CACHE_ROOT": str(project_root / "cache"),
        "SELFSIGHT_DATA_ROOT": str(project_root / "data"),
        "SELFSIGHT_RUN_ROOT": str(project_root / "runs"),
        "SELFSIGHT_MODEL_ROOT": "H:\\selfsight-models",
        "SELFSIGHT_ENV_ROOT": str(project_root / "envs"),
        "SELFSIGHT_TMP_ROOT": str(project_root / "tmp"),
    }
    for name, value in roots.items():
        monkeypatch.setenv(name, value)
