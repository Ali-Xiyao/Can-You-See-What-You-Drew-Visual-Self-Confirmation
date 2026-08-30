"""Directory-sourced observer weights must still be content-bound.

Qwen3-VL-8B-Instruct lives in a plain ModelScope directory outside
`SELFSIGHT_MODEL_ROOT`, so it has no HF revision to pin. The project's evidence
discipline still requires that every report binds an exact weight set, so these
observers carry a `dir-sha256:` digest instead.
"""

from __future__ import annotations

import json

import pytest

from selfsight.observers.transformers_vlm import (
    _external_local_path,
    _resolve_observer_path,
    directory_model_digest,
)


def _weights(root, *, shards=(("model-00001-of-00002.safetensors", 2048),)):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"]}), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0][0]}}), encoding="utf-8"
    )
    for name, size in shards:
        (root / name).write_bytes(b"\0" * size)
    return root


def test_digest_is_stable_and_order_independent(tmp_path):
    root = _weights(
        tmp_path / "m",
        shards=(("model-00001-of-00002.safetensors", 512), ("model-00002-of-00002.safetensors", 256)),
    )
    assert directory_model_digest(root) == directory_model_digest(root)


def test_digest_changes_when_a_shard_changes_size(tmp_path):
    root = _weights(tmp_path / "m", shards=(("model-00001-of-00001.safetensors", 512),))
    before = directory_model_digest(root)
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 513)
    assert directory_model_digest(root) != before


def test_digest_changes_when_config_changes(tmp_path):
    root = _weights(tmp_path / "m")
    before = directory_model_digest(root)
    (root / "config.json").write_text(json.dumps({"architectures": ["Other"]}), encoding="utf-8")
    assert directory_model_digest(root) != before


def test_missing_shards_fail_loudly(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No safetensors shards"):
        directory_model_digest(root)


def test_external_path_requires_the_env_var(tmp_path, monkeypatch):
    root = _weights(tmp_path / "m")
    monkeypatch.delenv("SELFSIGHT_EXTERNAL_MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="SELFSIGHT_EXTERNAL_MODEL_DIR"):
        _external_local_path("Qwen/Qwen3-VL-8B-Instruct", f"dir-sha256:{directory_model_digest(root)}")


def test_external_path_rejects_a_mismatched_digest(tmp_path, monkeypatch):
    root = _weights(tmp_path / "m")
    monkeypatch.setenv("SELFSIGHT_EXTERNAL_MODEL_DIR", str(root))
    with pytest.raises(ValueError, match="digest mismatch"):
        _external_local_path("Qwen/Qwen3-VL-8B-Instruct", "dir-sha256:" + "0" * 64)


def test_external_path_accepts_the_bound_digest(tmp_path, monkeypatch):
    root = _weights(tmp_path / "m")
    monkeypatch.setenv("SELFSIGHT_EXTERNAL_MODEL_DIR", str(root))
    revision = f"dir-sha256:{directory_model_digest(root)}"
    assert _external_local_path("Qwen/Qwen3-VL-8B-Instruct", revision) == root.resolve()
    assert _resolve_observer_path("Qwen/Qwen3-VL-8B-Instruct", revision) == root.resolve()


def test_directory_revision_must_be_prefixed(tmp_path, monkeypatch):
    root = _weights(tmp_path / "m")
    monkeypatch.setenv("SELFSIGHT_EXTERNAL_MODEL_DIR", str(root))
    with pytest.raises(ValueError, match="dir-sha256:"):
        _external_local_path("Qwen/Qwen3-VL-8B-Instruct", "abc123")


def test_thinking_variants_are_refused(monkeypatch):
    """Reasoning traces break short-answer normalization and repeat agreement."""

    from selfsight.observers.transformers_vlm import Qwen3VLObserver

    with pytest.raises(ValueError, match="Refusing a Thinking observer"):
        Qwen3VLObserver("Qwen/Qwen3-VL-8B-Thinking", "dir-sha256:" + "0" * 64, "cpu")
