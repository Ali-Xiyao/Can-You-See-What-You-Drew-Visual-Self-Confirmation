from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ACTIVE_ROOT = re.compile(
    r"H:\\selfsight-(?:cache|data|envs|runs|tmp)", re.IGNORECASE
)


def test_active_files_do_not_hardcode_external_non_model_roots() -> None:
    candidates = [PROJECT_ROOT / "README.md"]
    for relative_root in ("src", "scripts", "configs", "docs"):
        root = PROJECT_ROOT / relative_root
        candidates.extend(path for path in root.rglob("*") if path.is_file())

    allowed = {PROJECT_ROOT / "scripts" / "migrate_project_roots.ps1"}
    violations: list[str] = []
    for path in candidates:
        if (
            path in allowed
            or any(part.endswith(".egg-info") for part in path.parts)
            or path.suffix.lower() in {".png", ".pdf", ".pyc"}
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_ACTIVE_ROOT.search(text):
            violations.append(str(path.relative_to(PROJECT_ROOT)))
    assert violations == []


def test_windows_environment_is_repository_rooted_except_models() -> None:
    script = (PROJECT_ROOT / "scripts" / "set_h_env.ps1").read_text(encoding="utf-8")
    assert "SELFSIGHT_PROJECT_ROOT = $projectRoot" in script
    for leaf in ("cache", "data", "runs", "envs", "tmp"):
        assert f'Join-Path $projectRoot "{leaf}"' in script
    assert 'SELFSIGHT_MODEL_ROOT = "H:\\selfsight-models"' in script
