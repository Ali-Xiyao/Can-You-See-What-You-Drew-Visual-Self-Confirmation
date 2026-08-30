#!/usr/bin/env bash
# Git Bash equivalent of set_h_env.ps1, for driving the Windows interpreters from
# a POSIX shell. Emits Windows-style paths because the interpreters are native.
# Models are the single external exception; everything else is repository-local.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_POSIX="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_WIN="$(cygpath -w "${PROJECT_POSIX}")"

export SELFSIGHT_PROJECT_ROOT="${PROJECT_WIN}"
export SELFSIGHT_CACHE_ROOT="${PROJECT_WIN}\cache"
export SELFSIGHT_DATA_ROOT="${PROJECT_WIN}\data"
export SELFSIGHT_RUN_ROOT="${PROJECT_WIN}\runs"
export SELFSIGHT_ENV_ROOT="${PROJECT_WIN}\envs"
export SELFSIGHT_TMP_ROOT="${PROJECT_WIN}\tmp"
export SELFSIGHT_MODEL_ROOT="H:\selfsight-models"
export HF_HOME="${SELFSIGHT_CACHE_ROOT}\huggingface"
export HF_HUB_CACHE="${HF_HOME}\hub"
export TORCH_HOME="${SELFSIGHT_CACHE_ROOT}\torch"
export PIP_CACHE_DIR="${SELFSIGHT_CACHE_ROOT}\pip"
export TMP="${SELFSIGHT_TMP_ROOT}"
export TEMP="${SELFSIGHT_TMP_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
