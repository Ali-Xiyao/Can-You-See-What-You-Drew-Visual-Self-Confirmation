#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/set_a800_env.sh"

python3.10 -m venv "${SELFSIGHT_ENV_ROOT}/core"
"${SELFSIGHT_ENV_ROOT}/core/bin/python" -m pip install --upgrade pip setuptools wheel
"${SELFSIGHT_ENV_ROOT}/core/bin/python" -m pip install \
  torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
"${SELFSIGHT_ENV_ROOT}/core/bin/python" -m pip install -e '.[dev,training,figure]'

python3.10 -m venv "${SELFSIGHT_ENV_ROOT}/observer"
"${SELFSIGHT_ENV_ROOT}/observer/bin/python" -m pip install --upgrade pip setuptools wheel
"${SELFSIGHT_ENV_ROOT}/observer/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"${SELFSIGHT_ENV_ROOT}/observer/bin/python" -m pip install -e '.[observer]'

python3.10 -m venv "${SELFSIGHT_ENV_ROOT}/showo2"
"${SELFSIGHT_ENV_ROOT}/showo2/bin/python" -m pip install --upgrade pip setuptools wheel
"${SELFSIGHT_ENV_ROOT}/showo2/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"${SELFSIGHT_ENV_ROOT}/showo2/bin/python" -m pip install -e '.[showo2]'

# Janus ships PyTorch .bin weights. Modern Transformers refuses to load them with
# torch<2.6 because of CVE-2025-32434, so keep this audit-only backend isolated.
python3.10 -m venv "${SELFSIGHT_ENV_ROOT}/janus"
"${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip install --upgrade pip setuptools wheel
"${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
"${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip install \
  numpy==1.26.4 Pillow==11.3.0 PyYAML==6.0.3 \
  transformers==4.57.6 accelerate==1.12.0 timm==1.0.22 \
  sentencepiece==0.2.2 attrdict3==2.0.2 einops==0.8.1 safetensors==0.7.0
"${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip install --no-deps -e .

if [[ -f "${SELFSIGHT_MODEL_ROOT}/repositories/Janus/pyproject.toml" ]]; then
  "${SELFSIGHT_ENV_ROOT}/observer/bin/python" -m pip install --no-deps -e \
    "${SELFSIGHT_MODEL_ROOT}/repositories/Janus"
  "${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip install --no-deps -e \
    "${SELFSIGHT_MODEL_ROOT}/repositories/Janus"
fi

"${SELFSIGHT_ENV_ROOT}/core/bin/python" -m pip freeze > \
  "${SELFSIGHT_ENV_ROOT}/core.freeze.txt"
"${SELFSIGHT_ENV_ROOT}/observer/bin/python" -m pip freeze > \
  "${SELFSIGHT_ENV_ROOT}/observer.freeze.txt"
"${SELFSIGHT_ENV_ROOT}/showo2/bin/python" -m pip freeze > \
  "${SELFSIGHT_ENV_ROOT}/showo2.freeze.txt"
"${SELFSIGHT_ENV_ROOT}/janus/bin/python" -m pip freeze > \
  "${SELFSIGHT_ENV_ROOT}/janus.freeze.txt"

echo "A800 core, Show-o2, and observer environments created. Run the 32-prompt migration canary before any formal seed."
