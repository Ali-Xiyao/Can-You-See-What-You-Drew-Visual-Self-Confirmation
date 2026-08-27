$ErrorActionPreference = "Stop"

$roots = @{
    SELFSIGHT_CACHE_ROOT = "H:\selfsight-cache"
    SELFSIGHT_DATA_ROOT = "H:\selfsight-data"
    SELFSIGHT_RUN_ROOT = "H:\selfsight-runs"
    SELFSIGHT_MODEL_ROOT = "H:\selfsight-models"
    SELFSIGHT_ENV_ROOT = "H:\selfsight-envs"
    SELFSIGHT_TMP_ROOT = "H:\selfsight-tmp"
}

foreach ($entry in $roots.GetEnumerator()) {
    New-Item -ItemType Directory -Force -Path $entry.Value | Out-Null
    Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
}

$env:HF_HOME = Join-Path $env:SELFSIGHT_CACHE_ROOT "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:TORCH_HOME = Join-Path $env:SELFSIGHT_CACHE_ROOT "torch"
$env:PIP_CACHE_DIR = Join-Path $env:SELFSIGHT_CACHE_ROOT "pip"
$env:WANDB_DIR = Join-Path $env:SELFSIGHT_RUN_ROOT "wandb"
$env:WANDB_CACHE_DIR = Join-Path $env:SELFSIGHT_CACHE_ROOT "wandb"
$env:TEMP = $env:SELFSIGHT_TMP_ROOT
$env:TMP = $env:SELFSIGHT_TMP_ROOT
$env:PYTHONUTF8 = "1"
$env:PYTHONNOUSERSITE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

foreach ($path in @($env:HF_HOME, $env:HF_HUB_CACHE, $env:TRANSFORMERS_CACHE,
        $env:TORCH_HOME, $env:PIP_CACHE_DIR, $env:WANDB_DIR, $env:WANDB_CACHE_DIR)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

Write-Host "SelfSight paths are active for this PowerShell session:"
$roots.GetEnumerator() | Sort-Object Key | Format-Table Key, Value -AutoSize
