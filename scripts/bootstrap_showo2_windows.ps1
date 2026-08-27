param(
    [string]$CloneCudaEnv = "",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\set_h_env.ps1"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$showo2Env = Join-Path $env:SELFSIGHT_ENV_ROOT "showo2"

function Find-EnvironmentPython([string]$EnvironmentPath) {
    foreach ($candidate in @(
            (Join-Path $EnvironmentPath "python.exe"),
            (Join-Path $EnvironmentPath "Scripts\python.exe")
        )) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

$showo2Python = Find-EnvironmentPython $showo2Env
if (-not $showo2Python) {
    if (Test-Path -LiteralPath $showo2Env) {
        throw "Show-o2 environment path exists but has no Python; move it aside before retrying: $showo2Env"
    }
    if (-not $CloneCudaEnv) {
        $CloneCudaEnv = Join-Path $env:SELFSIGHT_ENV_ROOT "core"
    }
    $clonePython = Find-EnvironmentPython $CloneCudaEnv
    if (-not $clonePython) {
        throw "CUDA source environment has no Python: $CloneCudaEnv"
    }
    $conda = "C:\Users\Admin\anaconda3\Scripts\conda.exe"
    if (-not (Test-Path -LiteralPath $conda)) {
        throw "Conda is required to clone the working CUDA environment: $conda"
    }
    & $conda create --yes --prefix $showo2Env --clone $CloneCudaEnv
    if ($LASTEXITCODE -ne 0) { throw "Conda failed to clone the Show-o2 environment" }
    $showo2Python = Find-EnvironmentPython $showo2Env
}
if (-not $showo2Python) { throw "No Python executable is available under $showo2Env" }

if (-not $SkipInstall) {
    & $showo2Python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Failed to update Show-o2 packaging tools" }
    & $showo2Python -m pip install -e "${repoRoot}[showo2]"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the minimal Show-o2 dependency set" }
}

$showo2Source = Join-Path $env:SELFSIGHT_MODEL_ROOT "repositories\Show-o\show-o2"
if (-not (Test-Path -LiteralPath (Join-Path $showo2Source "models\modeling_showo2_qwen2_5.py"))) {
    throw "Locked Show-o2 source is missing. Run scripts/sync_repositories.py first."
}
$importCheck = @'
import pathlib
import sys
import torch
import transformers
import diffusers
from torch.nn.attention.flex_attention import BlockMask

source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import models
import transport

assert torch.cuda.is_available()
assert pathlib.Path(models.__file__).resolve().is_relative_to(source)
assert pathlib.Path(transport.__file__).resolve().is_relative_to(source)
print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
    "showo2_source": str(source),
})
'@
& $showo2Python -c $importCheck $showo2Source
if ($LASTEXITCODE -ne 0) { throw "Show-o2 import/CUDA canary failed" }

$lockScript = Join-Path $repoRoot "scripts\capture_environment_lock.py"
& $showo2Python $lockScript (Join-Path $env:SELFSIGHT_ENV_ROOT "locks\windows-showo2.json") `
    --role windows_showo2
if ($LASTEXITCODE -ne 0) { throw "Failed to capture the Show-o2 environment lock" }

Write-Host "Show-o2 Windows environment is ready: $showo2Env"
