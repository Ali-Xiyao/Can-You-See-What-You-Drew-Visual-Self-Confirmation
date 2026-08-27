param(
    [switch]$InstallObservers,
    [switch]$InstallJanusObserver,
    [switch]$InstallFigure,
    [string]$CloneCudaEnv = "",
    [string]$CloneObserverCudaEnv = "",
    [string]$CloneJanusCudaEnv = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\set_h_env.ps1"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

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

$python310 = "D:\python\python.exe"
if (-not (Test-Path -LiteralPath $python310)) {
    throw "Python 3.10 was not found at $python310. Update the script with an explicit Python 3.10 path."
}

$coreEnv = Join-Path $env:SELFSIGHT_ENV_ROOT "core"
$corePython = Find-EnvironmentPython $coreEnv
if (-not $corePython) {
    if ($CloneCudaEnv) {
        $conda = "C:\Users\Admin\anaconda3\Scripts\conda.exe"
        if (-not (Test-Path -LiteralPath $conda)) {
            throw "Conda was not found at $conda"
        }
        & $conda create --yes --prefix $coreEnv --clone $CloneCudaEnv
    } else {
        & $python310 -m venv $coreEnv
    }
    $corePython = Find-EnvironmentPython $coreEnv
}
if (-not $corePython) { throw "No Python executable was created under $coreEnv" }
& $corePython -m pip install --upgrade pip setuptools wheel
& $corePython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
if ($LASTEXITCODE -ne 0) {
    & $corePython -m pip install --index-url https://download.pytorch.org/whl/cu121 `
        torch==2.2.1 torchvision==0.17.1
}
& $corePython -m pip install -e "${repoRoot}[dev,training]"
if ($InstallFigure) {
    & $corePython -m pip install -e "${repoRoot}[figure]"
}

if ($InstallObservers) {
    $observerEnv = Join-Path $env:SELFSIGHT_ENV_ROOT "observer"
    $observerPython = Find-EnvironmentPython $observerEnv
    if (-not $observerPython) {
        if ($CloneObserverCudaEnv) {
            $conda = "C:\Users\Admin\anaconda3\Scripts\conda.exe"
            & $conda create --yes --prefix $observerEnv --clone $CloneObserverCudaEnv
        } else {
            & $python310 -m venv $observerEnv
        }
        $observerPython = Find-EnvironmentPython $observerEnv
    }
    if (-not $observerPython) { throw "No Python executable was created under $observerEnv" }
    & $observerPython -m pip install --upgrade pip setuptools wheel
    & $observerPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
    if ($LASTEXITCODE -ne 0) {
        & $observerPython -m pip install --index-url https://download.pytorch.org/whl/cu121 `
            torch==2.5.1 torchvision==0.20.1
    }
    & $observerPython -m pip install -e "${repoRoot}[observer]"
    $janusRepo = Join-Path $env:SELFSIGHT_MODEL_ROOT "repositories\Janus"
    if (Test-Path -LiteralPath (Join-Path $janusRepo "pyproject.toml")) {
        & $observerPython -m pip install --no-deps -e $janusRepo
    } else {
        Write-Warning "Janus repository is not synced yet; run scripts/sync_repositories.py before its canary."
    }
}

if ($InstallJanusObserver) {
    $janusEnv = Join-Path $env:SELFSIGHT_ENV_ROOT "janus"
    $janusPython = Find-EnvironmentPython $janusEnv
    if (-not $janusPython) {
        if ($CloneJanusCudaEnv) {
            $conda = "C:\Users\Admin\anaconda3\Scripts\conda.exe"
            if (-not (Test-Path -LiteralPath $conda)) {
                throw "Conda was not found at $conda"
            }
            & $conda create --yes --prefix $janusEnv --clone $CloneJanusCudaEnv
        } else {
            & $python310 -m venv $janusEnv
        }
        $janusPython = Find-EnvironmentPython $janusEnv
    }
    if (-not $janusPython) { throw "No Python executable was created under $janusEnv" }
    & $janusPython -m pip install --upgrade pip setuptools wheel
    & $janusPython -c "import torch; from packaging.version import Version; assert torch.cuda.is_available() and Version(torch.__version__.split('+')[0]) >= Version('2.6'); print(torch.__version__, torch.version.cuda)"
    if ($LASTEXITCODE -ne 0) {
        & $janusPython -m pip install --index-url https://download.pytorch.org/whl/cu118 `
            torch==2.6.0 torchvision==0.21.0
    }
    & $janusPython -m pip install `
        numpy==1.26.4 Pillow==11.3.0 PyYAML==6.0.3 `
        transformers==4.57.6 accelerate==1.12.0 timm==1.0.22 `
        sentencepiece==0.2.2 attrdict3==2.0.2 einops==0.8.1 safetensors==0.7.0
    & $janusPython -m pip install --no-deps -e $repoRoot
    $janusRepo = Join-Path $env:SELFSIGHT_MODEL_ROOT "repositories\Janus"
    if (Test-Path -LiteralPath (Join-Path $janusRepo "pyproject.toml")) {
        & $janusPython -m pip install --no-deps -e $janusRepo
    } else {
        Write-Warning "Janus repository is not synced yet; run scripts/sync_repositories.py before its canary."
    }
}

$lockScript = Join-Path $repoRoot "scripts\capture_environment_lock.py"
if (Test-Path -LiteralPath $lockScript) {
    & $corePython $lockScript (Join-Path $env:SELFSIGHT_ENV_ROOT "locks\windows-core.json") `
        --role windows_core
    if ($InstallObservers) {
        & $observerPython $lockScript (Join-Path $env:SELFSIGHT_ENV_ROOT "locks\windows-observer.json") `
            --role windows_observer
    }
    if ($InstallJanusObserver) {
        & $janusPython $lockScript (Join-Path $env:SELFSIGHT_ENV_ROOT "locks\windows-janus.json") `
            --role windows_janus
    }
}

Write-Host "Windows environments are ready under $env:SELFSIGHT_ENV_ROOT"
