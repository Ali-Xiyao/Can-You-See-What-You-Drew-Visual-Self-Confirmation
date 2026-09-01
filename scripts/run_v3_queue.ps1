# Unattended v3 queue: finish the box probe, measure headroom, calibrate the
# difficulty knob on a held-out-safe calibration split, freeze it, then re-run
# Gate A on data the difficulty was NOT chosen on.
#
# Every step writes a `.done` marker under runs\v3\queue, so re-running the
# script resumes instead of repeating GPU work. Nothing here makes a scientific
# decision that is not already registered in scripts\select_v3_difficulty.py.
#
#   powershell -NoProfile -File scripts\run_v3_queue.ps1
#
param(
    [string]$Root = "H:\Xiyao_Wang\062_Can You See What You Drew Visual Self-Confirmation"
)

Set-Location $Root
# Continue, not Stop. PS 5.1 turns a native exe's stderr into ErrorRecords, so a
# harmless FutureWarning from transformers becomes a terminating error and kills an
# unattended run without writing a marker. Exit codes are the mechanism that
# actually reports native failure, and every step below checks $LASTEXITCODE.
$ErrorActionPreference = "Continue"

# A detached process inherits nothing from the interactive shell, and the
# backbone loader reads SELFSIGHT_MODEL_ROOT straight out of the environment.
. .\scripts\set_h_env.ps1 | Out-Null

$showo2 = Join-Path $Root "envs\showo2\python.exe"
$core = Join-Path $Root "envs\core\python.exe"
$queue = Join-Path $Root "runs\v3\queue"
$log = Join-Path $queue "queue.log"
$settings = @(3, 4, 5, 6)
$families = @("existence", "spatial", "binding")

New-Item -ItemType Directory -Force -Path $queue | Out-Null

function Say([string]$message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $message
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function IsDone([string]$name) { Test-Path (Join-Path $queue "$name.done") }
function SetDone([string]$name) {
    Set-Content -Path (Join-Path $queue "$name.done") -Value (Get-Date -Format "o") -Encoding utf8
}

function Fail([string]$name, [string]$message) {
    Say "FAILED at $name : $message"
    Set-Content -Path (Join-Path $queue "FAILED") -Value "$name : $message" -Encoding utf8
    exit 1
}

Say "=== queue start (pid $PID) ==="

# ---------------------------------------------------------------- step 0: wait
# The spatial shard of the pool-box probe was still generating when this queue
# was written. Wait for its Gate A report rather than racing it.
if (-not (IsDone "wait_pool_box")) {
    Say "step 0  waiting for runs\v3\pool-box\gpu1\gate_a.json"
    $deadline = (Get-Date).AddMinutes(120)
    while (-not (Test-Path "runs\v3\pool-box\gpu1\gate_a.json")) {
        if ((Get-Date) -gt $deadline) { Fail "wait_pool_box" "gpu1 did not finish within 120 min" }
        Start-Sleep -Seconds 60
    }
    Say "step 0  done"
    SetDone "wait_pool_box"
}

# ------------------------------------------------------- step 1: headroom
# The registered pre-check: Oracle@K - Naive@K on the current (2-object, box
# vocabulary) design. `--score-cycle` is what the voided stage-1 run was missing;
# without it naive_cycle_rate is null and the headroom is never actually measured.
if (-not (IsDone "headroom")) {
    Say "step 1  selection headroom on pool-box (with --score-cycle)"
    & $showo2 scripts\run_v3_selection_headroom.py `
        --packets runs\v3\pool-box\gpu0\packets `
        --packets runs\v3\pool-box\gpu1\packets `
        --records data\selfsight-v3-pool\manifests\tier_a_probe.jsonl `
        --output runs\v3\selection-headroom\box `
        --score-cycle --device cuda:0
    if ($LASTEXITCODE -ne 0) { Fail "headroom" "exit $LASTEXITCODE" }
    Say "step 1  done"
    SetDone "headroom"
}

# ------------------------------------------------- step 2: calibration scenes
# One calibration manifest per object count, all three families in each. These
# scenes are excluded from every later split by build_v3_scenes' signature guard.
if (-not (IsDone "build_calibration")) {
    Say "step 2  building calibration scenes for object counts $($settings -join ', ')"
    foreach ($n in $settings) {
        & $core scripts\build_v3_scenes.py `
            --output-root "data\selfsight-v3-cal\obj$n" `
            --objects-per-scene $n `
            --split calibration=48
        if ($LASTEXITCODE -ne 0) { Fail "build_calibration" "objects_per_scene=$n exit $LASTEXITCODE" }
    }
    Say "step 2  done"
    SetDone "build_calibration"
}

# ----------------------------------------------------- step 3: calibration probe
# Two independent workers, not a memory pool: cuda:0 takes the low settings and
# cuda:1 the high ones, so both cards finish at roughly the same time.
if (-not (IsDone "probe_calibration")) {
    Say "step 3  bank probe over the calibration sweep (2 cards in parallel)"
    $work = {
        param($root, $python, $counts, $device, $tag)
        $ErrorActionPreference = "Continue"
        Set-Location $root
        foreach ($n in $counts) {
            & $python scripts\run_v3_bank_probe.py `
                --records "data\selfsight-v3-cal\obj$n\manifests\calibration.jsonl" `
                --output "runs\v3\calib\obj$n\$tag" `
                --device $device --rate-only
            if ($LASTEXITCODE -ne 0) { throw "objects_per_scene=$n on $device exit $LASTEXITCODE" }
        }
    }
    $low = Start-Job -ScriptBlock $work -ArgumentList $Root, $showo2, @(3, 4), "cuda:0", "gpu0"
    $high = Start-Job -ScriptBlock $work -ArgumentList $Root, $showo2, @(5, 6), "cuda:1", "gpu1"
    Wait-Job $low, $high | Out-Null
    Receive-Job $low | ForEach-Object { Say "  gpu0 | $_" }
    Receive-Job $high | ForEach-Object { Say "  gpu1 | $_" }
    $bad = @($low, $high) | Where-Object { $_.State -ne "Completed" }
    Remove-Job $low, $high
    if ($bad) { Fail "probe_calibration" "a shard did not complete" }
    Say "step 3  done"
    SetDone "probe_calibration"
}

# ------------------------------------------------------ step 4: freeze difficulty
if (-not (IsDone "select_difficulty")) {
    Say "step 4  applying the registered difficulty rule"
    $selectArgs = @("scripts\select_v3_difficulty.py", "--calib-root", "runs\v3\calib", "--output", "runs\v3\calib\calibration_report.json")
    foreach ($n in $settings) { $selectArgs += @("--setting", "$n") }
    & $core $selectArgs
    if ($LASTEXITCODE -ne 0) { Fail "select_difficulty" "exit $LASTEXITCODE" }
    Say "step 4  done"
    SetDone "select_difficulty"
}

$report = Get-Content "runs\v3\calib\calibration_report.json" -Raw | ConvertFrom-Json

# ---------------------------------------------- step 5: held-out splits, frozen
# Each family is built at its own chosen object count, so the difficulty knob is
# per family. Splits inside one invocation are signature-disjoint.
if (-not (IsDone "build_heldout")) {
    Say "step 5  building held-out splits at the frozen difficulty"
    foreach ($family in $families) {
        $entry = $report.chosen.$family
        if (-not $entry -or -not $entry.objects_per_scene) { Fail "build_heldout" "no difficulty chosen for $family" }
        $n = $entry.objects_per_scene
        Say "  $family -> objects_per_scene=$n (p=$($entry.p), status=$($entry.status))"
        & $core scripts\build_v3_scenes.py `
            --output-root "data\selfsight-v3\$family" `
            --objects-per-scene $n `
            --family $family `
            --split tier_a_probe=64 --split train=200 --split tier_a_outcome=270
        if ($LASTEXITCODE -ne 0) { Fail "build_heldout" "$family exit $LASTEXITCODE" }
    }
    Say "step 5  done"
    SetDone "build_heldout"
}

# --------------------------------------------------- step 6: Gate A, held out
# Measured on prompts the difficulty was not chosen on. One family per card is
# not possible with three families, so each card takes whole families in turn.
if (-not (IsDone "gate_a")) {
    Say "step 6  Gate A on the held-out probe split"
    $work = {
        param($root, $python, $names, $device, $tag)
        $ErrorActionPreference = "Continue"
        Set-Location $root
        foreach ($family in $names) {
            & $python scripts\run_v3_bank_probe.py `
                --records "data\selfsight-v3\$family\manifests\tier_a_probe.jsonl" `
                --output "runs\v3\gate-a-v2\$family" `
                --device $device --rate-only
            if ($LASTEXITCODE -ne 0) { throw "$family on $device exit $LASTEXITCODE" }
        }
    }
    $a = Start-Job -ScriptBlock $work -ArgumentList $Root, $showo2, @("existence", "binding"), "cuda:0", "gpu0"
    $b = Start-Job -ScriptBlock $work -ArgumentList $Root, $showo2, @("spatial"), "cuda:1", "gpu1"
    Wait-Job $a, $b | Out-Null
    Receive-Job $a | ForEach-Object { Say "  gpu0 | $_" }
    Receive-Job $b | ForEach-Object { Say "  gpu1 | $_" }
    $bad = @($a, $b) | Where-Object { $_.State -ne "Completed" }
    Remove-Job $a, $b
    if ($bad) { Fail "gate_a" "a shard did not complete" }
    Say "step 6  done"
    SetDone "gate_a"
}

# ------------------------------------------------------------- step 7: summary
Say "step 7  writing summary"
& $core scripts\summarize_v3_queue.py --output (Join-Path $queue "SUMMARY.md")
if ($LASTEXITCODE -ne 0) { Say "summary failed with exit $LASTEXITCODE (results are still on disk)" }

Say "=== queue finished ==="
