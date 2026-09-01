# Unattended v3 queue, second pass: finish the difficulty sweep under the
# redesigned gold atoms, freeze the difficulty, then measure Gate A on held-out
# prompts the difficulty was not chosen on.
#
# Every step writes a `.done` marker under runs\v3\queue2, so re-running the
# script resumes instead of repeating GPU work. Nothing here makes a scientific
# decision that is not already registered in scripts\select_v3_difficulty.py.
#
#   powershell -NoProfile -File scripts\run_v3_queue2.ps1
#
# Two failure modes killed the first queue and are designed out here:
#
#   * `Receive-Job` re-emits a background job's native stderr as an ErrorRecord.
#     A harmless FutureWarning from transformers therefore terminated the script
#     before it could write a FAILED marker -- twice, at the same line. This
#     version uses Start-Process with redirected output and never deserializes a
#     job, so a warning on stderr cannot influence control flow.
#   * run_v3_bank_probe returns 2 when the gate is red, which is a result, not a
#     crash. The old `-ne 0` check would have aborted the queue on a legitimate
#     red gate. Bank-probe steps accept 0 and 2 and record the verdict instead.
#
param(
    [string]$Root = "H:\Xiyao_Wang\062_Can You See What You Drew Visual Self-Confirmation"
)

Set-Location $Root
$ErrorActionPreference = "Continue"

# A detached process inherits nothing from the interactive shell, and the
# backbone loader reads SELFSIGHT_MODEL_ROOT straight out of the environment.
. .\scripts\set_h_env.ps1 | Out-Null

$showo2 = "envs\showo2\python.exe"
$core = "envs\core\python.exe"
$queue = "runs\v3\queue2"
$log = Join-Path $queue "queue.log"
$settings = @(3, 4, 5, 6)
$families = @("existence", "spatial", "binding")
$calibrationManifest = "data\selfsight-v3-cal\obj{0}\manifests\calibration.jsonl"

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

# Launch without waiting. Output goes to files; nothing is piped back into the
# shell, so a subprocess warning can never become a terminating error here.
function StartStep([string]$exe, [string[]]$stepArgs, [string]$tag) {
    $out = Join-Path $queue "$tag.out"
    $err = Join-Path $queue "$tag.err"
    $process = Start-Process -FilePath $exe -ArgumentList $stepArgs -WorkingDirectory $Root `
        -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    # Touching Handle caches the process handle. Without it -PassThru can report
    # a null ExitCode after the process ends, which would read as success.
    $null = $process.Handle
    return $process
}

# Run to completion in the foreground and return the exit code.
function RunStep([string]$exe, [string[]]$stepArgs, [string]$tag) {
    $process = StartStep $exe $stepArgs $tag
    $process.WaitForExit()
    return $process.ExitCode
}

function TailLog([string]$tag, [int]$lines = 6) {
    foreach ($suffix in @("out", "err")) {
        $path = Join-Path $queue "$tag.$suffix"
        if ((Test-Path $path) -and (Get-Item $path).Length -gt 0) {
            Get-Content $path -Tail $lines | ForEach-Object { Say "    $tag.${suffix} | $_" }
        }
    }
}

Say "=== queue2 start (pid $PID) ==="

# ------------------------------------------- step 1: finish the sweep at obj4
# obj3/5/6 already have banks on disk. Only obj4 was never generated -- the
# first queue died before reaching it. The sweep set {3,4,5,6} was registered
# before any of these numbers were seen, so it gets completed rather than
# trimmed to what happens to exist.
if (-not (IsDone "obj4_bank")) {
    Say "step 1  generating the obj4 calibration bank (48 prompts, split across 2 cards)"
    $records = [string]::Format($calibrationManifest, 4)
    $shards = @(
        @{ tag = "obj4-gpu0"; device = "cuda:0"; offset = 0; limit = 24; out = "runs\v3\calib\obj4\gpu0" },
        @{ tag = "obj4-gpu1"; device = "cuda:1"; offset = 24; limit = 24; out = "runs\v3\calib\obj4\gpu1" }
    )
    $running = @()
    foreach ($shard in $shards) {
        $running += @{
            tag = $shard.tag
            process = StartStep $showo2 @(
                "scripts\run_v3_bank_probe.py",
                "--records", $records,
                "--output", $shard.out,
                "--device", $shard.device,
                "--offset", "$($shard.offset)",
                "--limit", "$($shard.limit)",
                "--rate-only"
            ) $shard.tag
        }
        Say "  launched $($shard.tag) on $($shard.device) (records $($shard.offset)..$($shard.offset + $shard.limit - 1))"
    }
    foreach ($item in $running) {
        $item.process.WaitForExit()
        TailLog $item.tag
        # 2 means the gate is red on this shard, which is a measurement.
        if ($item.process.ExitCode -notin @(0, 2)) {
            Fail "obj4_bank" "$($item.tag) exit $($item.process.ExitCode)"
        }
        Say "  $($item.tag) finished (exit $($item.process.ExitCode))"
    }
    Say "step 1  done"
    SetDone "obj4_bank"
}

# ------------------------------------------------- step 2: score every bank
# One scoring path for all four settings, including the three that were
# generated before the redesign. Images depend only on the prompt, which did not
# change; only the gold atoms did. Writes to calib-v2, never back into calib,
# because run_bank_probe resumes on scene_id and would reuse the old scores.
if (-not (IsDone "rescore")) {
    Say "step 2  scoring the sweep under the redesigned gold atoms"
    $rescoreArgs = @(
        "scripts\rescore_v3_calibration.py",
        "--source", "runs\v3\calib",
        "--records-root", "data\selfsight-v3-cal",
        "--output", "runs\v3\calib-v2"
    )
    foreach ($n in $settings) { $rescoreArgs += @("--setting", "$n") }
    $code = RunStep $core $rescoreArgs "rescore"
    TailLog "rescore" 10
    if ($code -ne 0) { Fail "rescore" "exit $code" }
    Say "step 2  done"
    SetDone "rescore"
}

# ------------------------------------------------------ step 3: freeze difficulty
if (-not (IsDone "select_difficulty")) {
    Say "step 3  applying the registered difficulty rule"
    $selectArgs = @(
        "scripts\select_v3_difficulty.py",
        "--calib-root", "runs\v3\calib-v2",
        "--output", "runs\v3\calib-v2\calibration_report.json"
    )
    foreach ($n in $settings) { $selectArgs += @("--setting", "$n") }
    $code = RunStep $core $selectArgs "select"
    TailLog "select" 10
    if ($code -ne 0) { Fail "select_difficulty" "exit $code" }
    Say "step 3  done"
    SetDone "select_difficulty"
}

$report = Get-Content "runs\v3\calib-v2\calibration_report.json" -Raw | ConvertFrom-Json

# ---------------------------------------------- step 4: held-out splits, frozen
# Each family is built at its own chosen object count. The calibration manifest
# at that count is forbidden explicitly: splits are only disjoint within one
# invocation, and calibration was built separately.
if (-not (IsDone "build_heldout")) {
    Say "step 4  building held-out splits at the frozen difficulty"
    foreach ($family in $families) {
        $entry = $report.chosen.$family
        if (-not $entry -or -not $entry.objects_per_scene) {
            Fail "build_heldout" "no difficulty chosen for $family"
        }
        $n = $entry.objects_per_scene
        Say "  $family -> objects_per_scene=$n (p=$($entry.p), status=$($entry.status))"
        $code = RunStep $core @(
            "scripts\build_v3_scenes.py",
            "--output-root", "data\selfsight-v3\$family",
            "--objects-per-scene", "$n",
            "--family", $family,
            "--forbid", [string]::Format($calibrationManifest, $n),
            "--split", "tier_a_probe=64",
            "--split", "train=200",
            "--split", "tier_a_outcome=270"
        ) "heldout-$family"
        TailLog "heldout-$family" 4
        if ($code -ne 0) { Fail "build_heldout" "$family exit $code" }
    }
    Say "step 4  done"
    SetDone "build_heldout"
}

# ------------------------------------------------- step 5: Gate A, held out
# Both cards work on the same family at once and the families run in turn, so a
# card is never idle waiting for a longer family to finish.
if (-not (IsDone "gate_a")) {
    Say "step 5  Gate A on the held-out probe split"
    foreach ($family in $families) {
        $records = "data\selfsight-v3\$family\manifests\tier_a_probe.jsonl"
        $shards = @(
            @{ tag = "gatea-$family-gpu0"; device = "cuda:0"; offset = 0; limit = 32 },
            @{ tag = "gatea-$family-gpu1"; device = "cuda:1"; offset = 32; limit = 32 }
        )
        $running = @()
        foreach ($shard in $shards) {
            $card = $shard.tag.Substring($shard.tag.Length - 4)
            $running += @{
                tag = $shard.tag
                process = StartStep $showo2 @(
                    "scripts\run_v3_bank_probe.py",
                    "--records", $records,
                    "--output", "runs\v3\gate-a-v2\$family\$card",
                    "--device", $shard.device,
                    "--offset", "$($shard.offset)",
                    "--limit", "$($shard.limit)",
                    "--rate-only"
                ) $shard.tag
            }
            Say "  launched $($shard.tag) on $($shard.device)"
        }
        foreach ($item in $running) {
            $item.process.WaitForExit()
            TailLog $item.tag
            if ($item.process.ExitCode -notin @(0, 2)) {
                Fail "gate_a" "$($item.tag) exit $($item.process.ExitCode)"
            }
            Say "  $($item.tag) finished (exit $($item.process.ExitCode))"
        }
    }
    Say "step 5  done"
    SetDone "gate_a"
}

# ---------------------------------------------------- step 6: merge the shards
# Per-shard reports cover 32 prompts each and are measured against a prorated
# floor. The merged report is the one that counts, and it is produced by the
# same scorer as the calibration so the two are directly comparable.
if (-not (IsDone "merge_gate_a")) {
    Say "step 6  merging Gate A shards"
    foreach ($family in $families) {
        $code = RunStep $core @(
            "scripts\rescore_v3_calibration.py",
            "--source", "runs\v3\gate-a-v2\$family",
            "--records", "data\selfsight-v3\$family\manifests\tier_a_probe.jsonl",
            "--output", "runs\v3\gate-a-v2\$family"
        ) "merge-$family"
        TailLog "merge-$family" 4
        if ($code -ne 0) { Fail "merge_gate_a" "$family exit $code" }
    }
    Say "step 6  done"
    SetDone "merge_gate_a"
}

# ------------------------------------------------------------- step 7: summary
Say "step 7  writing summary"
$code = RunStep $core @(
    "scripts\summarize_v3_queue.py", "--output", (Join-Path $queue "SUMMARY.md")
) "summary"
TailLog "summary" 40
if ($code -ne 0) { Say "summary failed with exit $code (results are still on disk)" }

Say "=== queue2 finished ==="
