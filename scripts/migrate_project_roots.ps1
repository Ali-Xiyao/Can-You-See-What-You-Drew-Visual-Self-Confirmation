param(
    [string]$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..")),
    [bool]$CreateLegacyJunctions = $true
)

$ErrorActionPreference = "Stop"

$projectPath = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd("\")
$projectPrefix = $projectPath + "\"
$modelRoot = "H:\selfsight-models"
$mapping = [ordered]@{
    "H:\selfsight-cache" = Join-Path $projectPath "cache"
    "H:\selfsight-data" = Join-Path $projectPath "data"
    "H:\selfsight-envs" = Join-Path $projectPath "envs"
    "H:\selfsight-runs" = Join-Path $projectPath "runs"
    "H:\selfsight-tmp" = Join-Path $projectPath "tmp"
}

if (-not (Test-Path -LiteralPath $projectPath -PathType Container)) {
    throw "Project root does not exist: $projectPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $projectPath ".git") -PathType Container)) {
    throw "Refusing migration because the target is not the SelfSight Git project: $projectPath"
}
if (-not (Test-Path -LiteralPath $modelRoot -PathType Container)) {
    throw "The permitted external model root is missing: $modelRoot"
}

$records = @()
foreach ($entry in $mapping.GetEnumerator()) {
    $source = [System.IO.Path]::GetFullPath([string]$entry.Key)
    $target = [System.IO.Path]::GetFullPath([string]$entry.Value)
    if (-not $target.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Migration target escapes the project root: $target"
    }

    $status = "moved"
    $sourceItem = Get-Item -LiteralPath $source -Force -ErrorAction SilentlyContinue
    $targetItem = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    if ($null -ne $targetItem) {
        if ($null -eq $sourceItem) {
            $status = "already_moved"
        } elseif ($sourceItem.LinkType -eq "Junction") {
            $linkTarget = [System.IO.Path]::GetFullPath([string]($sourceItem.Target | Select-Object -First 1))
            if (-not $linkTarget.Equals($target, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Legacy junction points to the wrong target: $source -> $linkTarget"
            }
            $status = "already_moved"
        } else {
            throw "Both source and target exist; refusing to merge or overwrite: $source ; $target"
        }
    } else {
        if ($null -eq $sourceItem) {
            throw "Migration source is missing: $source"
        }
        if ($null -ne $sourceItem.LinkType) {
            throw "Migration source is unexpectedly a link: $source"
        }
        Move-Item -LiteralPath $source -Destination $target
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            throw "Move did not materialize the verified target: $target"
        }
    }

    if ($CreateLegacyJunctions) {
        $sourceItem = Get-Item -LiteralPath $source -Force -ErrorAction SilentlyContinue
        if ($null -eq $sourceItem) {
            New-Item -ItemType Junction -Path $source -Target $target | Out-Null
            $sourceItem = Get-Item -LiteralPath $source -Force
        }
        $linkTarget = [System.IO.Path]::GetFullPath([string]($sourceItem.Target | Select-Object -First 1))
        if ($sourceItem.LinkType -ne "Junction" -or
            -not $linkTarget.Equals($target, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Legacy evidence compatibility link failed validation: $source"
        }
    } elseif (Test-Path -LiteralPath $source) {
        throw "External source path still exists after migration: $source"
    }

    $records += [pscustomobject][ordered]@{
        legacy_path = $source
        project_path = $target
        status = $status
        legacy_junction = $CreateLegacyJunctions
    }
}

$manifest = [ordered]@{
    schema_version = 1
    migrated_at = [DateTimeOffset]::UtcNow.ToString("o")
    project_root = $projectPath
    model_root = $modelRoot
    model_root_external_exception = $true
    immutable_evidence_bytes_rewritten = $false
    mappings = $records
}
$manifestDirectory = Join-Path $projectPath "runs\manifests"
New-Item -ItemType Directory -Force -Path $manifestDirectory | Out-Null
$manifestPath = Join-Path $manifestDirectory "project-root-relocation.json"
$temporaryPath = "$manifestPath.tmp"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporaryPath -Encoding utf8
Move-Item -LiteralPath $temporaryPath -Destination $manifestPath -Force

Write-Host "Non-model SelfSight roots are now inside the project:"
$records | Format-Table legacy_path, project_path, status, legacy_junction -AutoSize
Write-Host "Relocation manifest: $manifestPath"
