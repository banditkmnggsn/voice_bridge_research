# VoiceBridge final benchmark runner.
# Runs smoke test -> cooldown -> idle baseline -> full benchmark -> analysis
# -> validation -> summary. Everything is logged to results\_logs.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1

$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot

$Py = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')).Path
$Results = Join-Path $PSScriptRoot 'results'
$LogDir = Join-Path $Results '_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Log = Join-Path $LogDir "run_all_$Stamp.log"
$SummaryPath = Join-Path $LogDir "RUN_SUMMARY_$Stamp.md"

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Steps = New-Object System.Collections.Generic.List[object]
$script:LastOutput = New-Object System.Collections.Generic.List[string]
$IdleBaseline = 'not measured'
$FullValidation = 'not run'
$SmokeValidation = 'not run'
$RunStart = Get-Date

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
    Write-Host $line
}

function Add-Step([string]$Name, [string]$Status, [double]$Seconds, [string]$Note = '') {
    $Steps.Add([pscustomobject]@{ Name = $Name; Status = $Status; Seconds = $Seconds; Note = $Note })
}

function Invoke-Py([string]$Name, [string[]]$PyArgs) {
    Write-Log "=== STEP $Name START: python -u $($PyArgs -join ' ')"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $script:LastOutput = New-Object System.Collections.Generic.List[string]
    & $Py -u @PyArgs 2>&1 | ForEach-Object {
        $text = "$_"
        $script:LastOutput.Add($text)
        Add-Content -LiteralPath $Log -Value $text -Encoding UTF8
        Write-Host $text
    }
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Log ("=== STEP $Name END: exit={0} duration={1:N1}s" -f $code, $sw.Elapsed.TotalSeconds)
    return [pscustomobject]@{ Code = $code; Seconds = $sw.Elapsed.TotalSeconds }
}

function Get-ValidationLine {
    $match = $script:LastOutput | Where-Object { $_ -match '^PASS=\d+ WARN=\d+ FAIL=\d+' } | Select-Object -Last 1
    if ($match) { return $match.Trim() }
    return $null
}

function Get-Newest([string]$Pattern) {
    $file = Get-ChildItem -LiteralPath $Results -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($file) { return $file.FullName }
    return '(not found)'
}

function Write-Summary([string]$Overall) {
    $total = ((Get-Date) - $RunStart).TotalMinutes
    $runs = Get-Newest 'runs_*.csv'
    $envPath = Get-Newest 'env_*.json'
    $report = Get-Newest 'report_*.md'
    $validation = Get-Newest 'validation_*.md'
    $throttle = 'unknown'
    if (Test-Path -LiteralPath $envPath) {
        try {
            $envJson = Get-Content -LiteralPath $envPath -Raw | ConvertFrom-Json
            $throttle = "$($envJson.power_throttling_disabled)"
        } catch { }
    }
    $lines = @(
        "# VoiceBridge benchmark run summary $Stamp",
        "",
        "Overall: **$Overall**",
        "",
        ("Started: {0}  |  Finished: {1}  |  Total: {2:N1} min" -f $RunStart.ToString('yyyy-MM-dd HH:mm:ss'), (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $total),
        "",
        "## Steps",
        "",
        "| Step | Status | Duration (s) | Note |",
        "|---|---|---|---|"
    )
    foreach ($s in $Steps) {
        $lines += ("| {0} | {1} | {2:N1} | {3} |" -f $s.Name, $s.Status, $s.Seconds, $s.Note)
    }
    $lines += @(
        "",
        "## Idle baseline (30 s, before full benchmark)",
        "",
        $IdleBaseline,
        "",
        "## Validation",
        "",
        "- Smoke test: $SmokeValidation",
        "- Full run: $FullValidation",
        "- power_throttling_disabled (env json): $throttle",
        "",
        "## Artifacts",
        "",
        "- Log: ``$Log``",
        "- Runs: ``$runs``",
        "- Env: ``$envPath``",
        "- Report: ``$report``",
        "- Validation: ``$validation``"
    )
    Set-Content -LiteralPath $SummaryPath -Value $lines -Encoding UTF8
    Write-Log "Summary written: $SummaryPath"
}

function Stop-Failed([string]$Name) {
    Write-Log "FAILED $Name"
    Write-Summary "FAILED at $Name"
    exit 1
}

Write-Log "VoiceBridge run_all started. Python: $Py"

# 1. Smoke test
$r = Invoke-Py 'smoke_benchmark' @('run_benchmark.py', '--test', '--modes', 'rule', '--repeats', '1', '--limit', '2')
if ($r.Code -ne 0) { Add-Step 'smoke_benchmark' 'FAILED' $r.Seconds "exit $($r.Code)"; Stop-Failed 'smoke_benchmark' }
Add-Step 'smoke_benchmark' 'OK' $r.Seconds

$r = Invoke-Py 'smoke_validate' @('validate_run.py', '--test')
$SmokeValidation = Get-ValidationLine
if (-not $SmokeValidation) { $SmokeValidation = 'no summary line' }
if ($r.Code -ne 0 -or $SmokeValidation -notmatch 'FAIL=0') {
    Add-Step 'smoke_validate' 'FAILED' $r.Seconds $SmokeValidation
    Stop-Failed 'smoke_validate'
}
Add-Step 'smoke_validate' 'OK' $r.Seconds $SmokeValidation
Write-Log "SMOKE TEST PASS ($SmokeValidation)"

# 2. Cooldown
Write-Log '=== STEP cooldown START: 120 s'
Start-Sleep -Seconds 120
Write-Log '=== STEP cooldown END'
Add-Step 'cooldown' 'OK' 120

# 3. Idle baseline
$idleCode = @'
import psutil, statistics as s; psutil.cpu_percent(None); x = [(psutil.cpu_percent(interval=1), psutil.virtual_memory()) for _ in range(30)]; c = [a for a, _ in x]; u = [m.used / 2**20 for _, m in x]; p = [m.percent for _, m in x]; print('IDLE_BASELINE samples=%d cpu_mean_percent=%.2f cpu_max_percent=%.2f ram_used_mean_mb=%.0f ram_used_mean_percent=%.1f ram_total_mb=%.0f' % (len(x), s.mean(c), max(c), s.mean(u), s.mean(p), x[-1][1].total / 2**20))
'@
$r = Invoke-Py 'idle_baseline' @('-c', $idleCode.Trim())
if ($r.Code -ne 0) { Add-Step 'idle_baseline' 'FAILED' $r.Seconds "exit $($r.Code)"; Stop-Failed 'idle_baseline' }
$idleLine = $script:LastOutput | Where-Object { $_ -match '^IDLE_BASELINE' } | Select-Object -Last 1
if ($idleLine) { $IdleBaseline = $idleLine.Trim() }
Add-Step 'idle_baseline' 'OK' $r.Seconds $IdleBaseline

# 4. Full benchmark
Write-Log 'FULL BENCHMARK STARTING'
$r = Invoke-Py 'full_benchmark' @('run_benchmark.py')
if ($r.Code -ne 0) { Add-Step 'full_benchmark' 'FAILED' $r.Seconds "exit $($r.Code)"; Stop-Failed 'full_benchmark' }
Add-Step 'full_benchmark' 'OK' $r.Seconds

# 5. Analysis
$r = Invoke-Py 'analyze' @('analyze_results.py')
if ($r.Code -ne 0) { Add-Step 'analyze' 'FAILED' $r.Seconds "exit $($r.Code)"; Stop-Failed 'analyze' }
Add-Step 'analyze' 'OK' $r.Seconds

# 6. Validation (FAIL is recorded but does not stop the summary)
$r = Invoke-Py 'validate' @('validate_run.py')
$FullValidation = Get-ValidationLine
if (-not $FullValidation) {
    $FullValidation = "no summary line (exit $($r.Code))"
    Add-Step 'validate' 'FAILED' $r.Seconds $FullValidation
    Stop-Failed 'validate'
}
$validateStatus = 'OK'
if ($FullValidation -notmatch 'FAIL=0') { $validateStatus = 'HAS_FAIL' }
Add-Step 'validate' $validateStatus $r.Seconds $FullValidation

# 7. Summary
$overall = 'SUCCESS'
if ($validateStatus -ne 'OK') { $overall = 'COMPLETED WITH VALIDATION FAIL' }
Write-Summary $overall
Write-Log "RUN_ALL DONE: $overall"
exit 0
