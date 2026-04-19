param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,

    [Parameter(Mandatory = $false)]
    [string]$BaselinePath = "",

    [Parameter(Mandatory = $false)]
    [switch]$FailOnWarn,

    [Parameter(Mandatory = $false)]
    [switch]$FailOnTrendRegression,

    [Parameter(Mandatory = $false)]
    [string]$DeltaOutPath = "output/ras_scaling_delta.csv",

    [Parameter(Mandatory = $false)]
    [double]$StrongEffDropTol = 0.15,

    [Parameter(Mandatory = $false)]
    [double]$WeakGrowthRiseTol = 0.25
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ExistingPath([string]$PathHint) {
    $candidates = @(
        $PathHint,
        (Join-Path "crates/parallel" $PathHint)
    )

    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return $c
        }
    }

    return ""
}

function KeyOf($row) {
    return "$($row.mode)|$($row.ranks)"
}

function Ensure-ParentDir([string]$FilePath) {
    $parent = Split-Path -Parent $FilePath
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

$_csvResolved = Resolve-ExistingPath $CsvPath
if ([string]::IsNullOrWhiteSpace($_csvResolved)) {
    Write-Error "RAS scaling CSV not found. tried: '$CsvPath' and 'crates/parallel/$CsvPath'"
}

$rows = @(Import-Csv -Path $_csvResolved)
if ($rows.Count -eq 0) {
    Write-Error "RAS scaling CSV is empty: $_csvResolved"
}

$requiredColumns = @(
    "mode", "ranks", "mesh_n", "dofs", "iterations", "final_residual", "time_ms",
    "strong_eff", "weak_growth", "owned", "ghost", "nnz_diag", "nnz_offd",
    "owned_cv", "ghost_cv", "score"
)

foreach ($c in $requiredColumns) {
    if (-not ($rows[0].PSObject.Properties.Name -contains $c)) {
        Write-Error "RAS scaling CSV missing required column: $c"
    }
}

$passCount = @($rows | Where-Object { $_.score -eq "pass" }).Count
$warnCount = @($rows | Where-Object { $_.score -eq "warn" }).Count
$failCount = @($rows | Where-Object { $_.score -eq "fail" }).Count

Write-Host "hpc_gate_summary,rows=$($rows.Count),pass=$passCount,warn=$warnCount,fail=$failCount"

if ($failCount -gt 0) {
    Write-Error "HPC scaling gate failed: $failCount fail rows in score column"
}

if ($FailOnWarn -and $warnCount -gt 0) {
    Write-Error "HPC scaling gate failed: warn rows present and FailOnWarn is enabled"
}

$baselineResolved = ""
if (-not [string]::IsNullOrWhiteSpace($BaselinePath)) {
    $baselineResolved = Resolve-ExistingPath $BaselinePath
}

if ([string]::IsNullOrWhiteSpace($baselineResolved)) {
    Write-Host "hpc_trend_compare,skipped=true,reason=no baseline provided"
    exit 0
}

$baseRows = @(Import-Csv -Path $baselineResolved)
if ($baseRows.Count -eq 0) {
    Write-Error "Baseline CSV is empty: $baselineResolved"
}

$baseMap = @{}
foreach ($b in $baseRows) {
    $baseMap[(KeyOf $b)] = $b
}

$trendPass = 0
$trendWarn = 0
$trendFail = 0
$deltaRows = New-Object System.Collections.Generic.List[object]

foreach ($r in $rows) {
    $k = KeyOf $r
    if (-not $baseMap.ContainsKey($k)) {
        Write-Host "trend_row,key=$k,status=warn,reason=missing baseline row"
        $trendWarn++
        $deltaRows.Add([pscustomobject]@{
            key = $k
            mode = $r.mode
            ranks = [int]$r.ranks
            status = "warn"
            reason = "missing baseline row"
            strong_eff_baseline = ""
            strong_eff_current = [double]$r.strong_eff
            strong_eff_delta = ""
            weak_growth_baseline = ""
            weak_growth_current = [double]$r.weak_growth
            weak_growth_delta = ""
            weak_growth_delta_ratio = ""
        })
        continue
    }

    $b = $baseMap[$k]
    $status = "pass"
    $reason = "ok"
    $strongEffDelta = ""
    $weakGrowthDelta = ""
    $weakGrowthDeltaRatio = ""

    if ($r.mode -eq "strong") {
        $cur = [double]$r.strong_eff
        $ref = [double]$b.strong_eff
        $strongEffDelta = $cur - $ref
        if ($ref -gt 0.0) {
            $drop = ($ref - $cur) / $ref
            if ($drop -gt ($StrongEffDropTol * 1.5)) {
                $status = "fail"
                $reason = "strong_eff regression"
            } elseif ($drop -gt $StrongEffDropTol) {
                $status = "warn"
                $reason = "strong_eff drift"
            }
        }
    } elseif ($r.mode -eq "weak") {
        $cur = [double]$r.weak_growth
        $ref = [double]$b.weak_growth
        $weakGrowthDelta = $cur - $ref
        if ($ref -gt 0.0) {
            $rise = ($cur - $ref) / $ref
            $weakGrowthDeltaRatio = $rise
            if ($rise -gt ($WeakGrowthRiseTol * 1.5)) {
                $status = "fail"
                $reason = "weak_growth regression"
            } elseif ($rise -gt $WeakGrowthRiseTol) {
                $status = "warn"
                $reason = "weak_growth drift"
            }
        }
    }

    switch ($status) {
        "pass" { $trendPass++ }
        "warn" { $trendWarn++ }
        "fail" { $trendFail++ }
    }

    Write-Host "trend_row,key=$k,status=$status,reason=$reason"

    $deltaRows.Add([pscustomobject]@{
        key = $k
        mode = $r.mode
        ranks = [int]$r.ranks
        status = $status
        reason = $reason
        strong_eff_baseline = [double]$b.strong_eff
        strong_eff_current = [double]$r.strong_eff
        strong_eff_delta = $strongEffDelta
        weak_growth_baseline = [double]$b.weak_growth
        weak_growth_current = [double]$r.weak_growth
        weak_growth_delta = $weakGrowthDelta
        weak_growth_delta_ratio = $weakGrowthDeltaRatio
    })
}

Write-Host "hpc_trend_summary,pass=$trendPass,warn=$trendWarn,fail=$trendFail"

Ensure-ParentDir $DeltaOutPath
$deltaRows | Export-Csv -Path $DeltaOutPath -NoTypeInformation
Write-Host "hpc_trend_delta_csv,path=$DeltaOutPath,rows=$($deltaRows.Count)"

if ($FailOnTrendRegression -and $trendFail -gt 0) {
    Write-Error "HPC trend comparison failed due to regression rows"
}
