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
    [string]$DeltaOutPath = "output/template_kpi_delta.csv",

    [Parameter(Mandatory = $false)]
    [double]$RelativeDriftWarn = 0.50,

    [Parameter(Mandatory = $false)]
    [double]$RelativeDriftFail = 1.00
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ExistingPath([string]$PathHint) {
    $candidates = @(
        $PathHint,
        (Join-Path "examples" $PathHint)
    )

    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return $c
        }
    }

    return ""
}

function Ensure-ParentDir([string]$FilePath) {
    $parent = Split-Path -Parent $FilePath
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Parse-IntField([string]$Value, [string]$Name, [string]$TemplateId) {
    $outVal = 0
    if (-not [int]::TryParse($Value, [ref]$outVal)) {
        throw "invalid integer field '$Name' for template '$TemplateId': '$Value'"
    }
    return $outVal
}

function Parse-Metrics([string]$MetricString, [string]$TemplateId) {
    $map = @{}
    if ([string]::IsNullOrWhiteSpace($MetricString)) {
        return $map
    }

    $pairs = $MetricString -split ';'
    foreach ($pair in $pairs) {
        if ([string]::IsNullOrWhiteSpace($pair)) {
            continue
        }
        $parts = $pair -split '=', 2
        if ($parts.Count -ne 2) {
            throw "invalid metric entry for template '$TemplateId': '$pair'"
        }

        $k = $parts[0].Trim()
        $vText = $parts[1].Trim()
        if ([string]::IsNullOrWhiteSpace($k)) {
            throw "empty metric key for template '$TemplateId'"
        }

        $v = 0.0
        $ok = [double]::TryParse(
            $vText,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$v
        )
        if (-not $ok) {
            throw "invalid metric value for template '$TemplateId': '$pair'"
        }
        if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) {
            throw "non-finite metric value for template '$TemplateId': '$pair'"
        }

        $map[$k] = $v
    }

    return $map
}

function RelDelta([double]$Current, [double]$Baseline) {
    $den = [Math]::Max([Math]::Abs($Baseline), 1.0e-12)
    return [Math]::Abs($Current - $Baseline) / $den
}

$expectedTemplates = @(
    "joule_heating",
    "fsi",
    "acoustics_structure",
    "electromagnetic_thermal_stress",
    "reaction_flow_thermal"
)

$_csvResolved = Resolve-ExistingPath $CsvPath
if ([string]::IsNullOrWhiteSpace($_csvResolved)) {
    Write-Error "Template KPI CSV not found. tried: '$CsvPath' and 'examples/$CsvPath'"
}

$rows = @(Import-Csv -Path $_csvResolved)
if ($rows.Count -eq 0) {
    Write-Error "Template KPI CSV is empty: $_csvResolved"
}

$requiredColumns = @(
    "template_id", "run_id", "tag", "steps", "converged_steps",
    "max_coupling_iters_used", "sync_retries", "rejected_sync_steps",
    "rollback_count", "metrics"
)
foreach ($c in $requiredColumns) {
    if (-not ($rows[0].PSObject.Properties.Name -contains $c)) {
        Write-Error "Template KPI CSV missing required column: $c"
    }
}

$rowMap = @{}
$gatePass = 0
$gateWarn = 0
$gateFail = 0

foreach ($r in $rows) {
    $id = "$($r.template_id)".Trim()
    if ([string]::IsNullOrWhiteSpace($id)) {
        Write-Error "template_id is empty in row"
    }

    $steps = Parse-IntField "$($r.steps)" "steps" $id
    $conv = Parse-IntField "$($r.converged_steps)" "converged_steps" $id
    $iters = Parse-IntField "$($r.max_coupling_iters_used)" "max_coupling_iters_used" $id
    $syncRetries = Parse-IntField "$($r.sync_retries)" "sync_retries" $id
    $rejected = Parse-IntField "$($r.rejected_sync_steps)" "rejected_sync_steps" $id
    $rollbacks = Parse-IntField "$($r.rollback_count)" "rollback_count" $id
    $metricsMap = Parse-Metrics "$($r.metrics)" $id

    $status = "pass"
    $reason = "ok"
    if ($steps -le 0 -or $iters -le 0) {
        $status = "fail"
        $reason = "invalid step/iteration counters"
    } elseif ($metricsMap.Count -lt 3) {
        $status = "fail"
        $reason = "insufficient metrics"
    } elseif ($syncRetries -gt 0 -or $rejected -gt 0 -or $rollbacks -gt 0) {
        $status = "warn"
        $reason = "adaptive retries/rejections/rollbacks observed"
    } elseif ($conv -eq 0) {
        $status = "warn"
        $reason = "no converged steps in quick profile"
    }

    switch ($status) {
        "pass" { $gatePass++ }
        "warn" { $gateWarn++ }
        "fail" { $gateFail++ }
    }

    Write-Host "template_kpi_gate_row,template=$id,status=$status,reason=$reason"
    $rowMap[$id] = [pscustomobject]@{
        template_id = $id
        steps = $steps
        converged_steps = $conv
        max_coupling_iters_used = $iters
        metrics = $metricsMap
    }
}

foreach ($exp in $expectedTemplates) {
    if (-not $rowMap.ContainsKey($exp)) {
        Write-Host "template_kpi_gate_row,template=$exp,status=fail,reason=missing template row"
        $gateFail++
    }
}

Write-Host "template_kpi_gate_summary,rows=$($rows.Count),pass=$gatePass,warn=$gateWarn,fail=$gateFail"

if ($gateFail -gt 0) {
    Write-Error "Template KPI gate failed: $gateFail fail rows"
}
if ($FailOnWarn -and $gateWarn -gt 0) {
    Write-Error "Template KPI gate failed: warn rows present and FailOnWarn is enabled"
}

$baselineResolved = ""
if (-not [string]::IsNullOrWhiteSpace($BaselinePath)) {
    $baselineResolved = Resolve-ExistingPath $BaselinePath
}

if ([string]::IsNullOrWhiteSpace($baselineResolved)) {
    Write-Host "template_kpi_trend_compare,skipped=true,reason=no baseline provided"
    exit 0
}

$baseRows = @(Import-Csv -Path $baselineResolved)
if ($baseRows.Count -eq 0) {
    Write-Error "Template KPI baseline CSV is empty: $baselineResolved"
}

$baseMap = @{}
foreach ($b in $baseRows) {
    $id = "$($b.template_id)".Trim()
    if ([string]::IsNullOrWhiteSpace($id)) {
        continue
    }

    $baseMap[$id] = [pscustomobject]@{
        template_id = $id
        steps = Parse-IntField "$($b.steps)" "steps" $id
        converged_steps = Parse-IntField "$($b.converged_steps)" "converged_steps" $id
        max_coupling_iters_used = Parse-IntField "$($b.max_coupling_iters_used)" "max_coupling_iters_used" $id
        metrics = Parse-Metrics "$($b.metrics)" $id
    }
}

$trendPass = 0
$trendWarn = 0
$trendFail = 0
$deltaRows = New-Object System.Collections.Generic.List[object]

foreach ($id in $expectedTemplates) {
    if (-not $rowMap.ContainsKey($id)) {
        $trendFail++
        $deltaRows.Add([pscustomobject]@{
            template_id = $id
            key = "row"
            baseline = ""
            current = ""
            rel_delta = ""
            status = "fail"
            reason = "missing current row"
        })
        continue
    }
    if (-not $baseMap.ContainsKey($id)) {
        $trendWarn++
        $deltaRows.Add([pscustomobject]@{
            template_id = $id
            key = "row"
            baseline = ""
            current = ""
            rel_delta = ""
            status = "warn"
            reason = "missing baseline row"
        })
        continue
    }

    $cur = $rowMap[$id]
    $base = $baseMap[$id]

    $checks = @(
        @{ key = "steps"; cur = [double]$cur.steps; base = [double]$base.steps },
        @{ key = "converged_steps"; cur = [double]$cur.converged_steps; base = [double]$base.converged_steps },
        @{ key = "max_coupling_iters_used"; cur = [double]$cur.max_coupling_iters_used; base = [double]$base.max_coupling_iters_used }
    )

    foreach ($mk in $base.metrics.Keys) {
        if ($cur.metrics.ContainsKey($mk)) {
            $checks += @{ key = $mk; cur = [double]$cur.metrics[$mk]; base = [double]$base.metrics[$mk] }
        } else {
            $trendWarn++
            $deltaRows.Add([pscustomobject]@{
                template_id = $id
                key = $mk
                baseline = [double]$base.metrics[$mk]
                current = ""
                rel_delta = ""
                status = "warn"
                reason = "missing metric in current row"
            })
        }
    }

    foreach ($check in $checks) {
        $rel = RelDelta $check.cur $check.base
        $status = "pass"
        $reason = "ok"
        if ($rel -gt $RelativeDriftFail) {
            $status = "fail"
            $reason = "relative drift"
            $trendFail++
        } elseif ($rel -gt $RelativeDriftWarn) {
            $status = "warn"
            $reason = "relative drift"
            $trendWarn++
        } else {
            $trendPass++
        }

        $deltaRows.Add([pscustomobject]@{
            template_id = $id
            key = $check.key
            baseline = $check.base
            current = $check.cur
            rel_delta = $rel
            status = $status
            reason = $reason
        })
    }
}

Write-Host "template_kpi_trend_summary,pass=$trendPass,warn=$trendWarn,fail=$trendFail"
Ensure-ParentDir $DeltaOutPath
$deltaRows | Export-Csv -Path $DeltaOutPath -NoTypeInformation
Write-Host "template_kpi_delta_csv,path=$DeltaOutPath,rows=$($deltaRows.Count)"

if ($FailOnTrendRegression -and $trendFail -gt 0) {
    Write-Error "Template KPI trend comparison failed due to regression rows"
}