param(
    [int]$Threads = 0,
    [string]$OutputCsv = "output/spmv_micro.csv",
    [string]$BaselinePath = "docs/baselines/spmv_micro_baseline.csv",
    [string]$Filter = "poisson_",
    [double]$ParallelWarnRatio = 1.10,
    [switch]$Quick,
    [switch]$FailOnParallelWarn
)

$ErrorActionPreference = "Stop"

function Get-EstimateFiles([string]$Root, [string]$FilterText) {
    if (-not (Test-Path $Root)) {
        return @()
    }
    Get-ChildItem -Path $Root -Recurse -File -Filter estimates.json |
        Where-Object {
            $_.FullName -match [regex]::Escape($FilterText) -and
            $_.DirectoryName -match '[\\/]new$'
        }
}

function Get-BenchmarkId([string]$FullName, [string]$Root) {
    $relative = $FullName.Substring($Root.Length).TrimStart('\', '/')
    $parts = $relative -split '[\\/]'
    if ($parts.Length -lt 3) {
        return $relative
    }
    return $parts[0]
}

function Get-BaselineMap([string]$Path) {
    $map = @{}
    if (-not (Test-Path $Path)) {
        return $map
    }
    $rows = Import-Csv -Path $Path
    foreach ($row in $rows) {
        if (-not [string]::IsNullOrWhiteSpace($row.benchmark_id)) {
            $map[$row.benchmark_id] = $row
        }
    }
    return $map
}

function Get-CaseKey([string]$BenchmarkId) {
    if ($BenchmarkId -match '^(serial|parallel)_(.+)$') {
        return $Matches[2]
    }
    return $BenchmarkId
}

function Get-CaseMode([string]$BenchmarkId) {
    if ($BenchmarkId -match '^(serial|parallel)_') {
        return $Matches[1]
    }
    return "other"
}

$oldThreads = $env:RAYON_NUM_THREADS
$oldQuick = $env:FEM_BENCH_QUICK

try {
    if ($Threads -gt 0) {
        $env:RAYON_NUM_THREADS = "$Threads"
        Write-Host "RAYON_NUM_THREADS=$($env:RAYON_NUM_THREADS)"
    }

    if ($Quick) {
        $env:FEM_BENCH_QUICK = "1"
        Write-Host "FEM_BENCH_QUICK=1"
    }

    $criterionRoot = Join-Path $PWD "target/criterion/spmv"
    if (Test-Path $criterionRoot) {
        Remove-Item -Recurse -Force $criterionRoot
    }

    New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName((Join-Path $PWD $OutputCsv))) | Out-Null

    cargo bench -p fem-benches --features parallel --bench micro -- $Filter

    $estimateFiles = Get-EstimateFiles -Root $criterionRoot -FilterText $Filter
    if ($estimateFiles.Count -eq 0) {
        throw "No Criterion estimates found under $criterionRoot for filter '$Filter'."
    }

    $baselineMap = Get-BaselineMap -Path (Join-Path $PWD $BaselinePath)
    $rows = foreach ($file in $estimateFiles) {
        $json = Get-Content -Raw -Path $file.FullName | ConvertFrom-Json
        $benchmarkId = Get-BenchmarkId -FullName $file.FullName -Root $criterionRoot
        $meanNs = [double]$json.mean.point_estimate
        $baseline = $null
        $ratio = $null
        $status = "no-baseline"
        if ($baselineMap.ContainsKey($benchmarkId)) {
            $baseline = [double]$baselineMap[$benchmarkId].baseline_ns
            if ($baseline -gt 0.0) {
                $ratio = $meanNs / $baseline
                $warnRatio = 1.10
                if ($baselineMap[$benchmarkId].warn_ratio) {
                    $warnRatio = [double]$baselineMap[$benchmarkId].warn_ratio
                }
                $status = if ($ratio -le $warnRatio) { "pass" } else { "warn" }
            }
        }
        [pscustomobject]@{
            benchmark_id = $benchmarkId
            mean_ns = [math]::Round($meanNs, 3)
            baseline_ns = if ($null -ne $baseline) { [math]::Round($baseline, 3) } else { $null }
            ratio = if ($null -ne $ratio) { [math]::Round($ratio, 4) } else { $null }
            status = $status
        }
    }

    $rows = $rows | Sort-Object benchmark_id
    $rows | Export-Csv -Path $OutputCsv -NoTypeInformation

    $pairRows = @()
    $grouped = $rows | Group-Object { Get-CaseKey $_.benchmark_id }
    foreach ($group in $grouped) {
        $serial = $group.Group | Where-Object { (Get-CaseMode $_.benchmark_id) -eq 'serial' } | Select-Object -First 1
        $parallel = $group.Group | Where-Object { (Get-CaseMode $_.benchmark_id) -eq 'parallel' } | Select-Object -First 1
        if ($null -eq $serial -or $null -eq $parallel) {
            continue
        }
        $serialNs = [double]$serial.mean_ns
        $parallelNs = [double]$parallel.mean_ns
        if ($serialNs -le 0.0) {
            continue
        }
        $ratio = $parallelNs / $serialNs
        $status = if ($ratio -le $ParallelWarnRatio) { 'pass' } else { 'warn' }
        $pairRows += [pscustomobject]@{
            case_id = $group.Name
            serial_ns = [math]::Round($serialNs, 3)
            parallel_ns = [math]::Round($parallelNs, 3)
            ratio = [math]::Round($ratio, 4)
            status = $status
        }
    }

    $warnCount = ($rows | Where-Object { $_.status -eq 'warn' }).Count
    $passCount = ($rows | Where-Object { $_.status -eq 'pass' }).Count
    $nobaseCount = ($rows | Where-Object { $_.status -eq 'no-baseline' }).Count
    Write-Host "spmv_bench_summary,count=$($rows.Count),pass=$passCount,warn=$warnCount,no_baseline=$nobaseCount,csv=$OutputCsv"

    $pairWarnCount = ($pairRows | Where-Object { $_.status -eq 'warn' }).Count
    $pairPassCount = ($pairRows | Where-Object { $_.status -eq 'pass' }).Count
    Write-Host "spmv_parallel_vs_serial_summary,count=$($pairRows.Count),pass=$pairPassCount,warn=$pairWarnCount,warn_ratio=$ParallelWarnRatio"
    foreach ($pair in $pairRows) {
        Write-Host "spmv_parallel_vs_serial_case,case=$($pair.case_id),serial_ns=$($pair.serial_ns),parallel_ns=$($pair.parallel_ns),ratio=$($pair.ratio),status=$($pair.status)"
    }

    if ($FailOnParallelWarn -and $pairWarnCount -gt 0) {
        throw "SpMV parallel-vs-serial benchmark reported $pairWarnCount warn rows."
    }
}
finally {
    if ($null -ne $oldThreads) { $env:RAYON_NUM_THREADS = $oldThreads } else { Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue }
    if ($null -ne $oldQuick) { $env:FEM_BENCH_QUICK = $oldQuick } else { Remove-Item Env:FEM_BENCH_QUICK -ErrorAction SilentlyContinue }
}
