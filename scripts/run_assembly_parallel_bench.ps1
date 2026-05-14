param(
    [int]$Threads = 0,
    [int]$ParallelMinElems = 64,
    [switch]$CompileOnly,
    [string]$ExportStamp = "",
    [ValidateSet("all", "serial", "parallel")]
    [string]$Mode = "all",
    [string]$Filter = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Bench([string]$Name, [string]$Command) {
    Write-Host ""
    Write-Host "=== $Name ==="
    Write-Host $Command
    Invoke-Expression $Command
}

function Copy-CriterionHistory([string]$Stamp) {
    if ([string]::IsNullOrWhiteSpace($Stamp)) {
        return $false
    }
    $src = Join-Path $PWD "target/criterion"
    if (-not (Test-Path $src)) {
        Write-Warning "No criterion output found at $src; skip export."
        return $false
    }
    $dstRoot = Join-Path $PWD "target/criterion-history"
    $dst = Join-Path $dstRoot $Stamp
    if (Test-Path $dst) {
        Remove-Item -Recurse -Force $dst
    }
    New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null
    Copy-Item -Recurse -Force $src $dst
    Write-Host "Exported criterion results to: $dst"
    return $true
}

function Build-BenchCommand([string]$FeatureFlag, [string]$BenchArgs, [string]$FilterExpr) {
    $cmd = "cargo bench -p fem-benches $FeatureFlag --bench assembly $BenchArgs"
    if (-not [string]::IsNullOrWhiteSpace($FilterExpr)) {
        $cmd = "$cmd -- $FilterExpr"
    }
    return $cmd.Trim()
}

$oldThreads = $env:RAYON_NUM_THREADS
$oldParallelMin = $env:FEM_ASSEMBLY_PARALLEL_MIN_ELEMS

try {
    if ($Threads -gt 0) {
        $env:RAYON_NUM_THREADS = "$Threads"
        Write-Host "RAYON_NUM_THREADS=$($env:RAYON_NUM_THREADS)"
    } else {
        Write-Host "RAYON_NUM_THREADS not set (Rayon default)"
    }

    $env:FEM_ASSEMBLY_PARALLEL_MIN_ELEMS = "$ParallelMinElems"
    Write-Host "FEM_ASSEMBLY_PARALLEL_MIN_ELEMS=$($env:FEM_ASSEMBLY_PARALLEL_MIN_ELEMS)"

    $benchArgs = if ($CompileOnly) { "--no-run" } else { "" }

    $serialCmd = Build-BenchCommand "" $benchArgs $Filter
    $parallelCmd = Build-BenchCommand "--features parallel" $benchArgs $Filter

    if ($Mode -eq "all" -or $Mode -eq "serial") {
        Invoke-Bench "Serial assembly benchmark" $serialCmd
    }
    if ($Mode -eq "all" -or $Mode -eq "parallel") {
        Invoke-Bench "Parallel assembly benchmark" $parallelCmd
    }

    $exported = Copy-CriterionHistory -Stamp $ExportStamp

    Write-Host ""
    Write-Host "Done. Criterion outputs are under target/criterion/."
    if ($exported) {
        Write-Host "History snapshot: target/criterion-history/$ExportStamp"
    }
}
finally {
    if ($null -ne $oldThreads) { $env:RAYON_NUM_THREADS = $oldThreads } else { Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue }
    if ($null -ne $oldParallelMin) { $env:FEM_ASSEMBLY_PARALLEL_MIN_ELEMS = $oldParallelMin } else { Remove-Item Env:FEM_ASSEMBLY_PARALLEL_MIN_ELEMS -ErrorAction SilentlyContinue }
}
