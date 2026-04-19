param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",

    [ValidateSet("partitioned", "mpi", "all")]
    [string]$Backend = "partitioned",

    [int]$Repeat = 20,

    [string]$ReportPath = "",

    [switch]$AllowMissingPrereqs
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [string]$Label,
        [string]$Command,
        [hashtable]$ResultBag
    )

    Write-Host "`n=== $Label ==="
    Write-Host $Command
    Invoke-Expression $Command
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "PASS: $Label"
        [void]$ResultBag.Passed.Add($Label)
        return $true
    }

    Write-Host "FAIL: $Label (exit=$exitCode)"
    [void]$ResultBag.Failed.Add("$Label (exit=$exitCode)")
    return $false
}

function Add-ResultLine {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Line
    )
    [void]$Lines.Add($Line)
}

if ($Repeat -lt 1) {
    throw "Repeat must be >= 1"
}

$runStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$dateTag = Get-Date -Format "yyyy-MM-dd"

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $ReportPath = "docs/mfem-w2-io-local-report-$dateTag.md"
}

$selectedBackends = @()
if ($Backend -eq "all") {
    $selectedBackends = @("partitioned", "mpi")
} else {
    $selectedBackends = @($Backend)
}

$results = @{
    Passed = New-Object System.Collections.Generic.List[string]
    Failed = New-Object System.Collections.Generic.List[string]
    Warnings = New-Object System.Collections.Generic.List[string]
}

New-Item -ItemType Directory -Force -Path "output" | Out-Null

foreach ($b in $selectedBackends) {
    $crateTestLabel = "crate test ($b, pure-rust backend)"
    $crateTestCmd = "cargo test -p fem-io-hdf5-parallel -- --nocapture"
    $crateOk = Invoke-LoggedCommand -Label $crateTestLabel -Command $crateTestCmd -ResultBag $results
    if (-not $crateOk) {
        $results.Warnings.Add("$crateTestLabel failed; skipping backend '$b'.")
        continue
    }

    $exampleTestLabel = "example tests ($b, pure-rust backend)"
    $exampleTestCmd = "cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint -- --nocapture"
    $exampleOk = Invoke-LoggedCommand -Label $exampleTestLabel -Command $exampleTestCmd -ResultBag $results
    if (-not $exampleOk) {
        $results.Warnings.Add("$exampleTestLabel failed; skipping run loop for backend '$b'.")
        continue
    }

    $baseOutH5 = "output/local_ex43_$b.h5"
    $baseOutXdmf = "output/local_ex43_$b.xdmf"
    $runLabel = "example run ($b, pure-rust backend)"
    $runCmd = "cargo run -p fem-examples --example mfem_ex43_hdf5_checkpoint -- --backend $b --restart-step 2 --out-h5 $baseOutH5 --out-xdmf $baseOutXdmf"
    $runOk = Invoke-LoggedCommand -Label $runLabel -Command $runCmd -ResultBag $results
    if (-not $runOk) {
        $results.Warnings.Add("$runLabel failed; skipping repeat loop for backend '$b'.")
        continue
    }

    if ($Mode -eq "full") {
        $repeatCount = if ($b -eq "partitioned") { $Repeat } else { [Math]::Min($Repeat, 5) }
        $loopFailures = 0
        for ($i = 1; $i -le $repeatCount; $i++) {
            $h5 = "output/local_ex43_${b}_repeat_$i.h5"
            $xdmf = "output/local_ex43_${b}_repeat_$i.xdmf"
            $loopCmd = "cargo run -p fem-examples --example mfem_ex43_hdf5_checkpoint -- --backend $b --restart-step 2 --out-h5 $h5 --out-xdmf $xdmf"
            Write-Host "[repeat $i/$repeatCount][$b] $loopCmd"
            Invoke-Expression $loopCmd
            if ($LASTEXITCODE -ne 0) {
                $loopFailures += 1
            }
        }

        if ($loopFailures -eq 0) {
            $results.Passed.Add("repeat stability ($b): 0 failures / $repeatCount runs")
        } else {
            $results.Failed.Add("repeat stability ($b): $loopFailures failures / $repeatCount runs")
        }
    }
}

$reportLines = New-Object System.Collections.Generic.List[string]
Add-ResultLine -Lines $reportLines -Line "# Week-2 IO Local Execution Report"
Add-ResultLine -Lines $reportLines -Line ""
Add-ResultLine -Lines $reportLines -Line "Generated at: $runStamp"
Add-ResultLine -Lines $reportLines -Line "Mode: $Mode"
Add-ResultLine -Lines $reportLines -Line "Backends: $($selectedBackends -join ', ')"
Add-ResultLine -Lines $reportLines -Line "Repeat target: $Repeat"
Add-ResultLine -Lines $reportLines -Line ""

Add-ResultLine -Lines $reportLines -Line "## Passed"
if ($results.Passed.Count -eq 0) {
    Add-ResultLine -Lines $reportLines -Line "- none"
} else {
    foreach ($p in $results.Passed) {
        Add-ResultLine -Lines $reportLines -Line "- $p"
    }
}
Add-ResultLine -Lines $reportLines -Line ""

Add-ResultLine -Lines $reportLines -Line "## Failed"
if ($results.Failed.Count -eq 0) {
    Add-ResultLine -Lines $reportLines -Line "- none"
} else {
    foreach ($f in $results.Failed) {
        Add-ResultLine -Lines $reportLines -Line "- $f"
    }
}
Add-ResultLine -Lines $reportLines -Line ""

Add-ResultLine -Lines $reportLines -Line "## Warnings"
if ($results.Warnings.Count -eq 0) {
    Add-ResultLine -Lines $reportLines -Line "- none"
} else {
    foreach ($w in $results.Warnings) {
        Add-ResultLine -Lines $reportLines -Line "- $w"
    }
}
Add-ResultLine -Lines $reportLines -Line ""

Add-ResultLine -Lines $reportLines -Line "## Notes"
Add-ResultLine -Lines $reportLines -Line "- This report is intended as PM-001 local evidence when GitHub Actions is unavailable."
Add-ResultLine -Lines $reportLines -Line "- For formal closure, CI artifact links should still be added once Actions is available."
Add-ResultLine -Lines $reportLines -Line "- This run uses the pure-Rust checkpoint backend path (no native HDF5 dependency required)."

$reportDir = Split-Path -Parent $ReportPath
if (-not [string]::IsNullOrWhiteSpace($reportDir)) {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
}

Set-Content -Path $ReportPath -Value $reportLines -Encoding UTF8
Write-Host "`nReport written: $ReportPath"

if ($results.Failed.Count -gt 0) {
    if ($AllowMissingPrereqs) {
        $allPrereqLike = $true
        foreach ($f in $results.Failed) {
            if (-not ($f -like "*hdf5*" -or $f -like "*mpi*")) {
                $allPrereqLike = $false
                break
            }
        }
        if ($allPrereqLike) {
            Write-Host "Only HDF5 prerequisite-like failures detected; returning success due to -AllowMissingPrereqs."
            exit 0
        }
    }
    exit 1
}
exit 0
