param(
    [switch]$CoreOnly,
    [switch]$MfemOnly,
    [switch]$NoQuiet
)

$ErrorActionPreference = "Stop"

$quietArgs = @()
if (-not $NoQuiet) {
    $quietArgs = @("--quiet")
}

$mfemExamples = @(
    "mfem_ex3",
    "mfem_ex13",
    "mfem_ex22",
    "mfem_ex25",
    "mfem_ex31",
    "mfem_ex32",
    "mfem_ex34",
    "mfem_volta",
    "mfem_tesla",
    "mfem_maxwell",
    "mfem_joule"
)

$coreExamples = @(
    "mfem_ex1_poisson",
    "mfem_ex2_elasticity",
    "mfem_ex4_darcy",
    "mfem_ex5_mixed_darcy",
    "mfem_ex7_neumann_mixed_bc",
    "mfem_ex9_dg_advection",
    "mfem_ex10_heat_equation",
    "mfem_ex10_wave_equation",
    "mfem_ex14_dc_current",
    "mfem_ex39_named_attributes",
    "mfem_ex15_dg_amr",
    "mfem_ex15_tet_nc_amr",
    "mfem_ex16_nonlinear_heat",
    "mfem_ex26_geom_mg"
)

$runList = @()
if ($CoreOnly -and $MfemOnly) {
    throw "Cannot use -CoreOnly and -MfemOnly together."
} elseif ($CoreOnly) {
    $runList = $coreExamples
} elseif ($MfemOnly) {
    $runList = $mfemExamples
} else {
    $runList = $mfemExamples + $coreExamples
}

Write-Host "Running $($runList.Count) examples from fem-examples..."

$passed = @()
$failed = @()

foreach ($example in $runList) {
    Write-Host "`n=== Running $example ==="
    $args = @("run", "-p", "fem-examples", "--example", $example) + $quietArgs

    & cargo @args
    if ($LASTEXITCODE -eq 0) {
        $passed += $example
    } else {
        $failed += $example
        break
    }
}

Write-Host "`n=== Summary ==="
Write-Host "Passed: $($passed.Count)"
if ($passed.Count -gt 0) {
    Write-Host ($passed -join ", ")
}

if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed.Count)"
    Write-Host ($failed -join ", ")
    exit 1
}

Write-Host "Failed: 0"
Write-Host "All requested examples passed."
exit 0

