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
    "ex1_poisson",
    "ex2_elasticity",
    "ex3_maxwell",
    "ex4_darcy",
    "ex5_mixed_darcy",
    "ex7_neumann_mixed_bc",
    "ex9_dg_advection",
    "ex10_heat_equation",
    "ex10_wave_equation",
    "ex13_eigenvalue",
    "ex14_dc_current",
    "ex39_named_attributes",
    "ex15_dg_amr",
    "ex15_tet_nc_amr",
    "ex16_nonlinear_heat",
    "ex26_geom_mg",
    "ex_stokes",
    "ex_navier_stokes"
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
