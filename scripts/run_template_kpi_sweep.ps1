param(
    [Parameter(Mandatory = $false)]
    [string]$OutputCsv = "output/template_kpi.csv",

    [Parameter(Mandatory = $false)]
    [string]$RunId = "local",

    [Parameter(Mandatory = $false)]
    [string]$Tag = "quick",

    [Parameter(Mandatory = $false)]
    [switch]$Append
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-ParentDir([string]$FilePath) {
    $parent = Split-Path -Parent $FilePath
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Invoke-TemplateExample([string]$Name, [string[]]$ExampleArgs) {
    Write-Host "template_kpi_run,start,example=$Name"
    $cmdArgs = @("run", "-p", "fem-examples", "--example", $Name, "--") + $ExampleArgs
    & cargo @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Template example failed: $Name"
    }
    Write-Host "template_kpi_run,done,example=$Name"
}

Ensure-ParentDir $OutputCsv
if ((-not $Append) -and (Test-Path $OutputCsv)) {
    Remove-Item -Path $OutputCsv -Force
}

$env:FEM_TEMPLATE_KPI_CSV = $OutputCsv
$env:FEM_TEMPLATE_KPI_RUN_ID = $RunId
$env:FEM_TEMPLATE_KPI_TAG = $Tag

Invoke-TemplateExample "mfem_ex48_template_joule_heating" @("--n", "8", "--max-coupling", "4", "--tol", "1e-6", "--no-subcycling")
Invoke-TemplateExample "mfem_ex49_template_fsi" @("--n", "8", "--steps", "4", "--max-coupling", "4", "--coupling-tol", "1e-6", "--no-subcycling")
Invoke-TemplateExample "mfem_ex50_template_acoustics_structure" @("--n", "8", "--steps", "4", "--max-coupling", "4", "--coupling-tol", "1e-6", "--no-subcycling")
Invoke-TemplateExample "mfem_ex51_template_em_thermal_stress" @("--n", "8", "--steps", "4", "--max-coupling", "4", "--coupling-tol", "1e-6", "--no-subcycling")
Invoke-TemplateExample "mfem_ex52_template_reaction_flow_thermal" @("--n", "8", "--steps", "4", "--max-coupling", "4", "--coupling-tol", "1e-6", "--no-subcycling")

if (-not (Test-Path $OutputCsv)) {
    throw "Template KPI CSV was not generated: $OutputCsv"
}

$rows = @(Import-Csv -Path $OutputCsv)
$templates = @($rows | Select-Object -ExpandProperty template_id -Unique)
Write-Host "template_kpi_summary,path=$OutputCsv,rows=$($rows.Count),templates=$($templates.Count)"

if ($templates.Count -lt 5) {
    throw "Template KPI sweep incomplete: expected >= 5 template rows, got $($templates.Count)"
}