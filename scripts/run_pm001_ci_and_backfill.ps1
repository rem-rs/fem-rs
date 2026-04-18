param(
    [Parameter(Mandatory = $true)]
    [string]$UpdatedBy,

    [string]$GhPath = "C:\Program Files\GitHub CLI\gh.exe",
    [string]$WorkflowFile = ".github/workflows/io-parity-hdf5.yml",
    [string]$WorkflowName = "IO Parity HDF5",
    [switch]$NoStatusFlip,
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

function Require-File {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

function Invoke-Gh {
    param([string[]]$GhArgs)
    $tmpOut = New-TemporaryFile
    $tmpErr = New-TemporaryFile

    try {
        $proc = Start-Process -FilePath $GhPath -ArgumentList $GhArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut.FullName -RedirectStandardError $tmpErr.FullName
        $stdout = Get-Content $tmpOut.FullName -Raw
        $stderr = Get-Content $tmpErr.FullName -Raw
        $out = @($stdout, $stderr) -join ""
        $exitCode = $proc.ExitCode
    } finally {
        Remove-Item $tmpOut.FullName -ErrorAction SilentlyContinue
        Remove-Item $tmpErr.FullName -ErrorAction SilentlyContinue
    }

    if ($exitCode -ne 0) {
        $joined = $GhArgs -join " "
        $msg = ($out | Out-String).Trim()
        if ($msg -match "not logged into any GitHub hosts") {
            throw "gh is installed but not authenticated. Run: gh auth login"
        }
        if ($msg -match "workflow .* not found on the default branch") {
            throw "Workflow not found on default branch. Push .github/workflows/io-parity-hdf5.yml to the default branch first, or run workflow by name that already exists on default."
        }
        throw "gh command failed: gh $joined`n$msg"
    }
    return ($out | Out-String)
}

function ConvertFrom-JsonStrict {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Raw,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $trimmed = $Raw.Trim()
    if (-not ($trimmed.StartsWith("[") -or $trimmed.StartsWith("{"))) {
        throw "Expected JSON from $Context but received non-JSON output:`n$Raw"
    }
    return $trimmed | ConvertFrom-Json
}

Require-File $GhPath
Require-File "scripts/complete_pm001_after_ci.ps1"

Write-Host "Checking gh auth status..."
$authOut = Invoke-Gh -GhArgs @("auth", "status")

Write-Host "Dispatching workflow: $WorkflowFile"
$null = Invoke-Gh -GhArgs @("workflow", "run", $WorkflowFile)

Write-Host "Locating latest workflow_dispatch run..."
$json = Invoke-Gh -GhArgs @(
    "run", "list",
    "--workflow", $WorkflowName,
    "--event", "workflow_dispatch",
    "--limit", "1",
    "--json", "databaseId,url,status,conclusion,headSha,createdAt"
)

$runs = ConvertFrom-JsonStrict -Raw (($json | Out-String).Trim()) -Context "gh run list"
if (-not $runs -or $runs.Count -lt 1) {
    throw "No workflow_dispatch run found for '$WorkflowName'."
}

$run = $runs[0]
$runId = [string]$run.databaseId
$runUrl = [string]$run.url
$runSha = [string]$run.headSha

Write-Host "Watching run $runId ..."
$null = Invoke-Gh -GhArgs @("run", "watch", $runId, "--exit-status")

Write-Host "Refreshing run result..."
$json2 = Invoke-Gh -GhArgs @(
    "run", "view", $runId,
    "--json", "url,status,conclusion,headSha"
)
$run2 = ConvertFrom-JsonStrict -Raw (($json2 | Out-String).Trim()) -Context "gh run view"

if ($run2.conclusion -ne "success") {
    throw "Workflow run did not succeed. URL: $($run2.url)"
}

$backfillArgs = @(
    "-SmokeRunUrl", [string]$run2.url,
    "-FullRunUrl", [string]$run2.url,
    "-CommitSha", [string]$run2.headSha,
    "-UpdatedBy", $UpdatedBy,
    "-AutoFillFromRunUrl"
)

if ($NoStatusFlip) {
    $backfillArgs += "-NoStatusFlip"
}
if ($Preview) {
    $backfillArgs += "-Preview"
}

Write-Host "Running PM-001 backfill script..."
& "scripts/complete_pm001_after_ci.ps1" @backfillArgs

Write-Host "Done."
Write-Host "Run URL: $($run2.url)"
