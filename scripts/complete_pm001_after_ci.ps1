param(
    [Parameter(Mandatory = $true)]
    [string]$SmokeRunUrl,

    [Parameter(Mandatory = $true)]
    [string]$FullRunUrl,

    [string]$CommitSha = "",

    [Parameter(Mandatory = $true)]
    [string]$UpdatedBy,

    [string]$SmokePartitionedJobUrl = "",
    [string]$SmokeMpiJobUrl = "",
    [string]$SmokePartitionedArtifactUrl = "",
    [string]$SmokeMpiArtifactUrl = "",
    [string]$FullPartitionedJobUrl = "",
    [string]$FullMpiJobUrl = "",
    [string]$FullPartitionedArtifactUrl = "",
    [string]$FullMpiArtifactUrl = "",
    [string]$TemplatePath = "docs/mfem-w2-io-ci-backfill-template.md",
    [string]$MatrixPath = "docs/mfem-parity-matrix-template.md",
    [string]$TrackerPath = "MFEM_ALIGNMENT_TRACKER.md",
    [switch]$AutoFillFromRunUrl,
    [switch]$AllowPlaceholderUrls,
    [switch]$NoStatusFlip,
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

function Test-HttpUrl {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    try {
        $uri = [Uri]$Value
        return ($uri.IsAbsoluteUri -and ($uri.Scheme -eq "http" -or $uri.Scheme -eq "https"))
    } catch {
        return $false
    }
}

function Require-HttpUrl {
    param(
        [string]$Name,
        [string]$Value
    )

    if (-not (Test-HttpUrl $Value)) {
        throw "$Name must be an absolute http/https URL. Got: '$Value'"
    }
}

function Resolve-CommitSha {
    param([string]$InputSha)

    if (-not [string]::IsNullOrWhiteSpace($InputSha)) {
        return $InputSha
    }

    try {
        $sha = (git rev-parse --verify HEAD).Trim()
        if ([string]::IsNullOrWhiteSpace($sha)) {
            throw "git returned empty commit SHA"
        }
        return $sha
    } catch {
        throw "CommitSha is empty and current git commit SHA could not be resolved. Pass -CommitSha explicitly."
    }
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

function Set-LineValue {
    param(
        [string]$Text,
        [string]$Prefix,
        [string]$Value
    )
    $escaped = [Regex]::Escape($Prefix)
    return [Regex]::Replace($Text, "(?m)^$escaped.*$", "$Prefix$Value")
}

Require-File $TemplatePath
if (-not $NoStatusFlip) {
    Require-File $MatrixPath
    Require-File $TrackerPath
}

Require-HttpUrl -Name "SmokeRunUrl" -Value $SmokeRunUrl
Require-HttpUrl -Name "FullRunUrl" -Value $FullRunUrl
$CommitSha = Resolve-CommitSha -InputSha $CommitSha

if ($AllowPlaceholderUrls -and -not $Preview) {
    throw "-AllowPlaceholderUrls is only permitted with -Preview."
}

if (($SmokeRunUrl -match "example/" -or $FullRunUrl -match "example/") -and -not $AllowPlaceholderUrls) {
    throw "Run URLs appear to be placeholders. Provide real workflow URLs before applying backfill, or use -AllowPlaceholderUrls with -Preview."
}

if ($AutoFillFromRunUrl) {
    if ([string]::IsNullOrWhiteSpace($SmokePartitionedJobUrl)) {
        $SmokePartitionedJobUrl = $SmokeRunUrl
    }
    if ([string]::IsNullOrWhiteSpace($SmokeMpiJobUrl)) {
        $SmokeMpiJobUrl = $SmokeRunUrl
    }
    if ([string]::IsNullOrWhiteSpace($SmokePartitionedArtifactUrl)) {
        $SmokePartitionedArtifactUrl = "$SmokeRunUrl#artifacts"
    }
    if ([string]::IsNullOrWhiteSpace($SmokeMpiArtifactUrl)) {
        $SmokeMpiArtifactUrl = "$SmokeRunUrl#artifacts"
    }
    if ([string]::IsNullOrWhiteSpace($FullPartitionedJobUrl)) {
        $FullPartitionedJobUrl = $FullRunUrl
    }
    if ([string]::IsNullOrWhiteSpace($FullMpiJobUrl)) {
        $FullMpiJobUrl = $FullRunUrl
    }
    if ([string]::IsNullOrWhiteSpace($FullPartitionedArtifactUrl)) {
        $FullPartitionedArtifactUrl = "$FullRunUrl#artifacts"
    }
    if ([string]::IsNullOrWhiteSpace($FullMpiArtifactUrl)) {
        $FullMpiArtifactUrl = "$FullRunUrl#artifacts"
    }
}

$now = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

$template = Get-Content $TemplatePath -Raw
$template = Set-LineValue -Text $template -Prefix "- Workflow run URL (smoke): " -Value $SmokeRunUrl
$template = Set-LineValue -Text $template -Prefix "- Workflow run URL (full): " -Value $FullRunUrl
$template = Set-LineValue -Text $template -Prefix "- Commit SHA: " -Value $CommitSha
$template = Set-LineValue -Text $template -Prefix "- Trigger type: " -Value "workflow_dispatch/schedule"
$template = Set-LineValue -Text $template -Prefix "- Updated by: " -Value $UpdatedBy
$template = Set-LineValue -Text $template -Prefix "- Updated at: " -Value $now

if (-not [string]::IsNullOrWhiteSpace($SmokePartitionedJobUrl)) {
    Require-HttpUrl -Name "SmokePartitionedJobUrl" -Value $SmokePartitionedJobUrl
    $template = Set-LineValue -Text $template -Prefix "- Partitioned job URL: " -Value $SmokePartitionedJobUrl
}
if (-not [string]::IsNullOrWhiteSpace($SmokeMpiJobUrl)) {
    Require-HttpUrl -Name "SmokeMpiJobUrl" -Value $SmokeMpiJobUrl
    $template = Set-LineValue -Text $template -Prefix "- MPI job URL: " -Value $SmokeMpiJobUrl
}
if (-not [string]::IsNullOrWhiteSpace($SmokePartitionedArtifactUrl)) {
    Require-HttpUrl -Name "SmokePartitionedArtifactUrl" -Value $SmokePartitionedArtifactUrl
    $template = Set-LineValue -Text $template -Prefix "- Artifact URL (partitioned): " -Value "$SmokePartitionedArtifactUrl (io-parity-smoke-partitioned)"
}
if (-not [string]::IsNullOrWhiteSpace($SmokeMpiArtifactUrl)) {
    Require-HttpUrl -Name "SmokeMpiArtifactUrl" -Value $SmokeMpiArtifactUrl
    $template = Set-LineValue -Text $template -Prefix "- Artifact URL (mpi): " -Value "$SmokeMpiArtifactUrl (io-parity-smoke-mpi)"
}

$fullSectionStart = $template.IndexOf("## Full Tier Evidence")
$closureSectionStart = $template.IndexOf("## PM-001 Closure Checklist")
if ($fullSectionStart -ge 0 -and $closureSectionStart -gt $fullSectionStart) {
    $fullSection = $template.Substring($fullSectionStart, $closureSectionStart - $fullSectionStart)

    if (-not [string]::IsNullOrWhiteSpace($FullPartitionedJobUrl)) {
        Require-HttpUrl -Name "FullPartitionedJobUrl" -Value $FullPartitionedJobUrl
        $fullSection = Set-LineValue -Text $fullSection -Prefix "- Partitioned job URL: " -Value $FullPartitionedJobUrl
    }
    if (-not [string]::IsNullOrWhiteSpace($FullMpiJobUrl)) {
        Require-HttpUrl -Name "FullMpiJobUrl" -Value $FullMpiJobUrl
        $fullSection = Set-LineValue -Text $fullSection -Prefix "- MPI job URL: " -Value $FullMpiJobUrl
    }
    if (-not [string]::IsNullOrWhiteSpace($FullPartitionedArtifactUrl)) {
        Require-HttpUrl -Name "FullPartitionedArtifactUrl" -Value $FullPartitionedArtifactUrl
        $fullSection = Set-LineValue -Text $fullSection -Prefix "- Artifact URL (partitioned): " -Value "$FullPartitionedArtifactUrl (io-parity-full-partitioned)"
    }
    if (-not [string]::IsNullOrWhiteSpace($FullMpiArtifactUrl)) {
        Require-HttpUrl -Name "FullMpiArtifactUrl" -Value $FullMpiArtifactUrl
        $fullSection = Set-LineValue -Text $fullSection -Prefix "- Artifact URL (mpi): " -Value "$FullMpiArtifactUrl (io-parity-full-mpi)"
    }

    $template = $template.Substring(0, $fullSectionStart) + $fullSection + $template.Substring($closureSectionStart)
}

$template = $template -replace "- \[ \]", "- [x]"

$matrix = ""
$tracker = ""
if (-not $NoStatusFlip) {
    $matrix = Get-Content $MatrixPath -Raw
    $matrix = $matrix -replace "\| PM-001 \|([^\n]*?)\| partial \|", "| PM-001 |$1| complete |"
    $matrix = $matrix -replace "pending CI artifact links for closure\.", "CI backfill completed (see docs/mfem-w2-io-ci-backfill-template.md)."

    $tracker = Get-Content $TrackerPath -Raw
    $ciNote = "  - CI backfill completed: smoke=$SmokeRunUrl ; full=$FullRunUrl"
    if ($tracker -notmatch [Regex]::Escape($ciNote)) {
        $tracker = $tracker + "`r`n" + $ciNote + "`r`n"
    }
}

if ($Preview) {
    Write-Host "Preview mode: no files written."
    Write-Host "Would update:"
    Write-Host "  Template: $TemplatePath"
    if ($NoStatusFlip) {
        Write-Host "  Matrix:   (skipped by -NoStatusFlip)"
        Write-Host "  Tracker:  (skipped by -NoStatusFlip)"
    } else {
        Write-Host "  Matrix:   $MatrixPath"
        Write-Host "  Tracker:  $TrackerPath"
    }
    Write-Host ""
    Write-Host "Quick run examples:"
    Write-Host "  Preview: .\scripts\complete_pm001_after_ci.ps1 -SmokeRunUrl <smoke> -FullRunUrl <full> -UpdatedBy <name> -AutoFillFromRunUrl -Preview"
    Write-Host "  Apply:   .\scripts\complete_pm001_after_ci.ps1 -SmokeRunUrl <smoke> -FullRunUrl <full> -UpdatedBy <name> -AutoFillFromRunUrl"
    Write-Host "  Apply (template only): .\scripts\complete_pm001_after_ci.ps1 -SmokeRunUrl <smoke> -FullRunUrl <full> -UpdatedBy <name> -AutoFillFromRunUrl -NoStatusFlip"
    exit 0
}

Set-Content -Path $TemplatePath -Value $template -Encoding UTF8
if (-not $NoStatusFlip) {
    Set-Content -Path $MatrixPath -Value $matrix -Encoding UTF8
    Set-Content -Path $TrackerPath -Value $tracker -Encoding UTF8
}

Write-Host "PM-001 CI backfill applied:"
Write-Host "  Template: $TemplatePath"
if ($NoStatusFlip) {
    Write-Host "  Matrix:   skipped (-NoStatusFlip)"
    Write-Host "  Tracker:  skipped (-NoStatusFlip)"
} else {
    Write-Host "  Matrix:   $MatrixPath"
    Write-Host "  Tracker:  $TrackerPath"
}
