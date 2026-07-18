param(
    [Parameter(Mandatory = $true)]
    [string]$Binary,
    [Parameter(Mandatory = $true)]
    [string]$Target,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDirectory = "dist",
    [ValidateSet("auto", "zip", "tar.gz")]
    [string]$ArchiveFormat = "auto"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Target -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "Target contains unsupported characters: $Target"
}
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+([+-][A-Za-z0-9.-]+)?$') {
    throw "Version is not a supported semantic version: $Version"
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BinaryPath = (Resolve-Path -LiteralPath $Binary).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path

$PackageName = "talos-xii-$Version-$Target"
$StagingDirectory = Join-Path $OutputRoot $PackageName
if (Test-Path -LiteralPath $StagingDirectory) {
    $ResolvedStaging = (Resolve-Path -LiteralPath $StagingDirectory).Path
    if (-not $ResolvedStaging.StartsWith($OutputRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging directory outside output root: $ResolvedStaging"
    }
    Remove-Item -LiteralPath $ResolvedStaging -Recurse -Force
}

New-Item -ItemType Directory -Path $StagingDirectory | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StagingDirectory "config") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StagingDirectory "docs") | Out-Null

$PackagedBinary = Join-Path $StagingDirectory (Split-Path -Leaf $BinaryPath)
Copy-Item -LiteralPath $BinaryPath -Destination $PackagedBinary
if (-not $IsWindows) {
    & chmod +x $PackagedBinary
    if ($LASTEXITCODE -ne 0) {
        throw "chmod failed with exit code $LASTEXITCODE"
    }
}
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "README.md") -Destination $StagingDirectory
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "LICENSE") -Destination $StagingDirectory
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "data/config.json") -Destination (Join-Path $StagingDirectory "config/config.example.json")
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "data/pools.json") -Destination (Join-Path $StagingDirectory "config/pools.example.json")
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "docs/USAGE.md") -Destination (Join-Path $StagingDirectory "docs")
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "docs/REQUIREMENTS.md") -Destination (Join-Path $StagingDirectory "docs")

if ($ArchiveFormat -eq "auto") {
    $ArchiveFormat = if ($IsWindows) { "zip" } else { "tar.gz" }
}
$Extension = if ($ArchiveFormat -eq "zip") { ".zip" } else { ".tar.gz" }
$ArchivePath = Join-Path $OutputRoot "$PackageName$Extension"
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

if ($ArchiveFormat -eq "zip") {
    Compress-Archive -LiteralPath $StagingDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
} else {
    Push-Location $OutputRoot
    try {
        & tar -czf (Split-Path -Leaf $ArchivePath) $PackageName
        if ($LASTEXITCODE -ne 0) {
            throw "tar failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$Hash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$ChecksumPath = "$ArchivePath.sha256"
"$Hash  $(Split-Path -Leaf $ArchivePath)" | Set-Content -LiteralPath $ChecksumPath -Encoding ascii
Remove-Item -LiteralPath $StagingDirectory -Recurse -Force

Write-Output $ArchivePath
Write-Output $ChecksumPath
