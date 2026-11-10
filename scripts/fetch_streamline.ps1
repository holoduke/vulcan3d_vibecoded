# Downloads the NVIDIA Streamline SDK (DLSS) into external/streamline.
# Usage (from the project root):  powershell -File scripts/fetch_streamline.ps1
$ErrorActionPreference = "Stop"
$ver = "v2.14.16"
$url = "https://github.com/NVIDIA-RTX/Streamline/releases/download/$ver/streamline-sdk-$ver.zip"
$root = Split-Path -Parent $PSScriptRoot
$dl = Join-Path $root "external/_dl"
$dst = Join-Path $root "external/streamline"
New-Item -ItemType Directory -Force $dl | Out-Null
$zip = Join-Path $dl "streamline.zip"
if (-not (Test-Path $zip)) { Write-Host "Downloading Streamline $ver ..."; Invoke-WebRequest $url -OutFile $zip }
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Expand-Archive $zip -DestinationPath $dst
Write-Host "Streamline extracted to $dst - re-run build.ps1 to enable DLSS (fsr2_enabled = 1, fsr_backend = 2)."
