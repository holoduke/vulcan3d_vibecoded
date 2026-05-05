param(
    [string]$Config = "Debug",
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# Pull machine-level PATH and VULKAN_SDK into this session.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")
$env:VULKAN_SDK = [Environment]::GetEnvironmentVariable("VULKAN_SDK", "Machine")

# vcvars64.bat needs vswhere.exe; the VS Installer dir often isn't in PATH.
$vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if (Test-Path $vsInstaller) { $env:Path = "$vsInstaller;$env:Path" }

# Source vcvars64.bat into this PowerShell session.
Write-Host "Loading MSVC environment..." -ForegroundColor Cyan
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match "^(.+?)=(.*)$") {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$build = Join-Path $root "build"

if ($Clean -and (Test-Path $build)) {
    Write-Host "Cleaning $build" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $build
}

if (-not (Test-Path $build)) {
    Write-Host "Configuring (CMake -> Ninja, $Config)" -ForegroundColor Cyan
    # NOTE: splat the CMake args. Inline `-DCMAKE_BUILD_TYPE=$Config` is
    # parsed unreliably by PowerShell when other inline flags are present
    # (the `=` sign confuses argument-mode parsing) — `$Config` ends up
    # passed literally as the string "$Config", and CMake stores that in
    # the cache, which later trips a "$<CONFIG:$Config>" generator-expression
    # error during Generate. Splatting forces each element to be a discrete
    # string argument with `$Config` already expanded.
    $cmakeArgs = @("-S", $root, "-B", $build, "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=$Config",
                   "-DCMAKE_C_COMPILER=cl",
                   "-DCMAKE_CXX_COMPILER=cl")
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
}

Write-Host "Building" -ForegroundColor Cyan
cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "build failed" }

if ($Test) {
    Write-Host "Running tests" -ForegroundColor Cyan
    ctest --test-dir $build --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}
