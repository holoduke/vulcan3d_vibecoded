@echo off
setlocal EnableDelayedExpansion

rem vcvars internally calls vswhere; ensure the Installer dir is on PATH.
set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer"

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" goto :nov

set "CONFIG=Debug"
if not "%~1"=="" set "CONFIG=%~1"

rem %~dp0 has trailing backslash; strip it so quotes don't escape.
set "ROOT=%~dp0"
if "!ROOT:~-1!"=="\" set "ROOT=!ROOT:~0,-1!"
set "BUILD=!ROOT!\build"

call "!VCVARS!" > NUL
if errorlevel 1 goto :vcfail

if not exist "!BUILD!" goto :doconfig
goto :dobuild

:doconfig
echo === Configuring ===
cmake -S "!ROOT!" -B "!BUILD!" -G Ninja -DCMAKE_BUILD_TYPE=!CONFIG! -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
if errorlevel 1 exit /b 1

:dobuild
echo === Building ===
cmake --build "!BUILD!"
if errorlevel 1 exit /b 1

echo === Running tests ===
ctest --test-dir "!BUILD!" --output-on-failure
if errorlevel 1 exit /b 1

echo === Done ===
endlocal
exit /b 0

:nov
echo vcvars64.bat not found
exit /b 1
:vcfail
echo vcvars failed
exit /b 1
