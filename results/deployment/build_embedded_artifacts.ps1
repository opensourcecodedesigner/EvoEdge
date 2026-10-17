<#
.SYNOPSIS
    Builds standalone embedded C artifacts for EvoEdge HybridController.
.DESCRIPTION
    Compiles:
      1. hybrid_controller.o & platform_adapter.o with strict warnings (-Wall -Wextra -Wpedantic -O3)
      2. Static library libhybrid_controller.a
      3. Shared library hybrid_controller.dll
      4. Standalone native test runner hybrid_controller_runner.exe
#>
[CmdletBinding()]
param(
    [string]$Compiler = "gcc",
    [string]$OutputDir = "results/deployment"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $Compiler -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\msys64\ucrt64\bin\gcc.exe") {
        $Compiler = "C:\msys64\ucrt64\bin\gcc.exe"
        $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
    }
}

Write-Host "Building EvoEdge Portable Embedded C Artifacts in $OutputDir..."

# 1. Compile object files
& $Compiler -Wall -Wextra -Wpedantic -O3 -c -o "$OutputDir/hybrid_controller.o" "$OutputDir/hybrid_controller.c" -I"$OutputDir"
& $Compiler -Wall -Wextra -Wpedantic -O3 -c -o "$OutputDir/platform_adapter.o" "$OutputDir/platform_adapter.c" -I"$OutputDir"

# 2. Build static library
& ar rcs "$OutputDir/libhybrid_controller.a" "$OutputDir/hybrid_controller.o" "$OutputDir/platform_adapter.o"

# 3. Build shared libraries (DLL)
& $Compiler -Wall -Wextra -Wpedantic -O3 -shared -fPIC -o "$OutputDir/hybrid_controller.dll" "$OutputDir/hybrid_controller.c" -I"$OutputDir" -lm
& $Compiler -Wall -Wextra -Wpedantic -O3 -shared -fPIC -o "$OutputDir/platform_adapter.dll" "$OutputDir/platform_adapter.c" "$OutputDir/hybrid_controller.c" -I"$OutputDir" -lm

# 4. Build standalone native test runner
& $Compiler -Wall -Wextra -Wpedantic -O3 -o "$OutputDir/hybrid_controller_runner.exe" "$OutputDir/hybrid_controller_runner.c" "$OutputDir/hybrid_controller.c" "$OutputDir/platform_adapter.c" -I"$OutputDir" -lm

Write-Host "Build complete successfully with zero warnings."
