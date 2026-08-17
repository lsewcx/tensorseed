$ErrorActionPreference = "Stop"

$venvActivate = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

conan profile detect --force
Push-Location "$PSScriptRoot\..\cpp"
try {
    conan install . --output-folder=build --build=missing
    cmake --preset conan-default
    cmake --build --preset conan-release
}
finally {
    Pop-Location
}
