$SimRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunTemp = Join-Path $SimRoot (".tmp\tests\pytest-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $RunTemp | Out-Null

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:TEMP = $RunTemp
$env:TMP = $RunTemp

$VenvPython = Join-Path $SimRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

try {
    & $Python -B (Join-Path $SimRoot "scope_guard.py")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $Python -B -c "import cv2, numpy, pytest; print('test dependencies ok')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Test dependencies are missing. Create sim\.venv and install sim\requirements.txt."
        exit $LASTEXITCODE
    }
    & $Python -B -m pytest -c (Join-Path $SimRoot "pytest.ini") $SimRoot
    $TestExitCode = $LASTEXITCODE
    & $Python -B (Join-Path $SimRoot "scope_guard.py")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    exit $TestExitCode
}
finally {
    $Resolved = [System.IO.Path]::GetFullPath($RunTemp)
    $Allowed = [System.IO.Path]::GetFullPath((Join-Path $SimRoot ".tmp\tests")) +
        [System.IO.Path]::DirectorySeparatorChar
    if ($Resolved.StartsWith($Allowed, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $Resolved)) {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    }
}
