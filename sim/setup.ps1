param(
    [switch]$Auto
)

$SimRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheRoot = Join-Path $SimRoot ".cache"
$TempRoot = Join-Path $SimRoot ".tmp"
$VenvRoot = Join-Path $SimRoot ".venv"
New-Item -ItemType Directory -Force -Path $CacheRoot, $TempRoot | Out-Null

$env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot "pycache"

python -B (Join-Path $SimRoot "scope_guard.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath (Join-Path $VenvRoot "Scripts\python.exe"))) {
    # Reuse the already validated local MuJoCo/OpenCV wheels while keeping any
    # newly installed packages and scripts inside sim/.venv.
    py -3.13 -m venv --system-site-packages $VenvRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$Python = Join-Path $VenvRoot "Scripts\python.exe"
$RequirementsFile = if ($Auto) {
    Join-Path $SimRoot "requirements-auto.txt"
} else {
    Join-Path $SimRoot "requirements.txt"
}
& $Python -B -m pip install -r $RequirementsFile
$InstallExitCode = $LASTEXITCODE
& $Python -B (Join-Path $SimRoot "scope_guard.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
exit $InstallExitCode
