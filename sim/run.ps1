param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SimArguments
)

$SimRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheRoot = Join-Path $SimRoot ".cache"
$TempRoot = Join-Path $SimRoot ".tmp"
New-Item -ItemType Directory -Force -Path $CacheRoot, $TempRoot | Out-Null

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot "pycache"
$env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
$env:XDG_CACHE_HOME = $CacheRoot
$env:XDG_CONFIG_HOME = Join-Path $CacheRoot "config"
$env:YOLO_CONFIG_DIR = Join-Path $CacheRoot "ultralytics"
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:MPLCONFIGDIR = Join-Path $CacheRoot "matplotlib"
$env:TEMP = $TempRoot
$env:TMP = $TempRoot

$VenvPython = Join-Path $SimRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
& $Python -B (Join-Path $SimRoot "scope_guard.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $Python -B (Join-Path $SimRoot "sim_run.py") @SimArguments
$RunExitCode = $LASTEXITCODE
& $Python -B (Join-Path $SimRoot "scope_guard.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
exit $RunExitCode
