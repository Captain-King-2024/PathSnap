param(
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [ValidateSet('student_test', 'evaluator_test')][string]$Split = 'student_test',
    [string]$Python = 'python',
    [switch]$BuiltIn
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    Write-Host "On Laptop 2, open http://<Laptop-1-IPv4>:${Port}/"
    Write-Host 'Use the IPv4 address of the active Wi-Fi or Ethernet adapter from ipconfig.'
    if ($BuiltIn) {
        & $Python -m api.server --host 0.0.0.0 --port $Port --split $Split
    } else {
        $oldSplit = $env:PATHSNAP_SPLIT
        try {
            $env:PATHSNAP_SPLIT = $Split
            & $Python -m uvicorn api.app --host 0.0.0.0 --port $Port
        } finally {
            $env:PATHSNAP_SPLIT = $oldSplit
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "PathSnap exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
