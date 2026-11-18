param(
    [Parameter(Mandatory)][string]$HostAddress,
    [ValidateRange(1024, 65535)][int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$address = $null
if (!([System.Net.IPAddress]::TryParse($HostAddress, [ref]$address)) -or
    $address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
    throw 'Enter the IPv4 address of Laptop 1, without http:// or a port.'
}

$base = "http://${address}:${Port}"
try {
    $health = Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 5
} catch {
    throw "Cannot reach PathSnap at $base. Check the host address, port, trusted LAN and host firewall."
}
if ($health.service -ne 'PathSnap' -or $health.status -ne 'ok') {
    throw "The server at $base did not return the PathSnap health response."
}
$list = Invoke-RestMethod -Uri "$base/api/trips" -TimeoutSec 5
if (@($list.trips).Count -ne $health.trip_count -or $health.trip_count -lt 1) {
    throw 'The server returned an inconsistent trip list.'
}
$trip = [uri]::EscapeDataString($list.trips[0])
$route = Invoke-RestMethod -Uri "$base/api/match/$trip" -TimeoutSec 60
if ($route.trip -ne $list.trips[0] -or $route.observations -lt 1 -or
    @($route.matched_segments).Count -ne $route.observations) {
    throw 'The server returned an incomplete route.'
}
Write-Host "Connected to PathSnap. Split: $($health.split). Trips: $($health.trip_count)."
Write-Host "Test route $trip returned $($route.observations) matched observations."
Write-Host "Open $base/ in your browser."
