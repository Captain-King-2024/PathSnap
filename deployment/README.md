# Two-laptop demo

Laptop 1 runs the API, model and data. Laptop 2 opens the viewer in its browser. Only Laptop 1 needs Python or the PathSnap dataset. Connect both laptops to the same trusted Wi-Fi, Ethernet LAN, or hotspot that allows devices to reach each other.

On Laptop 1, run from the repository root:

```powershell
.\deployment\start-host.ps1
ipconfig
```

Keep the server running. Run `ipconfig` in a second terminal, and use the IPv4 address of the active Wi-Fi or Ethernet adapter. To check the host itself, open `http://127.0.0.1:8000/api/health`.

On Laptop 2, open `http://HOST_IP:8000/` in a browser, replacing `HOST_IP` with Laptop 1's IPv4 address. Select a trip and run **Match route**. If you have a copy of the connection-check script, run it from PowerShell to verify the API and one complete prediction:

```powershell
.\deployment\check-client.ps1 -HostAddress HOST_IP
```

If the script or page times out, check that both laptops are on the same network, the host server is still running, and the host address is correct. A Windows Firewall prompt should allow Python on the trusted Private network only. On a managed campus network, client isolation can block laptop-to-laptop traffic even when both use the same Wi-Fi. Try an approved hotspot or another trusted LAN. If port 8000 is occupied, choose another with `-Port` and use the same port on both laptops.

The host defaults to the packaged student-test workflow. For the evaluator demo, prepare the evaluator inputs and start the host with `-Split evaluator_test`. See the [API setup](../api/README.md).
