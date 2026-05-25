<!-- ADVANCED_ENHANCEMENTS_GUIDE.md -->
# Advanced Port Scanner Enhancements - Level 3

**Status:** Implementation Guide | **Complexity:** Medium-High | **Requirements:** Admin/Root for some features

---

## Overview: Three Advanced Enhancements

This document describes three sophisticated enhancements to the port scanner, addressing the operational signatures that remain visible despite previous enhancements:

| Enhancement | Issue Addressed | Visibility | Privilege | Complexity |
| --- | --- | --- | --- | --- |
| **4. SMB Real Negotiation** | Static port→service assumption | Reduced | User | Medium |
| **5. Dynamic Service Probes** | Non-standard ports invisible | Eliminated | User | Medium |
| **6. SYN Stealth Scanning** | Full handshake in logs (Event 5156) | Eliminated | Admin | High |

---

## Challenge 1: Operational Signatures

Current limitations with previous "stealth" enhancements:

### TCP Fingerprinting

```text
BEFORE (With Jitter):
  [Event 5156] Connection to 192.168.1.50:445 BLOCKED (TCP SYN)
  [Event 5156] Connection to 192.168.1.50:3389 BLOCKED (TCP SYN)
  [Event 5156] Connection to 192.168.1.50:5985 BLOCKED (TCP SYN)
  → IDS sees: port 445,3389,5985 being tested (SMB/RDP/WinRM fingerprint)

AFTER (With Jitter):
  Same events, just with unpredictable timing between them
  → Still identifiable as service fingerprinting
```

**Solution: SYN Stealth** - Send only SYN flag, don't complete handshake

- No completed connection = no connection logging (usually)
- Half-open = only received packet logging (much less suspicious)
- Risk: Some systems trigger on unreplied SYN-ACK (port closed status)

---

### Payload Consistency
```
BEFORE:
  Fixed banner read: Always exactly 1024 bytes → Honeypot signature
  Same probe order: HTTP, SSH, SMB, RDP (consistent pattern)
  Same timeout: Always 2.0 seconds (pattern)

AFTER (With dynamic probes):
  Variable reads: 512-8192 bytes (random selection each port)
  Randomized probes: Different order, skip unavailable services
  Jittered timeout: 0.3-1.5 seconds random per port
```

**Solution: Dynamic Service Probes** - Send service-specific negotiation

- Detect HTTP by sending HTTP HEAD request (not port 80 assumption)
- Detect SSH by sending SSH version negotiation
- Detect SMB by real negotiation (impacket)
- Non-predictable payload = harder to detect via IDS

---

### Timing Bounds

```text
BEFORE (Predictable):
  Port scan delay: Always 1-100ms (tight range)
  Inter-target beacon: Always 0.3-2.0s (tight range)
  Human analyst sees: Regular pattern, not human behavior

AFTER (Unpredictable):
  Port scan: 1-300ms with 15% chance of 150-800ms
  Inter-target: Mixed 3-15s, 0.5-3s, 50-500ms, 10-100ms (varies)
  → Baseline over minutes/hours shows irregular pattern (harder to detect)
```

**Solution: Wide Variance Jittering** - Make ranges asymmetric and unpredictable

- 70% normal speed, 15% very fast, 15% very slow
- Mixes human-paced clicks (~500ms-2s) with scripted speed
- Avoids tight patterns that baseline analysis can detect

---

## Enhancement 4: SMB Real Negotiation

### Problem

- Port 445 detected but SMB version unknown
- Can't distinguish SMB1 (very vulnerable) from SMB3 (hardened)
- No dialect information for version-specific exploitation

### Solution

Use impacket library to perform REAL SMB protocol negotiation:

```python
def _negotiate_smb_dialect(target_ip: str, port: int = 445) -> Optional[str]:
    """
    Real SMB negotiation to detect dialect
    Returns: 'SMB1', 'SMB2', 'SMB3'
    """
    try:
        from impacket.smbconnection import SMBConnection
        conn = SMBConnection(target_ip, target_ip, sess_port=port)
        
        dialect = conn.getDialect()
        # Returns: 1=SMB1, 2=SMB2, 3+ =SMB3
        
        conn.close()
    except:
        return None
```

### What This Reveals

```text
Port 445 open:
  - SMB Signing: REQUIRED (blocks WMI exploitation)  ✗
  - SMB Signing: NOT REQUIRED (allows exploitation)  ✓
  - SMB Dialect: SMB1 (very vulnerable, old Windows) ✓✓
  - SMB Dialect: SMB3 (modern, hardened)              ✗

Allows automatic routing:
  SMB1 + No Signing → Highest priority
  SMB3 + Signing → Lower priority
```

### Implementation

```python
scanner = PortScanner(enable_dynamic_probes=True)
result = scanner.scan('192.168.1.50')

for port in result.open_ports:
    if port.port == 445:
        print(f"SMB Dialect: {port.smb_dialect}")
        print(f"Product: {port.product}")
        # Output: 
        # SMB Dialect: SMB2
        # Product: Microsoft SMB (SMB2)
```

### Requirements

- impacket library (already used in Phase 1 & 2)
- No privilege elevation needed
- Slightly slower (extra negotiation overhead ~200-500ms)

---

## Enhancement 5: Dynamic Service Identification

### Problem

- Port → Service mapping assumes standard ports
- Port 445 is SMB, port 22 is SSH, port 80 is HTTP
- But services run on non-standard ports: SMB on 8888, SSH on 2222, HTTP on 8080
- Dynamic detection impossible

### Solution

Send protocol-specific probes to detect service regardless of port:

```python
def _detect_service_by_probe(target_ip: str, port: int) -> Tuple[str, str, str]:
    """
    Send protocol-specific handshakes to detect service
    Returns: (service_name, product, version)
    """
    probes = {
        'ssh': b'SSH-2.0-portscanner\r\n',      # Trigger SSH response
        'http': b'HEAD / HTTP/1.0\r\n....\r\n',  # HTTP HEAD request
        'tls': b'\x16\x03\x01...',               # TLS ClientHello
    }
    
    for probe_name, probe_bytes in probes.items():
        sock = socket.create_connection((target_ip, port))
        sock.send(probe_bytes)
        response = sock.recv(2048)
        sock.close()
        
        # If response matches expected protocol → service found!
```

### Service Detection Examples

#### Example 1: SSH on Port 8888 (Non-Standard)

```text
Traditional: Port 8888 → Assume HTTP → Try HTTP probes → Fail
Enhanced:   Port 8888 → Send SSH probe → Get "SSH-2.0-..." → SUCCESS!
Result: SSH service correctly identified on non-standard port
```

#### Example 2: HTTP Proxy on Port 3128

```text
Traditional: Port 3128 → Unknown service (not in PORT_SERVICES)
Enhanced:   Port 3128 → Send HTTP probe → Get "HTTP/1.1 ..." → SUCCESS!
Result: Enhanced HTTP detected on proxy port
```

#### Example 3: Multiple Services on Single Host

```text
192.168.1.50:
  - Port 445: SMB (standard)
  - Port 8888: SMB (administrator's alternate compliance port)
  - Port 22: SSH (standard)
  - Port 2222: SSH (jail/container SSH)
  - Port 80: Apache (standard)
  - Port 8080: IIS (admin panel)

OLD SCANNER:
  Found: 445 (SMB), 22 (SSH), 80 (HTTP)
  → 3 vectors

ENHANCED SCANNER WITH PROBES:
  Found: 445, 8888 (both SMB), 22, 2222 (both SSH), 80, 8080 (both HTTP)
  → 6 vectors (2x coverage)
```

### Implementation
```python
# Enable dynamic probes (default: True)
scanner = PortScanner(enable_dynamic_probes=True)
result = scanner.scan('192.168.1.50', 
                     ports=list(range(8000, 9000)))  # Scan range

# For each port:
for port in result.open_ports:
    print(f"{port.port}: {port.service.value()}, detected by {port.service_detected_by}")
    # Output:
    # 8000: HTTP, detected by probe
    # 8001: SSH, detected by probe
    # 8002: Unknown, detected by port
```

### Operational Benefit
- Covers 100% of running services (not just standard ports)
- Enables comprehensive attack surface mapping
- Finds "hidden" admin services (compliance bypass)

### Overhead
- ~100-200ms per port extra (sending/receiving probes)
- Cumulative: +5-10 seconds for 50-port scan

---

## Enhancement 6: SYN Stealth Scanning

### The Problem: Connection Logging

**Standard TCP Scan:**
```
C → S: SYN (port scan)
S → C: SYN-ACK (port open)
C → S: ACK (complete handshake)  ← COMPLETES CONNECTION
S → C: FIN (close)

Result in Windows Event Log:
  [Event 5156] Inbound TCP connection: SRC=attacker, DST=target:445
  → Clearly on firewall/IDS logs
```

**SYN Stealth Scan:**
```
C → S: SYN (port scan)
S → C: SYN-ACK (port open)           ← Received (not replied to!)
C → S: RST (send reset without ACK)  ← Don't complete handshake

Result in Windows Event Log:
  [Event 5156] FILTERED inbound packet (port blocked) OR
  [No connection logged because handshake wasn't completed]
  → May not appear in connection logs at all
```

### Implementation Requirements

```python
# Requires BOTH of:
# 1. Elevated privileges (admin/root)
# 2. scapy library (for raw packet crafting)

def _check_port_syn_stealth(target_ip: str, port: int) -> Optional[PortStatus]:
    """
    SYN stealth scan - half-open connection
    Requires: os.geteuid() == 0 (root) or ctypes.windll.shell32.IsUserAnAdmin()
    """
    if not SCAPY_AVAILABLE:
        return None  # Fall back to TCP scan
    
    if not is_root():
        return None  # Requires admin
    
    from scapy.all import IP, TCP, sr1
    
    # Craft SYN-only packet (no ACK)
    sport = random.randint(49152, 65535)
    packet = IP(dst=target_ip) / TCP(sport=sport, dport=port, flags="S")
    
    # Send and listen for SYN-ACK
    response = sr1(packet, timeout=2.0, verbose=False)
    
    if response and response[TCP].flags == 0x12:  # SYN-ACK flag
        # Port is OPEN - Don't complete handshake!
        # Instead send RST to close
        rst = IP(dst=target_ip) / TCP(sport=sport, dport=port, flags="R")
        sr1(rst, timeout=0.5, verbose=False)
        
        return PortStatus(port=port, open=True, scan_method="SYN_STEALTH")
```

### Usage

```python
# Install scapy: pip install scapy
# Run as root/admin

scanner = PortScanner(enable_syn_stealth=True)
# Automatically uses SYN if root + scapy available
# Falls back to TCP if not

result = scanner.scan('192.168.1.50')
for port in result.open_ports:
    print(f"{port.port}: {port.scan_method}")
    # Output:
    # 445: SYN_STEALTH
    # 3389: SYN_STEALTH
    # 80: SYN_STEALTH
```

### Tradeoffs

**Advantages:**
- ✅ No completed connection logged (Event 5156 may not appear)
- ✅ Appears as port blocked/filtered (not reconnaissance)
- ✅ Very fast (half TCP overhead)
- ✅ Doesn't trigger connection-based IDS (only packet-based)

**Disadvantages:**
- ❌ Requires root/admin (not stealth if caught!)
- ❌ Some systems log unreplied SYN-ACK anyway
- ❌ Requires scapy library (larger dependency)
- ❌ Some IDS systems detect SYN-only patterns
- ❌ May not get response banners (half-open = no data exchange)

### When to Use

- ✓ Internal network scans (you have admin)
- ✓ Post-compromise reconnaissance (elevated privileges)
- ❌ External reconnaissance (no admin rights)
- ❌ Public networks (will be detected as suspicious)

---

## Operational Signatures: What STILL Remains Visible

Even with all 6 enhancements, some patterns remain detectable by sophisticated analysts:

### Still Visible (Hard to Hide)
1. **Port Selection Pattern**
   - Scanning 445, 3389, 5985, 22, etc. = Windows reconnaissance
   - Can't hide this without sacrificing coverage
   - Mitigation: Scan random ports in between

2. **Volume Pattern**
   - 50 targets × 13 ports = 650 connections in 2 minutes
   - Baseline analysis sees spike
   - Mitigation: Spread over longer time (stealthier but takes hours)

3. **Source IP Pattern**
   - Same IP scanning multiple targets = grouped reconnaissance
   - Mitigation: Use rotating IPs / proxy network

4. **Known-Vulnerable Versions**
   - Detecting Apache 2.4.0 triggers alerts (CVE database)
   - Mitigation: None (this is good for defenders!)

### Countermeasures by Defenders
- Monitor for multiple port failures (445, 22, 3389 all timeout)
- Large windows of no traffic followed by bursts
- Requests for unusual ports from internal tools
- Anomalies in connection logs for "never-connected-before" IPs

---

## Integration Examples

### Example 1: Complete Advanced Scan
```python
from network_propagation_port_scanner import PortScanner

# All enhancements enabled
scanner = PortScanner(
    timeout=2.0,
    max_workers=10,
    jitter_enabled=True,                # ✅ Timing evasion
    enable_syn_stealth=True,            # ✅ Half-open connections
    enable_dynamic_probes=True          # ✅ Protocol detection
)

# Scan with all advantages
result = scanner.scan('192.168.1.50')

for port in result.open_ports:
    print(f"{port.port}: {port.product} v{port.service_version}")
    print(f"  Detected by: {port.service_detected_by}")
    print(f"  Method: {port.scan_method}")
    print(f"  SMB Dialect: {port.smb_dialect}")
    
# Output:
# 445: Microsoft SMB (SMB3) v3
#   Detected by: probe
#   Method: SYN_STEALTH
#   SMB Dialect: SMB3
# 3389: Microsoft RDP vunknown
#   Detected by: port
#   Method: SYN_STEALTH
#   SMB Dialect: None
```

### Example 2: Bypassing Port Assumptions
```python
# Scan for SMB on ANY port (not just 445)
scanner = PortScanner(enable_dynamic_probes=True)
result = scanner.scan('192.168.1.50', 
                     ports=list(range(8000, 10000)))  # Ports 8000-10000

smb_instances = [p for p in result.open_ports 
                 if 'SMB' in str(p.service)]

print(f"Found {len(smb_instances)} SMB instances:")
for smb in smb_instances:
    print(f"  Port {smb.port}: {smb.product}")
    
# Output:
# Found 2 SMB instances:
#   Port 445: Microsoft SMB (SMB3)
#   Port 8888: Microsoft SMB (SMB2)
```

### Example 3: Network Reconnaissance with All Evasion
```python
from network_propagation_orchestrator import MultiVectorOrchestrator

orchestrator = MultiVectorOrchestrator()

# Phase 0 with all enhancements
result = orchestrator.full_propagation_chain(
    target_ip='192.168.1.50',
    # Scanner automatically uses advanced options
)

# Phases 1-2 automatically benefit from:
# - Accurate service detection (right exploit for right service)
# - SMB dialect info (version-specific tricks)
# - SYN stealth (less noisy)
```

---

## Deployment Checklist

### Before Deploying
- [ ] Review local laws/regulations on port scanning
- [ ] Ensure written authorization for targets
- [ ] Test on isolated lab network first
- [ ] Confirm impacket library installed (SMB negotiation)
- [ ] For SYN stealth: Confirm scapy installed + admin rights

### Installation
```bash
# Required for SMB negotiation
pip install impacket

# Optional for SYN stealth
pip install scapy

# Then test
python -c "from network_propagation_port_scanner import PortScanner; print('OK')"
```

### Operational Security
- [ ] Run from compromised internal system (not external)
- [ ] Use for post-compromise reconnaissance only
- [ ] Don't leave scanner running unsupervised
- [ ] Clean up network logs after scan (if applicable)
- [ ] Monitor for IDS/firewall alerts during scan

---

## Conclusion

The six enhancements create a highly sophisticated reconnaissance capability:

| Phase | Enhancement | Impact |
|-------|-------------|--------|
| 1 | Jittered timing | Timing analysis harder |
| 2 | Banner grabbing | Version detection accurate |
| 3 | IPv6 support | Broader network scope |
| 4 | SMB negotiation | SMB version detection |
| 5 | Dynamic probes | Non-standard ports detected |
| 6 | SYN stealth | Connection logging bypassed |

**Combined effect:** Comprehensive, stealthy, accurate reconnaissance of target network architecture and services.

**Remaining operational security:** Depends on network monitoring, threat hunting, and behavioral anomaly detection.

---

**Status:** Implementation Guide Complete | **Recommended:** Deploy in lab first, then operationally on authorized targets only
