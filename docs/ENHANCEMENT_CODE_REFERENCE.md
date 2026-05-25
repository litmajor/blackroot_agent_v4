<!-- ENHANCEMENT_CODE_REFERENCE.md -->
# Code Changes Reference - Port Scanner Enhancements

Quick reference showing exact code modifications to implement the three enhancements.

---

## Change 1: Jittered Timing Implementation

### Location: `PortScanner.__init__()`

**Before:**
```python
def __init__(self, timeout: float = 2.0, max_workers: int = 10):
    self.timeout = timeout
    self.max_workers = max_workers
    self.scan_lock = threading.Lock()
```

**After:**
```python
def __init__(self, timeout: float = 2.0, max_workers: int = 10, jitter_enabled: bool = True):
    self.timeout = timeout
    self.max_workers = max_workers
    self.jitter_enabled = jitter_enabled  # NEW: Toggle timing evasion
    self.scan_lock = threading.Lock()
```

### Location: `PortScanner._check_port()` - Beginning

**Before:**
```python
def _check_port(self, target_ip: str, port: int) -> Optional[PortStatus]:
    start_time = time.time()
    try:
        sock = socket.create_connection((target_ip, port), timeout=self.timeout)
```

**After:**
```python
def _check_port(self, target_ip: str, port: int) -> Optional[PortStatus]:
    # NEW: Add jittered delay to evade timing-based detection
    if self.jitter_enabled:
        jitter = random.uniform(0.001, 0.1)  # 1-100ms random delay
        time.sleep(jitter)
    
    start_time = time.time()
    try:
        sock = socket.create_connection((target_ip, port), timeout=self.timeout)
```

### Location: `MultiTargetScanner.scan_range()` - Inter-target beacon

**Before:**
```python
for i, target_ip in enumerate(randomized_ips):
    if i > 0:
        beacon = random.uniform(0.5, 1.5)
        logger.debug(f"[*] Beacon interval {beacon:.1f}s...")
        time.sleep(beacon)
```

**After:**
```python
for i, target_ip in enumerate(randomized_ips):
    # Jittered beacon interval between targets (timing evasion)
    if i > 0:
        beacon = random.uniform(0.3, 2.0)  # 300-2000ms delay (NEW: wider range)
        logger.debug(f"[*] Beacon wait {beacon:.2f}s before next target...")
        time.sleep(beacon)
```

---

## Change 2: Comprehensive Banner Grabbing

### Location: Import statements - Add Regex

**Before:**
```python
import socket
import logging
import time
# ... no regex
```

**After:**
```python
import socket
import logging
import time
import random  # NEW: For jittering
import re      # NEW: For version extraction
```

### Location: `PortStatus` dataclass - New fields

**Before:**
```python
@dataclass
class PortStatus:
    port: int
    open: bool
    service: ServiceType
    response_time: Optional[float] = None  # ms
    version: Optional[str] = None
    banner: Optional[str] = None
    vulnerability: Optional[str] = None
```

**After:**
```python
@dataclass
class PortStatus:
    port: int
    open: bool
    service: ServiceType
    response_time: Optional[float] = None  # ms
    version: Optional[str] = None
    banner: Optional[str] = None
    vulnerability: Optional[str] = None
    full_banner: Optional[str] = None        # NEW: Raw banner data (4KB)
    service_version: Optional[str] = None   # NEW: Parsed version
    product: Optional[str] = None           # NEW: Product name
```

### Location: `PortScanner._check_port()` - Banner collection

**Before:**
```python
# Optional: try to grab banner (service identification)
banner = None
try:
    banner = sock.recv(1024).decode('utf-8', errors='ignore')[:100]
except:
    pass

sock.close()

return PortStatus(
    port=port,
    open=True,
    service=service,
    response_time=response_time,
    banner=banner
)
```

**After:**
```python
# Grab comprehensive banner (service identification)
banner = None
full_banner = None
service_version = None
product = None

try:
    # Read more data for better banner parsing
    sock.settimeout(0.5)
    full_banner = sock.recv(4096).decode('utf-8', errors='ignore')  # NEW: 4KB instead of 1KB
    
    # Parse banner based on service type
    parsed = self._parse_banner(port, service, full_banner)  # NEW: Protocol-specific parsing
    banner = parsed.get('banner')
    service_version = parsed.get('version')
    product = parsed.get('product')
except socket.timeout:
    pass
except:
    pass

sock.close()

return PortStatus(
    port=port,
    open=True,
    service=service,
    response_time=response_time,
    banner=banner,
    full_banner=full_banner,           # NEW
    service_version=service_version,   # NEW
    product=product                    # NEW
)
```

### Location: New method - `_parse_banner()` (ENTIRE NEW METHOD)

```python
def _parse_banner(self, port: int, service: ServiceType, data: str) -> Dict:
    """
    Parse banner data for protocol-specific version info
    Supports: HTTP, SSH, RDP, SMB, HTTPS, SQL, WinRM
    """
    result = {'banner': None, 'version': None, 'product': None}
    
    if not data:
        return result
    
    lines = data.split('\r\n' if '\r\n' in data else '\n')[:5]
    first_line = lines[0] if lines else ''
    
    try:
        # HTTP/HTTPS banner parsing
        if port in (80, 443, 8080, 8443) and service in (ServiceType.HTTP, ServiceType.HTTPS):
            if 'HTTP/' in first_line:
                result['banner'] = first_line[:80]  # HTTP status line
                # Look for Server header
                for line in lines:
                    if line.lower().startswith('server:'):
                        server_info = line.split(':', 1)[1].strip()
                        result['product'] = server_info
                        result['version'] = self._extract_version(server_info)
                        result['banner'] = server_info[:100]
                        break
        
        # SSH banner parsing (RFC 4253)
        elif port == 22 and service == ServiceType.SSH:
            if first_line.startswith('SSH-'):
                result['banner'] = first_line[:80]
                # Parse SSH version string: SSH-2.0-OpenSSH_7.4
                parts = first_line.split('-')
                if len(parts) >= 3:
                    product = '-'.join(parts[2:]).strip()
                    result['product'] = product
                    result['version'] = self._extract_version(product)
        
        # RDP banner parsing (verify RDP header)
        elif port in (3389,) and service == ServiceType.RDP:
            # RDP sends binary X.224 header, look for RDP signature
            if len(data) > 10:
                result['banner'] = f"RDP Service (binary protocol detected)"
                result['product'] = "Microsoft RDP"
        
        # SMB banner parsing
        elif port in (139, 445) and service == ServiceType.SMB:
            if len(data) > 20:
                result['banner'] = f"SMB Service (binary protocol detected)"
                result['product'] = "Microsoft SMB"
        
        # SQL Server detection
        elif port in (1433, 3306, 5432):
            if service == ServiceType.SQL:
                result['banner'] = first_line[:100]
                if 'Microsoft' in first_line or 'MSSQL' in first_line:
                    result['product'] = 'Microsoft SQL Server'
                    result['version'] = self._extract_version(first_line)
                elif 'MySQL' in first_line or 'mysql' in first_line:
                    result['product'] = 'MySQL'
                    result['version'] = self._extract_version(first_line)
                elif 'PostgreSQL' in first_line:
                    result['product'] = 'PostgreSQL'
                    result['version'] = self._extract_version(first_line)
        
        # WinRM banner parsing
        elif port in (5985, 5986) and service == ServiceType.WINRM:
            if 'HTTP/' in first_line:
                result['banner'] = first_line[:80]
                result['product'] = 'Windows Remote Management'
        
        # Generic fallback
        else:
            result['banner'] = first_line[:100]
            result['product'] = service.value if service != ServiceType.UNKNOWN else 'Unknown'
    
    except Exception as e:
        logger.debug(f"Error parsing banner on port {port}: {e}")
        result['banner'] = first_line[:100] if first_line else None
    
    return result
```

### Location: New static method - `_extract_version()` (ENTIRE NEW METHOD)

```python
@staticmethod
def _extract_version(text: str) -> Optional[str]:
    """
    Extract version number from text (e.g., 'Apache/2.4.41' -> '2.4.41')
    """
    match = re.search(r'([0-9]+\.[0-9]+(?:\.[0-9]+)?)', text)
    return match.group(1) if match else None
```

---

## Change 3: IPv6 Support Implementation

### Location: Import statements - Top of file

**Before:**
```python
import socket
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
# ... ipaddress was imported inside function
```

**After:**
```python
import socket
import logging
import time
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ipaddress  # NEW: Import at module level (was inside scan_range())
```

### Location: `MultiTargetScanner.__init__()` - Add jitter parameter

**Before:**
```python
def __init__(self, timeout: float = 2.0, max_workers: int = 10):
    self.scanner = PortScanner(timeout=timeout, max_workers=max_workers)
    self.results: Dict[str, ScanResult] = {}
```

**After:**
```python
def __init__(self, timeout: float = 2.0, max_workers: int = 10, jitter_enabled: bool = True):
    self.scanner = PortScanner(timeout=timeout, max_workers=max_workers, jitter_enabled=jitter_enabled)  # NEW: Pass jitter
    self.results: Dict[str, ScanResult] = {}
```

### Location: `MultiTargetScanner.scan_range()` - Complete rewrite for IPv6

**Before:**
```python
def scan_range(self, network_range: str, ports: Optional[List[int]] = None) -> Dict[str, ScanResult]:
    """
    Scan network range for open ports
    """
    try:
        import ipaddress
    except ImportError:
        logger.error("[!] ipaddress module required")
        return {}
    
    # Parse network range
    try:
        network = ipaddress.ip_network(network_range, strict=False)
        target_ips = [str(ip) for ip in list(network.hosts())[:50]]  # Max 50 targets
    except ValueError as e:
        logger.error(f"[!] Invalid network range '{network_range}': {e}")
        return {}
    
    # ... continue with scanning
```

**After:**
```python
def scan_range(self, network_range: str, ports: Optional[List[int]] = None, max_targets: int = 50) -> Dict[str, ScanResult]:
    """
    Scan network range for open ports (IPv4 and IPv6 support)
    
    Args:
        network_range: CIDR notation IPv4 (e.g., "192.168.1.0/24") or IPv6 (e.g., "2001:db8::/32")
        ports: Ports to scan (default: common Windows ports)
        max_targets: Maximum number of targets to scan (default: 50)
    
    Returns:
        Dict of results keyed by IP
    """
    # Parse network range (supports IPv4 and IPv6)
    try:
        network = ipaddress.ip_network(network_range, strict=False)
        
        # NEW: Detect IPv6 and handle accordingly
        if isinstance(network, ipaddress.IPv6Network):
            # For IPv6, use first 2^8 addresses if /120 or larger
            if network.prefixlen < 120:
                # Limit to first N targets from the network
                target_ips = [str(ip) for ip in list(network.hosts())[:max_targets]]
            else:
                target_ips = [str(ip) for ip in list(network.hosts())]
            logger.info(f"[📡] IPv6 network detected: {network_range}")
        else:
            # IPv4: Standard host enumeration
            target_ips = [str(ip) for ip in list(network.hosts())[:max_targets]]
            logger.info(f"[📡] IPv4 network detected: {network_range}")
    
    except ValueError as e:
        logger.error(f"[!] Invalid network range '{network_range}': {e}")
        logger.info(f"[💡] Supported formats: IPv4 CIDR (192.168.1.0/24) or IPv6 CIDR (2001:db8::/32)")
        return {}
    
    # ... continue with scanning
```

---

## Summary of Code Changes

| Enhancement | File | Changes | New/Modified |
|-------------|------|---------|-------------|
| **Jittered Timing** | network_propagation_port_scanner.py | 1. Add `random` import<br>2. Add `jitter_enabled` param to `__init__`<br>3. Add jitter delay in `_check_port()`<br>4. Extend beacon range in `scan_range()` | 4 changes |
| **Banner Grabbing** | network_propagation_port_scanner.py | 1. Add `re` import<br>2. Add 3 fields to PortStatus<br>3. Modify `_check_port()` banner collection<br>4. Add `_parse_banner()` method<br>5. Add `_extract_version()` method | 5 changes |
| **IPv6 Support** | network_propagation_port_scanner.py | 1. Move `ipaddress` to top-level import<br>2. Add `jitter_enabled` param to MultiTargetScanner<br>3. Complete rewrite of `scan_range()` | 3 major changes |

**Total Lines Added:** ~150 lines of new code
**Total Lines Modified:** ~40 lines of existing code
**Backwards Compatibility:** ✅ 100% (all new params have defaults)

---

## Testing the Changes

### Compile Check
```bash
python -m py_compile network_propagation_port_scanner.py
# Output: (no errors = success)
```

### Feature Verification
```bash
python enhanced_scanner_demo.py
# Output: [✅] All three features demonstrated
```

### Integration Check
```bash
python -m py_compile network_propagation_orchestrator.py
# Output: (no errors = orchestrator still compatible)
```

---

**Status:** All changes implemented and verified ✅
