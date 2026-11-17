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
import os
import sys
import uuid
import subprocess
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# psutil is required for process scanning; degrade gracefully if missing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

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


class EvidenceLevel(Enum):
    """Provenance and depth of defensive telemetry evidence."""
    INSTALLED = 1       # Level 1: Static presence (Registry key, binary on disk)
    RUNNING = 2         # Level 2: Active execution (Process running, driver mapped in memory)
    ENFORCING = 3       # Level 3: Active enforcement (RTP active, behavioral monitoring, tamper protection)


@dataclass
class EvidenceItem:
    """Detailed evidence record with provenance and confidence."""
    source_type: str         # "process", "driver", "registry", "enforcement"
    name: str                # e.g., "msmpeng.exe", "WdFilter", "WinDefend"
    product: str             # e.g., "Windows Defender"
    level: EvidenceLevel     # INSTALLED, RUNNING, ENFORCING
    confidence: float        # 0.0 to 1.0 (measurement provenance strength)
    score: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


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
    """Result of an EDR telemetry and threat scan."""
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    detected_processes: List[str] = field(default_factory=list)
    detected_drivers: List[str] = field(default_factory=list)
    detected_registry: List[str] = field(default_factory=list)
    threat_level: ThreatLevel = ThreatLevel.CLEAR
    total_score: int = 0
    detected_at: datetime = field(default_factory=datetime.now)
    # Structured telemetry and evidence provenance
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    highest_level: EvidenceLevel = EvidenceLevel.INSTALLED
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


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
    # NOTE on registry_paths:
    #   The registry scanner does EXACT subkey matching under Services\
    #   (last component of the path, case-insensitive).
    #   Only add paths whose final component is unique to the product —
    #   do NOT use short prefixes like "CB" that also match Windows built-ins
    #   (cbdhsvc = Connected Devices Platform Host, NcbService = Network
    #   Connection Broker, etc.).
    THREAT_SIGNATURES = {
        "CrowdStrike_Falcon": ThreatSignature(
            name="CrowdStrike Falcon",
            process_names=["CSFalconService.exe", "CSFalconServiceAPI.exe"],
            kernel_drivers=["CsFalcon.sys", "CsFalconConfig.sys", "csagent.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\CSFalcon",
                            r"HKLM\SYSTEM\CurrentControlSet\Services\CSAgent"],
            score=9
        ),
        "Windows_Defender": ThreatSignature(
            name="Windows Defender",
            # MsMpEng + NisSrv are real enforcement sensors.
            # SecurityHealthService is a UI/tray wrapper — no enforcement
            # capability, excluded from process scoring.
            process_names=["MsMpEng.exe", "NisSrv.exe"],
            kernel_drivers=["WdFilter.sys", "WdNisDrv.sys", "WdBoot.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\WinDefend",
                            r"HKLM\SYSTEM\CurrentControlSet\Services\WdFilter"],
            score=7
        ),
        "Carbon_Black": ThreatSignature(
            name="Carbon Black",
            # cb.exe / cbsensor.exe are the real CB processes.
            # cbdhsvc = Windows Connected Devices Platform Host (built-in) — NOT CB.
            # NcbService = Windows Network Connection Broker (built-in) — NOT CB.
            process_names=["cb.exe", "cbcompute.exe", "cbsensor.exe", "cbdefense.exe"],
            kernel_drivers=["cbk7.sys", "CbDefense.sys"],
            # Exact service names — NOT "CB" substring which catches Windows builtins
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\CbDefense",
                            r"HKLM\SYSTEM\CurrentControlSet\Services\cbk7",
                            r"HKLM\SOFTWARE\Carbon Black\Sensor"],
            score=8
        ),
        "Sentinelone": ThreatSignature(
            name="SentinelOne",
            process_names=["SentinelAgent.exe", "LogProcessorService.exe", "SentinelServiceHost.exe"],
            kernel_drivers=["sentinelmonitor.sys", "SentinelOne.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\SentinelAgent",
                            r"HKLM\SOFTWARE\Sentinel Labs"],
            score=8
        ),
        "Trend_Micro": ThreatSignature(
            name="Trend Micro",
            process_names=["TMCCSF.exe", "KeenAsync.exe", "tmwfp.exe", "coreServiceShell.exe"],
            kernel_drivers=["tmwfp.sys", "vsapint.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\TmFilter",
                            r"HKLM\SYSTEM\CurrentControlSet\Services\tmwfp"],
            score=7
        ),
        "McAfee_ENS": ThreatSignature(
            name="McAfee ENS",
            process_names=["mfemms.exe", "mcmgrsvc.exe", "mfetp.exe"],
            kernel_drivers=["mfewfp.sys", "mfehidk.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\McFilter",
                            r"HKLM\SYSTEM\CurrentControlSet\Services\mfehidk"],
            score=7
        ),
        "VMware_Carbon_Black_Cloud": ThreatSignature(
            name="VMware CB Cloud",
            process_names=["mdr-service.exe", "reprcmd.exe", "repmgr.exe"],
            kernel_drivers=["carbon_black_cloud.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\VMwareCBService",
                            r"HKLM\SOFTWARE\VMware, Inc.\VMware Carbon Black Cloud"],
            score=8
        ),
        "Cylance": ThreatSignature(
            name="Cylance",
            process_names=["cyserver.exe", "cylmqctx.exe", "CylanceSvc.exe"],
            kernel_drivers=["cylmqio.sys", "CylanceProtectDrv.sys"],
            registry_paths=[r"HKLM\SYSTEM\CurrentControlSet\Services\Cylance",
                            r"HKLM\SOFTWARE\Cylance\Desktop"],
            score=7
        ),
        "Process_Monitor": ThreatSignature(
            name="Process Monitor",
            process_names=["procmon.exe", "procmon64.exe"],
            kernel_drivers=["PROCMON24"],
            registry_paths=[],
            score=4  # Forensics tool, lower risk
        ),
        "WireShark": ThreatSignature(
            name="WireShark",
            process_names=["wireshark.exe"],
            kernel_drivers=["npcap", "NPF"],
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
                import ctypes as _ct
                return bool(_ct.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        else:
            return os.geteuid() == 0 if hasattr(os, 'geteuid') else False
    
    def scan_for_threats(self) -> DetectionResult:
        """
        Comprehensive threat scan with result caching and evidence provenance.
        
        Returns:
            DetectionResult with all detected threats, confidence, and provenance.
        """
        # Check cache first
        if self._scan_cache is not None and self._cache_time is not None:
            cache_age = (datetime.now() - self._cache_time).total_seconds()
            if cache_age < self._cache_ttl:
                log.debug(f"Using cached threat scan (age={cache_age:.1f}s)")
                return self._scan_cache
        
        result = DetectionResult()
        scan_start = time.time()
        
        score_breakdown: Dict[str, Any] = {
            "processes": {},
            "drivers": {},
            "registry": {},
            "enforcement": {},
            "total": 0
        }
        evidence_items: List[EvidenceItem] = []

        # 1. Scan processes (Level 2: Running)
        result.detected_processes = self._scan_processes()
        for proc_entry in result.detected_processes:
            sig_name, proc_name = proc_entry.split(":", 1)
            sig_score = next((s.score for s in self.THREAT_SIGNATURES.values() if s.name == sig_name), 5)
            score_breakdown["processes"][proc_name] = sig_score
            evidence_items.append(EvidenceItem(
                source_type="process",
                name=proc_name,
                product=sig_name,
                level=EvidenceLevel.RUNNING,
                confidence=0.95,
                score=sig_score,
                details={"process_name": proc_name}
            ))

        # 2. Scan kernel drivers (Level 2: Running, verified via SCM)
        result.detected_drivers = self._scan_kernel_drivers()
        for drv_entry in result.detected_drivers:
            sig_name, drv_name = drv_entry.split(":", 1)
            sig_score = next((s.score for s in self.THREAT_SIGNATURES.values() if s.name == sig_name), 5)
            score_breakdown["drivers"][drv_name] = sig_score
            evidence_items.append(EvidenceItem(
                source_type="driver",
                name=drv_name,
                product=sig_name,
                level=EvidenceLevel.RUNNING,
                confidence=1.0,
                score=sig_score,
                details={"driver_service": drv_name}
            ))

        # 3. Scan registry (Level 1: Installed)
        if sys.platform == 'win32':
            result.detected_registry = self._scan_registry()
            for reg_entry in result.detected_registry:
                sig_name, reg_path = reg_entry.split(":", 1)
                # Level 1 installation presence contributes baseline 2 points per key
                reg_score = 2
                reg_key_name = reg_path.split("\\")[-1]
                score_breakdown["registry"][reg_key_name] = reg_score
                evidence_items.append(EvidenceItem(
                    source_type="registry",
                    name=reg_key_name,
                    product=sig_name,
                    level=EvidenceLevel.INSTALLED,
                    confidence=0.60,
                    score=reg_score,
                    details={"path": reg_path}
                ))

        # 4. Level 3: Active enforcement probe (RTP/Behavior/Tamper status)
        enforce_score, enforce_details = self._probe_defender_enforcement()
        if enforce_score > 0:
            score_breakdown["enforcement"] = enforce_details
            score_breakdown["enforcement"]["score"] = enforce_score
            evidence_items.append(EvidenceItem(
                source_type="enforcement",
                name="Windows Defender Active Enforcement",
                product="Windows Defender",
                level=EvidenceLevel.ENFORCING,
                confidence=1.0,
                score=enforce_score,
                details=enforce_details
            ))

        # Calculate totals
        total = (
            sum(score_breakdown["processes"].values()) +
            sum(score_breakdown["drivers"].values()) +
            sum(score_breakdown["registry"].values()) +
            enforce_score
        )
        score_breakdown["total"] = total
        result.total_score = total
        result.score_breakdown = score_breakdown
        result.evidence_items = evidence_items

        # Determine highest evidence level & mean confidence
        if any(e.level == EvidenceLevel.ENFORCING for e in evidence_items):
            result.highest_level = EvidenceLevel.ENFORCING
        elif any(e.level == EvidenceLevel.RUNNING for e in evidence_items):
            result.highest_level = EvidenceLevel.RUNNING
        elif any(e.level == EvidenceLevel.INSTALLED for e in evidence_items):
            result.highest_level = EvidenceLevel.INSTALLED
        else:
            result.highest_level = EvidenceLevel.INSTALLED

        if evidence_items:
            result.confidence = sum(e.confidence for e in evidence_items) / len(evidence_items)
        else:
            result.confidence = 0.0

        # Determine threat level
        result.threat_level = self._calculate_threat_level(result)
        
        # Log result
        scan_time = time.time() - scan_start
        log.debug(f"Threat scan complete: {result.threat_level.name} (score={result.total_score}, confidence={result.confidence:.2f}, time={scan_time:.2f}s)")
        
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

        if not HAS_PSUTIL:
            log.debug("psutil not available — process scan skipped")
            return detected

        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    
                    # Check against all signatures
                    for sig_key, sig in self.THREAT_SIGNATURES.items():
                        for target_proc in sig.process_names:
                            if target_proc.lower() == proc_name or target_proc.lower() in proc_name:
                                detected.append(f"{sig.name}:{proc_name}")
                                log.warning(f"[+] Detected {sig.name}: {proc_name} (PID: {proc.info['pid']})")
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            log.error(f"Process scan failed: {e}")
        
        return detected
    
    def _scan_kernel_drivers(self) -> List[str]:
        """
        Scan for actively running kernel drivers via Service Control Manager.
        Works in unprivileged user mode and inspects actual running state (STATE: 4 RUNNING).
        """
        detected = []
        if sys.platform != 'win32':
            return detected

        try:
            for sig_key, sig in self.THREAT_SIGNATURES.items():
                for target_driver in sig.kernel_drivers:
                    # Strip .sys extension to get service name
                    svc_name = target_driver.replace(".sys", "")
                    try:
                        res = subprocess.run(
                            ["sc.exe", "query", svc_name],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if "STATE" in res.stdout and "RUNNING" in res.stdout:
                            entry = f"{sig.name}:{target_driver}"
                            if entry not in detected:
                                detected.append(entry)
                                log.warning(f"[+] Detected running kernel driver: {target_driver} ({sig.name})")
                    except Exception as e:
                        log.debug(f"SCM query failed for {svc_name}: {e}")
        except Exception as e:
            log.debug(f"Kernel driver scan failed: {e}")
        
        return detected
    
    def _scan_registry(self) -> List[str]:
        """
        Scan registry for security tool artifacts.

        Uses EXACT final-component matching against known service/software names.
        This prevents false positives from Windows built-ins like:
          - cbdhsvc  (Connected Devices Platform Host)
          - NcbService (Network Connection Broker)
          - {GUID} keys that happen to contain 'CB'
        """
        detected = []

        if not HAS_WINREG or sys.platform != 'win32':
            return detected

        try:
            hive_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]

            # Build exact-match sets from signature registry_paths
            # Key: final component (lowercased), Value: (sig_name, sig.name)
            exact_service_names: dict = {}
            for sig_key, sig in self.THREAT_SIGNATURES.items():
                for reg_path in sig.registry_paths:
                    final = reg_path.split("\\")[-1].lower()
                    if final in ("hklm", "hkcu", "hkcr", "hku", "hkcc", ""):
                        continue
                    if final.startswith("{") and final.endswith("}"):
                        continue
                    exact_service_names[final] = (sig_key, sig.name)

            for hive, base_path in hive_paths:
                try:
                    key = winreg.OpenKey(hive, base_path)
                    num_subkeys = winreg.QueryInfoKey(key)[0]
                    for i in range(num_subkeys):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey_lower = subkey_name.lower()
                            if subkey_lower in exact_service_names:
                                sig_key, sig_display = exact_service_names[subkey_lower]
                                entry = f"{sig_display}:{base_path}\\{subkey_name}"
                                if entry not in detected:
                                    detected.append(entry)
                                    log.warning(f"[+] Detected registry (exact): {subkey_name} ({sig_display})")
                        except WindowsError:
                            pass
                    winreg.CloseKey(key)
                except Exception as e:
                    log.debug(f"Registry scan error for {base_path}: {e}")
        except Exception as e:
            log.debug(f"Registry scan failed: {e}")

        return detected

    def _probe_defender_enforcement(self) -> Tuple[int, Dict[str, Any]]:
        """
        Level 3 enforcement probe: query Windows Defender Real-Time Protection
        status via Get-MpComputerStatus.

        Returns:
            (score, details_dict)
            Score is additive:
              +5 for active RTP
              +2 for active BehaviorMonitor
              +2 for active TamperProtection
        """
        if sys.platform != 'win32':
            return 0, {}

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-MpComputerStatus | Select-Object RealTimeProtectionEnabled,"
                 "BehaviorMonitorEnabled,IsTamperProtected,AntivirusEnabled | ConvertTo-Json"],
                capture_output=True, text=True, timeout=8
            )
            if result.returncode != 0 or not result.stdout.strip():
                return 0, {}

            import json
            status = json.loads(result.stdout.strip())
            rtp = bool(status.get("RealTimeProtectionEnabled", False))
            behav = bool(status.get("BehaviorMonitorEnabled", False))
            tamper = bool(status.get("IsTamperProtected", False))
            av_enabled = bool(status.get("AntivirusEnabled", False))

            score = 0
            if rtp:
                score += 5
            if behav:
                score += 2
            if tamper:
                score += 2

            details = {
                "RealTimeProtectionEnabled": rtp,
                "BehaviorMonitorEnabled": behav,
                "IsTamperProtected": tamper,
                "AntivirusEnabled": av_enabled,
                "enforcement_score": score
            }

            if score > 0:
                log.warning(f"[+] Defender Level 3 enforcement active: RTP={rtp} Behavior={behav} TamperProtected={tamper} (score +{score})")

            return score, details

        except Exception as e:
            log.debug(f"Defender enforcement probe failed: {e}")

        return 0, {}
    
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
    Orchestrates autonomous EDR detection and telemetry reporting.
    
    Architecture (Refactored to Pure Telemetry):
    1. Scan for EDR/AV threats (periodic, cached, state-aware)
    2. Assess evidence levels: Installed vs Running vs Actively Enforcing
    3. Measure detection confidence and provenance
    4. Provide telemetry streams to anomaly classifier / health monitors
    5. Pure passive observer — NO automated behavioral noise or evasion actuation.
    """
    
    def __init__(self, scan_interval: int = 30, auto_start: bool = False, detector: Optional[EDRDetector] = None):
        """
        Initialize orchestrator with optional shared detector instance.
        
        Args:
            scan_interval: Seconds between threat scans (default 30)
            auto_start: Start monitoring loop immediately
            detector: Existing EDRDetector instance (shares cache and state)
        """
        self.scan_interval = scan_interval
        self.detector = detector if detector is not None else EDRDetector()
        
        # Anomaly classifier telemetry feed
        self.anomaly_classifier: Optional[AnomalyClassifier] = None
        if HAS_ANOMALY_CLASSIFIER:
            self.anomaly_classifier = AnomalyClassifier()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Telemetry history tracking
        self.threat_history: Dict[datetime, ThreatLevel] = {}
        self.current_threat_level = ThreatLevel.CLEAR
        self.current_noise_intensity = 0  # Retained as analytical threat index
        self.last_scan_id: Optional[str] = None
        self.last_scan_time: Optional[datetime] = None
        
        # Synchronize with detector's existing scan result if already populated
        if self.detector.last_result is not None:
            self._ingest_scan_result(self.detector.last_result)
        
        if auto_start:
            self.start_monitoring()
    
    def scan_now(self) -> DetectionResult:
        """
        Execute a synchronous threat scan and immediately ingest into telemetry state.
        
        Returns:
            DetectionResult populated with complete evidence and threat metrics.
        """
        detections = self.detector.scan_for_threats()
        self._ingest_scan_result(detections)
        return detections
    
    def _ingest_scan_result(self, detections: DetectionResult):
        """Consume and synchronize scan results into orchestrator state."""
        last_level = self.current_threat_level
        self.current_threat_level = detections.threat_level
        self.last_scan_id = detections.scan_id
        self.last_scan_time = detections.detected_at
        self.threat_history[detections.detected_at] = detections.threat_level
        
        new_intensity = self._calculate_evasion_intensity(detections.threat_level)
        if new_intensity != self.current_noise_intensity:
            self._update_evasion_intensity(new_intensity)
            self.current_noise_intensity = new_intensity
        
        # Feed telemetry metrics to anomaly classifier
        if self.anomaly_classifier:
            self.anomaly_classifier.observe("edr_threat_score", float(detections.total_score))
            self.anomaly_classifier.observe("edr_confidence", float(detections.confidence))
        
        if detections.threat_level != last_level:
            log.warning(
                f"[!] EDR threat state transition: {last_level.name} -> {detections.threat_level.name} "
                f"(ScanID: {detections.scan_id}, Score: {detections.total_score}, Confidence: {detections.confidence:.2f}, Highest Level: {detections.highest_level.name})"
            )
    
    def start_monitoring(self):
        """Start autonomous threat telemetry monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        log.info("[+] Adaptive EDR Telemetry Orchestrator started (passive telemetry mode)")
    
    def stop_monitoring(self):
        """Stop autonomous threat telemetry monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        log.info("[-] Adaptive EDR Telemetry Orchestrator stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop - telemetry and state tracking."""
        consecutive_clear_scans = 0
        adaptive_interval = self.scan_interval
        
        while not self._stop_event.is_set():
            try:
                detections = self.scan_now()
                
                if detections.threat_level == ThreatLevel.CLEAR:
                    consecutive_clear_scans += 1
                    if consecutive_clear_scans >= 5:
                        adaptive_interval = min(self.scan_interval * 3, 300)
                    log.debug(f"[OK] Threat clear (scan {consecutive_clear_scans}) - backoff interval: {adaptive_interval}s")
                else:
                    consecutive_clear_scans = 0
                    adaptive_interval = self.scan_interval
                    log.debug(f"[*] Threat level stable: {detections.threat_level.name}")
                
            except Exception as e:
                log.error(f"Monitor loop error: {e}")
            
            self._stop_event.wait(adaptive_interval)
    
    def _calculate_evasion_intensity(self, threat_level: ThreatLevel) -> int:
        """Map threat level to analytical threat index (0-10%)."""
        intensity_map = {
            ThreatLevel.CLEAR: 0,
            ThreatLevel.LOW: 2,
            ThreatLevel.MEDIUM: 4,
            ThreatLevel.HIGH: 7,
            ThreatLevel.CRITICAL: 10,
        }
        return intensity_map.get(threat_level, 0)
    
    def _update_evasion_intensity(self, intensity: int):
        """Record threat index state change (passive telemetry only)."""
        log.info(f"[~] Telemetry threat index updated: {intensity}% (telemetry mode)")
    
    def get_threat_summary(self) -> Dict:
        """Get comprehensive threat assessment and telemetry summary."""
        res = self.detector.last_result
        return {
            "orchestrator_threat_level": self.current_threat_level.name,
            "current_threat_level": self.current_threat_level.name,
            "current_threat_score": res.total_score if res else 0,
            "confidence": round(res.confidence, 2) if res else 0.0,
            "highest_evidence_level": res.highest_level.name if res else "NONE",
            "evasion_intensity": self.current_noise_intensity,
            "scan_id": res.scan_id if res else self.last_scan_id,
            "detector_id": hex(id(self.detector)),
            "last_scan": (res.detected_at.isoformat() if (res and res.detected_at) else (self.last_scan_time.isoformat() if self.last_scan_time else None)),
            "detections": {
                "processes": len(res.detected_processes) if res else 0,
                "drivers": len(res.detected_drivers) if res else 0,
                "registry": len(res.detected_registry) if res else 0,
                "enforcement": (1 if (res and any(e.level == EvidenceLevel.ENFORCING for e in res.evidence_items)) else 0),
            },
            "score_breakdown": res.score_breakdown if res else {},
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
