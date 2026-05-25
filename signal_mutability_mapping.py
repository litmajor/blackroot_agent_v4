"""
signal_mutability_mapping.py - Signal classification and evasion causality mapping
===================================================================================

Core insight: Effectiveness measurement is broken because we score IMMUTABLE signals
(installed AV) the same as MUTABLE signals (debugger flags). 

This module:
1. Classifies signals as mutable vs immutable
2. Maps which evasion tactics affect which signals
3. Separates score calculation into mutable/immutable portions
4. Enables accurate effectiveness measurement
5. Empowers adaptive learning with causality

Signal Mutability Classification:
🔒 IMMUTABLE - Environmental signals that no evasion affects
   - Windows Defender installed
   - Kernel drivers present
   - Sandboxed environment
   - EDR subscription status
   
🔄 MUTABLE - Signals affected by evasion tactics
   - Debugger detection (PEB flags, breakpoints, timing)
   - API call patterns (behavioral detection)
   - Memory anomalies
   - Timing fingerprints
   - TEB/PEB inspection results
   - Exception handling patterns
   - Module enumeration
   - File system patterns
   - Network behavior

Evasion Effect Mapping:
   Describes which evasion tactic affects which mutable signals
   
Example:
   "anti_debug" → ["peb_flag_checked", "timing_analysis", "breakpoint_detection"]
   "memory_encryption" → ["memory_scan", "yara_detection"]
   "behavioral_noise" → ["behavioral_analysis", "api_pattern_detection"]
"""

import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Signal with mutability metadata"""
    name: str                           # e.g., "windows_defender", "peb_debugger_flag"
    category: str                       # "av", "edr", "debugger", "memory", "behavior"
    mutable: bool                       # Can evasion affect this signal?
    description: str = ""
    affected_by_evasions: List[str] = field(default_factory=list)  # List of evasion tactics
    
    def is_reducible(self) -> bool:
        """True if any evasion tactic can affect this signal"""
        return self.mutable and len(self.affected_by_evasions) > 0


class SignalMutabilityRegistry:
    """Registry of all threat signals with mutability classification"""
    
    # 🔒 IMMUTABLE SIGNALS (Environmental - no evasion affects these)
    IMMUTABLE_SIGNALS = {
        # AV/EDR Presence
        "windows_defender_installed": Signal(
            name="windows_defender_installed",
            category="av",
            mutable=False,
            description="Windows Defender subscription detected"
        ),
        "edr_products_present": Signal(
            name="edr_products_present",
            category="edr",
            mutable=False,
            description="EDR product installed (CrowdStrike, SentinelOne, etc)"
        ),
        "kernel_drivers_loaded": Signal(
            name="kernel_drivers_loaded",
            category="kernel",
            mutable=False,
            description="Kernel-mode drivers detected (minifilters, etc)"
        ),
        
        # Environment
        "sandboxed_environment": Signal(
            name="sandboxed_environment",
            category="environment",
            mutable=False,
            description="Running in sandbox (Cuckoo, Joe, etc)"
        ),
        "virtual_machine_detected": Signal(
            name="virtual_machine_detected",
            category="environment",
            mutable=False,
            description="Running in VM (VMware, VirtualBox, etc)"
        ),
        "honeypot_characteristics": Signal(
            name="honeypot_characteristics",
            category="environment",
            mutable=False,
            description="Honeypot-specific artifacts detected"
        ),
    }
    
    # 🔄 MUTABLE SIGNALS (Can be influenced by evasion)
    MUTABLE_SIGNALS = {
        # Debugger Detection
        "peb_debugger_flag": Signal(
            name="peb_debugger_flag",
            category="debugger",
            mutable=True,
            description="BeingDebugged flag in PEB",
            affected_by_evasions=["debugger_trace_hiding", "peb_manipulation"]
        ),
        "nt_global_flag": Signal(
            name="nt_global_flag",
            category="debugger",
            mutable=True,
            description="NtGlobalFlag set in PEB",
            affected_by_evasions=["debugger_trace_hiding"]
        ),
        "hardware_breakpoint": Signal(
            name="hardware_breakpoint",
            category="debugger",
            mutable=True,
            description="Hardware breakpoints detected",
            affected_by_evasions=["interrupt_evasion", "timing_masking"]
        ),
        "debugger_timing": Signal(
            name="debugger_timing",
            category="debugger",
            mutable=True,
            description="Execution timing anomalies suggest debugger",
            affected_by_evasions=["timing_masking", "behavioral_noise"]
        ),
        "exception_handling_patterns": Signal(
            name="exception_handling_patterns",
            category="debugger",
            mutable=True,
            description="SEH chain manipulation detected",
            affected_by_evasions=["debugger_trace_hiding", "seh_clearing"]
        ),
        
        # Memory Analysis
        "memory_scan_detection": Signal(
            name="memory_scan_detection",
            category="memory",
            mutable=True,
            description="Suspicious memory scanning detected",
            affected_by_evasions=["memory_encryption", "memory_hiding"]
        ),
        "yara_rule_match": Signal(
            name="yara_rule_match",
            category="memory",
            mutable=True,
            description="YARA rule matched in memory",
            affected_by_evasions=["memory_encryption"]
        ),
        "dll_shadow_detection": Signal(
            name="dll_shadow_detection",
            category="memory",
            mutable=True,
            description="Shadowed DLL detected",
            affected_by_evasions=["memory_hiding", "dll_encryption"]
        ),
        
        # API Behavior
        "suspicious_api_pattern": Signal(
            name="suspicious_api_pattern",
            category="behavior",
            mutable=True,
            description="API call sequence appears suspicious",
            affected_by_evasions=["iat_hooks", "behavioral_noise"]
        ),
        "api_interception_detected": Signal(
            name="api_interception_detected",
            category="behavior",
            mutable=True,
            description="API interception or function hooking detected",
            affected_by_evasions=["kernel_hiding", "iat_hooks"]
        ),
        "stolen_api_calls": Signal(
            name="stolen_api_calls",
            category="behavior",
            mutable=True,
            description="Direct syscalls detected (stolen APIs)",
            affected_by_evasions=["iat_hooks"]
        ),
        
        # Process Behavior
        "process_hollowing_detected": Signal(
            name="process_hollowing_detected",
            category="behavior",
            mutable=True,
            description="Process hollowing/code injection detected",
            affected_by_evasions=["ppid_spoofing", "process_hiding"]
        ),
        "parent_process_anomaly": Signal(
            name="parent_process_anomaly",
            category="behavior",
            mutable=True,
            description="Parent process doesn't make sense",
            affected_by_evasions=["ppid_spoofing", "kernel_hiding"]
        ),
        "process_name_mismatch": Signal(
            name="process_name_mismatch",
            category="behavior",
            mutable=True,
            description="Process name doesn't match actual behavior",
            affected_by_evasions=["kernel_hiding"]
        ),
        
        # Behavioral Footprint
        "execution_pattern_anomaly": Signal(
            name="execution_pattern_anomaly",
            category="behavior",
            mutable=True,
            description="Execution pattern doesn't match normal behavior",
            affected_by_evasions=["behavioral_noise"]
        ),
        "timing_inconsistency": Signal(
            name="timing_inconsistency",
            category="behavior",
            mutable=True,
            description="Timing patterns suggest automated execution",
            affected_by_evasions=["behavioral_noise", "timing_masking"]
        ),
        "no_user_interaction": Signal(
            name="no_user_interaction",
            category="behavior",
            mutable=True,
            description="No mouse/keyboard activity detected",
            affected_by_evasions=["user_activity_emulation"]
        ),
    }
    
    def __init__(self):
        self.all_signals: Dict[str, Signal] = {}
        self.all_signals.update(self.IMMUTABLE_SIGNALS)
        self.all_signals.update(self.MUTABLE_SIGNALS)
        logger.info(f"SignalMutabilityRegistry initialized: {len(self.IMMUTABLE_SIGNALS)} immutable, {len(self.MUTABLE_SIGNALS)} mutable")
    
    def get_signal(self, signal_name: str) -> Optional[Signal]:
        """Get signal by name"""
        return self.all_signals.get(signal_name)
    
    def get_mutable_signals(self) -> Dict[str, Signal]:
        """Get all mutable signals"""
        return {k: v for k, v in self.all_signals.items() if v.mutable}
    
    def get_immutable_signals(self) -> Dict[str, Signal]:
        """Get all immutable signals"""
        return {k: v for k, v in self.all_signals.items() if not v.mutable}


class EvasionEffectMapper:
    """Maps evasion tactics to the signals they affect"""
    
    EVASION_EFFECTS = {
        # Debugger Evasion
        "debugger_trace_hiding": {
            "description": "Hide debugger traces and anti-debugging artifacts",
            "affects_signals": [
                "peb_debugger_flag",
                "nt_global_flag",
                "exception_handling_patterns",
            ],
            "typical_reduction": 40,  # % threat score reduction if signal was present
        },
        
        # Memory Protection
        "memory_encryption": {
            "description": "Encrypt sensitive memory regions",
            "affects_signals": [
                "memory_scan_detection",
                "yara_rule_match",
                "dll_shadow_detection",
            ],
            "typical_reduction": 35,
        },
        
        # Process Hiding
        "kernel_hiding": {
            "description": "Hide process from kernel enumeration",
            "affects_signals": [
                "parent_process_anomaly",
                "process_name_mismatch",
                "process_hollowing_detected",
            ],
            "typical_reduction": 30,
        },
        
        # API Hooking
        "iat_hooks": {
            "description": "Install IAT hooks on critical Windows APIs",
            "affects_signals": [
                "suspicious_api_pattern",
                "api_interception_detected",
                "stolen_api_calls",
            ],
            "typical_reduction": 25,
        },
        
        # Behavioral Masking
        "behavioral_noise": {
            "description": "Add realistic user behavior patterns",
            "affects_signals": [
                "execution_pattern_anomaly",
                "timing_inconsistency",
                "suspicious_api_pattern",
            ],
            "typical_reduction": 20,
        },
        
        # User Activity Emulation
        "user_activity_emulation": {
            "description": "Emulate realistic user activity (mouse, keyboard, files)",
            "affects_signals": [
                "no_user_interaction",
                "timing_inconsistency",
                "execution_pattern_anomaly",
            ],
            "typical_reduction": 25,
        },
        
        # PPID Spoofing
        "ppid_spoofing": {
            "description": "Spoof parent process ID to appear legitimate",
            "affects_signals": [
                "parent_process_anomaly",
                "process_hollowing_detected",
            ],
            "typical_reduction": 20,
        },
        
        # Timing Masking
        "timing_masking": {
            "description": "Mask suspicious timing patterns",
            "affects_signals": [
                "debugger_timing",
                "timing_inconsistency",
            ],
            "typical_reduction": 30,
        },
        
        # Interrupt Evasion
        "interrupt_evasion": {
            "description": "Bypass hardware breakpoints and interrupt-based debugging",
            "affects_signals": [
                "hardware_breakpoint",
                "debugger_timing",
            ],
            "typical_reduction": 25,
        },
    }
    
    @classmethod
    def get_affected_signals(cls, evasion_tactic: str) -> List[str]:
        """Get list of signals affected by an evasion tactic"""
        if evasion_tactic in cls.EVASION_EFFECTS:
            return cls.EVASION_EFFECTS[evasion_tactic]["affects_signals"]
        return []
    
    @classmethod
    def get_typical_reduction(cls, evasion_tactic: str) -> int:
        """Get typical threat score reduction from evasion tactic"""
        if evasion_tactic in cls.EVASION_EFFECTS:
            return cls.EVASION_EFFECTS[evasion_tactic]["typical_reduction"]
        return 0
    
    @classmethod
    def get_evasion_effectiveness(cls, evasion_tactic: str, 
                                  detected_signals: Set[str]) -> int:
        """
        Calculate effectiveness of evasion tactic against detected signals.
        
        Only counts reduction for signals the tactic ACTUALLY affects.
        """
        if evasion_tactic not in cls.EVASION_EFFECTS:
            return 0
        
        affected_signals = set(cls.EVASION_EFFECTS[evasion_tactic]["affects_signals"])
        relevant_signals = affected_signals & detected_signals
        
        if not relevant_signals:
            logger.debug(f"Evasion '{evasion_tactic}' has no relevant signals to affect")
            return 0
        
        # Effectiveness = how much of detected signals does this tactic affect?
        effectiveness = int((len(relevant_signals) / len(detected_signals)) * 100)
        return min(100, effectiveness)


class ScoreSeparator:
    """
    Separates threat score into MUTABLE and IMMUTABLE portions.
    
    Only mutable score should be used for evasion effectiveness measurement.
    """
    
    def __init__(self, signal_registry: SignalMutabilityRegistry):
        self.registry = signal_registry
    
    def separate_signals(self, detected_signals: Dict[str, bool]) -> Dict[str, Any]:
        """
        Split detected signals into mutable and immutable categories.
        
        Returns:
            {
                'mutable': {signal_name: detected...},
                'immutable': {signal_name: detected...},
                'mutable_count': int,
                'immutable_count': int,
            }
        """
        mutable = {}
        immutable = {}
        
        for signal_name, detected in detected_signals.items():
            signal = self.registry.get_signal(signal_name)
            if signal:
                if signal.mutable:
                    mutable[signal_name] = detected
                else:
                    immutable[signal_name] = detected
        
        return {
            'mutable': mutable,
            'immutable': immutable,
            'mutable_count': sum(1 for v in mutable.values() if v),
            'immutable_count': sum(1 for v in immutable.values() if v),
        }
    
    def calculate_mutable_score(self, separated: Dict[str, Any]) -> int:
        """Calculate threat score from ONLY mutable signals"""
        mutable_count = separated['mutable_count']
        total_mutable = len(separated['mutable'])
        
        if total_mutable == 0:
            return 0
        
        # Score based on what % of mutable signals are present
        return int((mutable_count / total_mutable) * 100)
    
    def calculate_immutable_score(self, separated: Dict[str, Any]) -> int:
        """Calculate threat score from ONLY immutable signals"""
        immutable_count = separated['immutable_count']
        total_immutable = len(separated['immutable'])
        
        if total_immutable == 0:
            return 0
        
        # Score based on what % of immutable signals are present
        # These are environmental, so they're more critical
        return int((immutable_count / total_immutable) * 100)


# Global registry instances
_signal_registry = SignalMutabilityRegistry()
_evasion_mapper = EvasionEffectMapper()
_score_separator = ScoreSeparator(_signal_registry)


def get_signal_registry() -> SignalMutabilityRegistry:
    """Get global signal mutability registry"""
    return _signal_registry


def get_evasion_mapper() -> EvasionEffectMapper:
    """Get global evasion effect mapper"""
    return _evasion_mapper


def get_score_separator() -> ScoreSeparator:
    """Get global score separator"""
    return _score_separator
