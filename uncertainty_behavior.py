"""
UNCERTAINTY-DRIVEN BEHAVIOR: High uncertainty triggers exploration
===================================================================

Problem:
  - Uncertainty tracked but not used to drive behavior
  - System commits with 20% confidence then is surprised when it fails

Solution:
  - High uncertainty (> 0.7) triggers "explore" mode:
    * Lower risk tolerance
    * More diverse options
    * Gather more intel before committing
  - Cascades to strategy selection
  - Confidence + Uncertainty together drive decision-making

Author: Blackroot Agent v4
Date: April 6, 2026
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


class UncertaintyMode(Enum):
    """Behavioral mode based on uncertainty level"""
    CONFIDENT = "confident"      # Uncertainty < 0.3: Commit fully
    CAUTIOUS = "cautious"        # Uncertainty 0.3-0.7: Proceed carefully
    EXPLORING = "exploring"      # Uncertainty > 0.7: Gather intel first
    PANIC = "panic"              # Uncertainty > 0.85: Abort or defensive only


@dataclass
class UncertaintyBehavior:
    """How system should behave at different uncertainty levels"""
    mode: UncertaintyMode
    risk_multiplier: float       # Effective risk tolerance multiplier
    exploration_pressure: float  # How much to favor exploration
    intel_gathering_priority: float  # How important to gather info
    decision_reversibility: float  # How easily can decision be undone?
    
    def describe(self) -> str:
        """Human-readable description of behavior"""
        descriptions = {
            UncertaintyMode.CONFIDENT: (
                "High confidence in environment. Commit fully to selected approach. "
                "Execute with aggressive timeline."
            ),
            UncertaintyMode.CAUTIOUS: (
                "Moderate uncertainty. Proceed with selected approach but stay flexible. "
                "Have backup plans ready. Monitor for changes."
            ),
            UncertaintyMode.EXPLORING: (
                "High uncertainty in environment. Prioritize information gathering over commitment. "
                "Execute reconnaissance, avoid irreversible actions. Build situational awareness."
            ),
            UncertaintyMode.PANIC: (
                "Extreme uncertainty. Assume adversarial environment. "
                "Minimize exposed operations. Focus only on survival and data protection."
            ),
        }
        return descriptions.get(self.mode, "Unknown mode")


class UncertaintyBehaviorEngine:
    """
    Translate uncertainty into behavior change.
    
    High uncertainty should cause:
    - More exploration
    - Less aggressive decisions
    - Lower risk tolerance
    - Preference for reversible actions
    """
    
    def get_behavior(self, uncertainty: float, confidence: float = None) -> UncertaintyBehavior:
        """
        Determine behavior based on uncertainty level.
        
        Args:
            uncertainty: 0-1 (0=certain, 1=completely unknown)
            confidence: 0-1 (optional, for nuance)
        
        Returns:
            UncertaintyBehavior with multipliers and priorities
        """
        
        if uncertainty < 0.3:
            # High knowledge, low uncertainty
            return UncertaintyBehavior(
                mode=UncertaintyMode.CONFIDENT,
                risk_multiplier=1.0,  # Full risk tolerance
                exploration_pressure=0.3,  # Mild exploration bonus
                intel_gathering_priority=0.2,  # Low priority, we know enough
                decision_reversibility=0.5,  # Can pivot if needed
            )
        
        elif uncertainty < 0.7:
            # Moderate uncertainty
            return UncertaintyBehavior(
                mode=UncertaintyMode.CAUTIOUS,
                risk_multiplier=0.7,  # 30% reduction in risk tolerance
                exploration_pressure=0.6,  # Medium exploration bonus
                intel_gathering_priority=0.6,  # Important to gather more
                decision_reversibility=0.7,  # Keep options open
            )
        
        elif uncertainty < 0.85:
            # High uncertainty
            return UncertaintyBehavior(
                mode=UncertaintyMode.EXPLORING,
                risk_multiplier=0.4,  # 60% reduction in risk tolerance
                exploration_pressure=1.0,  # Maximum exploration bonus
                intel_gathering_priority=0.95,  # Critical to gather intel
                decision_reversibility=0.9,  # Only reversible actions
            )
        
        else:
            # Extreme uncertainty
            return UncertaintyBehavior(
                mode=UncertaintyMode.PANIC,
                risk_multiplier=0.1,  # 90% reduction in risk tolerance
                exploration_pressure=0.5,  # Balanced exploration
                intel_gathering_priority=0.7,  # Gather intel, but defensively
                decision_reversibility=0.95,  # Only fully reversible actions
            )
    
    def get_exploration_strategy(self, uncertainty: float) -> str:
        """
        Recommend strategy shift based on uncertainty.
        
        Returns:
            Strategy name (should be fed to strategy_engine)
        """
        behavior = self.get_behavior(uncertainty)
        
        if behavior.mode == UncertaintyMode.CONFIDENT:
            return "DYNAMIC"  # Balanced approach
        elif behavior.mode == UncertaintyMode.CAUTIOUS:
            return "LATERAL_THEN_EXFIL"  # Careful lateral movement
        elif behavior.mode == UncertaintyMode.EXPLORING:
            return "STEALTH_LONG_GAME"  # Information gathering mode
        else:
            return "SURVIVAL"  # Pure defense
    
    def filter_options_by_uncertainty(
        self,
        options: list,
        uncertainty: float,
    ) -> Tuple[list, str]:
        """
        Filter available options based on uncertainty.
        
        High uncertainty excludes irreversible high-risk decisions.
        
        Args:
            options: Available decision options
            uncertainty: Current uncertainty level
        
        Returns:
            (filtered_options, reason)
        """
        behavior = self.get_behavior(uncertainty)
        
        if behavior.mode == UncertaintyMode.CONFIDENT:
            return options, "All options available (high certainty)"
        
        elif behavior.mode == UncertaintyMode.CAUTIOUS:
            # Filter out only the riskiest options
            risky_keywords = ["kernel", "eol", "advanced"]
            filtered = [opt for opt in options if not any(k in opt.lower() for k in risky_keywords)]
            return filtered or options, "Excluded highest-risk techniques"
        
        elif behavior.mode == UncertaintyMode.EXPLORING:
            # Only reconnaissance and safe lateral movement
            safe_keywords = ["recon", "enum", "survey", "gather", "scan"]
            filtered = [opt for opt in options if any(k in opt.lower() for k in safe_keywords)]
            return filtered or ["osint", "port_scan"], "Restricted to reconnaissance only"
        
        else:
            # Panic: only defensive
            defensive = ["network_isolation", "alert_monitoring", "backup_data"]
            filtered = [opt for opt in options if any(d in opt.lower() for d in defensive)]
            return filtered or ["c2_disable"], "Restricted to defensive operations only"
    
    def should_gather_intel_instead(self, uncertainty: float, planned_action: str) -> bool:
        """
        Should we gather more intel instead of executing planned action?
        
        When uncertainty is very high and action is risky, better to gather
        information first.
        """
        behavior = self.get_behavior(uncertainty)
        
        # If in exploring/panic mode, prioritize intel
        if behavior.mode in [UncertaintyMode.EXPLORING, UncertaintyMode.PANIC]:
            risky_actions = ["kernel", "exploit", "lateral", "exfil", "escalation"]
            is_risky = any(r in planned_action.lower() for r in risky_actions)
            
            if is_risky:
                return True  # Gather intel instead
        
        return False


# =============================================================================
# INTEGRATION EXAMPLE: How uncertainty shapes decision-making
# =============================================================================

if __name__ == "__main__":
    engine = UncertaintyBehaviorEngine()
    
    print("=" * 70)
    print("UNCERTAINTY-DRIVEN BEHAVIOR: From Commits to Exploration")
    print("=" * 70)
    
    scenarios = [
        ("Well-known environment", 0.15),
        ("Moderate uncertainty", 0.50),
        ("Highly uncertain", 0.75),
        ("Extreme uncertainty", 0.90),
    ]
    
    for name, uncertainty in scenarios:
        print(f"\n{name} (uncertainty={uncertainty:.0%})")
        print("-" * 70)
        
        behavior = engine.get_behavior(uncertainty)
        
        print(f"Mode: {behavior.mode.value.upper()}")
        print(f"Risk multiplier: {behavior.risk_multiplier:.0%}")
        print(f"Exploration pressure: {behavior.exploration_pressure:.0%}")
        print(f"Strategy: {engine.get_exploration_strategy(uncertainty)}")
        
        options = ["phishing", "exploit_unpatched", "osint", "lateral", "exfil"]
        filtered, reason = engine.filter_options_by_uncertainty(options, uncertainty)
        print(f"Available options: {filtered if filtered else 'None (too risky)'}")
        print(f"  Reason: {reason}")
        
        # Check if should gather intel
        should_gather = engine.should_gather_intel_instead(uncertainty, "kernel_exploit")
        if should_gather:
            print(f"⚠️  Should gather intel instead of kernel_exploit")
        
        print(f"\nBehavior Guide:")
        print(f"{behavior.describe()}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Uncertainty is not tracked—it's USED")
    print("=" * 70)
    print("High uncertainty causes:")
    print("  ✓ Strategy shift (to STEALTH_LONG_GAME or SURVIVAL)")
    print("  ✓ Option filtering (exclude risky techniques)")
    print("  ✓ Intel gathering priority (prioritize recon over action)")
    print("  ✓ Risk tolerance reduction (0.4x to 0.1x)")
