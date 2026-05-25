"""
Strategy Engine: Global Operational Strategy
=============================================

The missing layer between context and action.

Instead of:
  Context → Action (stateless, no coherence)

We do:
  Context → Strategy → Actions (coherent across phases)

Strategy examples:
- "aggressive_speed": Fast exfil, high risk acceptable, max resources
- "stealth_long_game": Slow infiltration, minimize detection, preserve optionality  
- "survival": Detect imminent, minimize everything, prepare extraction
- "persistence": Build infrastructure for long-term presence
- "lateral_then_exfil": Get throughout network first, then extract

Every decision is constrained by the strategy.

Author: Blackroot Agent v4
Date: April 6, 2026
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
from world_state import WorldState, OperationalPhase

logger = logging.getLogger("StrategyEngine")


class Strategy(Enum):
    """Global operational strategies"""
    AGGRESSIVE_SPEED = "aggressive_speed"  # Get in, get data, get out fast
    STEALTH_LONG_GAME = "stealth_long_game"  # Slow, careful, minimize detection
    LATERAL_THEN_EXFIL = "lateral_then_exfil"  # Build reach first, then exfil
    PERSISTENCE_BUILD = "persistence_build"  # Underground, long-term presence
    SURVIVAL = "survival"  # Minimize risk, prepare extraction
    DYNAMIC = "dynamic"  # Adjust per phase based on conditions


@dataclass
class StrategyProfile:
    """Strategy parameters"""
    name: Strategy
    risk_tolerance: float  # 0-1, how much risk acceptable
    time_aggression: float  # 0-1, how fast to move
    privilege_priority: float  # 0-1, how much to escalate
    lateral_depth: float  # 0-1, how far to spread
    evasion_strictness: float  # 0-1, how carefully to hide
    resource_optimization: float  # 0-1, how carefully to use resources
    
    def __post_init__(self):
        """Validate parameters sum correctly"""
        assert 0 <= self.risk_tolerance <= 1
        assert 0 <= self.time_aggression <= 1
        assert 0 <= self.privilege_priority <= 1
        assert 0 <= self.lateral_depth <= 1
        assert 0 <= self.evasion_strictness <= 1
        assert 0 <= self.resource_optimization <= 1


class StrategyEngine:
    """
    Selects global strategy based on world state.
    
    All downstream decisions are constrained by strategy.
    
    Usage:
        engine = StrategyEngine()
        strategy = engine.choose(world_state)
        
        # Now all decisions use this strategy
        decision = causal_engine.choose_infiltration(
            ...,
            strategy=strategy
        )
    """
    
    STRATEGY_PROFILES = {
        Strategy.AGGRESSIVE_SPEED: StrategyProfile(
            name=Strategy.AGGRESSIVE_SPEED,
            risk_tolerance=0.8,
            time_aggression=0.9,
            privilege_priority=0.7,
            lateral_depth=0.4,
            evasion_strictness=0.3,
            resource_optimization=0.2,
        ),
        Strategy.STEALTH_LONG_GAME: StrategyProfile(
            name=Strategy.STEALTH_LONG_GAME,
            risk_tolerance=0.2,
            time_aggression=0.2,
            privilege_priority=0.4,
            lateral_depth=0.7,
            evasion_strictness=0.9,
            resource_optimization=0.8,
        ),
        Strategy.LATERAL_THEN_EXFIL: StrategyProfile(
            name=Strategy.LATERAL_THEN_EXFIL,
            risk_tolerance=0.5,
            time_aggression=0.7,
            privilege_priority=0.8,
            lateral_depth=0.95,
            evasion_strictness=0.6,
            resource_optimization=0.5,
        ),
        Strategy.PERSISTENCE_BUILD: StrategyProfile(
            name=Strategy.PERSISTENCE_BUILD,
            risk_tolerance=0.3,
            time_aggression=0.1,
            privilege_priority=0.9,
            lateral_depth=0.8,
            evasion_strictness=0.85,
            resource_optimization=0.9,
        ),
        Strategy.SURVIVAL: StrategyProfile(
            name=Strategy.SURVIVAL,
            risk_tolerance=0.1,
            time_aggression=0.05,
            privilege_priority=0.2,
            lateral_depth=0.1,
            evasion_strictness=1.0,
            resource_optimization=1.0,
        ),
    }
    
    def __init__(self):
        """Initialize strategy engine"""
        logger.info("[StrategyEngine] Initialized")
    
    def choose(self, world: WorldState) -> Strategy:
        """
        Decide strategy based on world state.
        
        This is where global coherence is enforced.
        
        Args:
            world: Current world state
            
        Returns:
            Strategy to pursue
        """
        
        # Rule 1: If critical risk, switch to survival
        if world.risk.total_exposure() > 0.75:
            logger.warning("[StrategyEngine] CRITICAL RISK - switching to SURVIVAL")
            return Strategy.SURVIVAL
        
        # Rule 2: If time running out, become aggressive
        if world.resources.time_remaining_hours < 6:
            logger.info("[StrategyEngine] Time window closing - switching to AGGRESSIVE_SPEED")
            return Strategy.AGGRESSIVE_SPEED
        
        # Rule 3: If environment hardened + high uncertainty, stealth
        if world.environment.is_hardened() and world.environment.uncertainty > 0.4:
            logger.info("[StrategyEngine] Hardened environment - choosing STEALTH_LONG_GAME")
            return Strategy.STEALTH_LONG_GAME
        
        # Rule 4: Dynamic strategy - be aggressive early, careful later
        if world.current_phase in [OperationalPhase.INITIAL_RECONNAISSANCE, OperationalPhase.INFILTRATION]:
            if world.risk.detection_probability < 0.3:
                logger.info("[StrategyEngine] Early phase, low risk - AGGRESSIVE_SPEED")
                return Strategy.AGGRESSIVE_SPEED
        
        if world.current_phase in [OperationalPhase.LATERAL_MOVEMENT]:
            logger.info("[StrategyEngine] Lateral phase - LATERAL_THEN_EXFIL")
            return Strategy.LATERAL_THEN_EXFIL
        
        if world.current_phase in [OperationalPhase.DATA_EXFILTRATION]:
            if world.risk.detection_probability > 0.5:
                logger.info("[StrategyEngine] Exfil phase + moderate risk - STEALTH_LONG_GAME")
                return Strategy.STEALTH_LONG_GAME
            elif world.resources.time_remaining_hours < 12:
                logger.info("[StrategyEngine] Exfil phase + time pressure - AGGRESSIVE_SPEED")
                return Strategy.AGGRESSIVE_SPEED
        
        # Default: Balanced stealth-oriented
        logger.info("[StrategyEngine] No rule matched - default STEALTH_LONG_GAME")
        return Strategy.STEALTH_LONG_GAME
    
    def get_profile(self, strategy: Strategy) -> StrategyProfile:
        """Get detailed profile for a strategy"""
        return self.STRATEGY_PROFILES[strategy]
    
    def adapt_decision_by_strategy(
        self,
        options: list,
        scores: Dict[str, float],
        strategy: Strategy
    ) -> str:
        """
        Reweight decision scores based on strategy.
        
        Instead of just picking highest score, constrain by strategy.
        
        Args:
            options: List of choices
            scores: {"option1": 0.75, "option2": 0.60}
            strategy: Global strategy
            
        Returns:
            Chosen option (potentially different from highest score)
        """
        profile = self.get_profile(strategy)
        
        adjusted_scores = {}
        
        for option, base_score in scores.items():
            # Parse option characteristics
            is_risky = self._is_risky(option)
            is_fast = self._is_fast(option)
            is_stealthy = self._is_stealthy(option)
            escalates = self._requires_privilege(option)
            spreads_reach = self._increases_reach(option)
            uses_resources = self._uses_significant_resources(option)
            
            # Apply strategy multipliers
            adjusted = base_score
            
            if is_risky:
                adjusted *= (0.2 + 0.8 * profile.risk_tolerance)
            
            if is_fast:
                adjusted *= (0.2 + 0.8 * profile.time_aggression)
            
            if is_stealthy:
                adjusted *= (0.2 + 0.8 * profile.evasion_strictness)
            
            if escalates:
                adjusted *= (0.2 + 0.8 * profile.privilege_priority)
            
            if spreads_reach:
                adjusted *= (0.2 + 0.8 * profile.lateral_depth)
            
            if uses_resources:
                adjusted *= (0.2 + 0.8 * profile.resource_optimization)
            
            adjusted_scores[option] = adjusted
        
        chosen = max(adjusted_scores, key=adjusted_scores.get)
        logger.info(
            f"[StrategyEngine] Strategy {strategy.value} reweighted scores: "
            f"original winner={max(scores, key=scores.get)}, "
            f"strategy-adjusted winner={chosen}"
        )
        
        return chosen
    
    # ========== Heuristics ==========
    
    def _is_risky(self, option: str) -> bool:
        """Does this option have high detection risk?"""
        risky_keywords = ["active", "exploit", "kernel", "wmi", "psexec", "bulk"]
        return any(kw in option.lower() for kw in risky_keywords)
    
    def _is_fast(self, option: str) -> bool:
        """Does this option execute quickly?"""
        fast_keywords = ["direct", "https", "bulk", "immediate", "kernel"]
        return any(kw in option.lower() for kw in fast_keywords)
    
    def _is_stealthy(self, option: str) -> bool:
        """Does this option minimize detection?"""
        stealthy_keywords = ["passive", "osint", "dns", "stealth", "scheduled", "cloud"]
        return any(kw in option.lower() for kw in stealthy_keywords)
    
    def _requires_privilege(self, option: str) -> bool:
        """Does this require privilege escalation?"""
        privilege_keywords = ["escalate", "system", "kernel", "service", "abuse"]
        return any(kw in option.lower() for kw in privilege_keywords)
    
    def _increases_reach(self, option: str) -> bool:
        """Does this spread across machines?"""
        reach_keywords = ["lateral", "propagate", "wmi", "psexec", "spray", "segment"]
        return any(kw in option.lower() for kw in reach_keywords)
    
    def _uses_significant_resources(self, option: str) -> bool:
        """Does this consume bandwidth/time/privilege?"""
        resource_keywords = ["bulk", "exfil", "large", "bandwidth", "https", "upload"]
        return any(kw in option.lower() for kw in resource_keywords)


if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    engine = StrategyEngine()
    
    print("\n" + "="*70)
    print("STRATEGY ENGINE DEMONSTRATION")
    print("="*70)
    
    # Create world state
    world = WorldState()
    
    # Scenario 1: Early phase, no detection
    print("\n--- Scenario 1: Early phase, no detection ---")
    world.current_phase = OperationalPhase.INFILTRATION
    world.risk.detection_probability = 0.1
    strategy1 = engine.choose(world)
    print(f"Strategy: {strategy1.value}")
    profile1 = engine.get_profile(strategy1)
    print(f"  Risk tolerance: {profile1.risk_tolerance:.1%}")
    print(f"  Time aggression: {profile1.time_aggression:.1%}")
    
    # Scenario 2: Critical risk
    print("\n--- Scenario 2: Critical risk detected ---")
    world.risk.detection_probability = 0.8
    strategy2 = engine.choose(world)
    print(f"Strategy: {strategy2.value}")
    profile2 = engine.get_profile(strategy2)
    print(f"  Evasion strictness: {profile2.evasion_strictness:.1%}")
    
    # Scenario 3: Time running out
    print("\n--- Scenario 3: Time running out ---")
    world.risk.detection_probability = 0.4
    world.resources.time_remaining_hours = 4
    strategy3 = engine.choose(world)
    print(f"Strategy: {strategy3.value}")
    profile3 = engine.get_profile(strategy3)
    print(f"  Time aggression: {profile3.time_aggression:.1%}")
    
    # Scenario 4: Strategy reweighting
    print("\n--- Scenario 4: Strategy-based decision reweighting ---")
    options = ["phishing", "exploit_unpatched", "supply_chain", "bruteforce_creds"]
    scores = {
        "phishing": 0.75,
        "exploit_unpatched": 0.95,
        "supply_chain": 0.60,
        "bruteforce_creds": 0.40
    }
    
    world.risk.detection_probability = 0.2
    world.resources.time_remaining_hours = 24
    strategy = engine.choose(world)
    
    print(f"\nOriginal scores: {scores}")
    chosen = engine.adapt_decision_by_strategy(options, scores, strategy)
    print(f"Strategy ({strategy.value}): {chosen}")
    
    # Try with different strategy (enforced)
    print(f"\nIf forced to STEALTH_LONG_GAME:")
    chosen_stealth = engine.adapt_decision_by_strategy(
        options, scores, Strategy.STEALTH_LONG_GAME
    )
    print(f"Would choose: {chosen_stealth}")
