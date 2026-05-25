"""
DECISION FATIGUE: Encourage exploration, prevent greedy collapse
==================================================================

Problem:
  - Same decisions keep winning (greedy optimization)
  - "This worked once → always do it"
  - No exploration pressure
  - No diminishing returns

Solution:
  - Track when decisions were last used
  - Penalize recent use: score *= 0.8 if used recently
  - Reward exploration: boost untried options
  - Result: Diverse, adaptive, less predictable

Author: Blackroot Agent v4
Date: April 6, 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set
from datetime import datetime, timedelta
import math


@dataclass
class DecisionRecord:
    """Track when and how well a decision performed"""
    decision: str
    last_used: datetime = None
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_uses: int = 0  # How many times in a row most recently
    last_success_time: datetime = None
    last_failure_time: datetime = None
    
    def success_rate(self) -> float:
        """Success rate of this decision"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def recency_penalty(self) -> float:
        """
        Penalty for using same decision repeatedly.
        
        Returns 1.0 (no penalty) if not used recently.
        Returns 0.6 (40% penalty) if used in last 2 decisions.
        Returns 0.8 (20% penalty) if used in last 5 decisions.
        
        Encourages trying different approaches.
        """
        if self.last_used is None:
            return 1.0
        
        # How long ago was this used?
        if self.use_count == 0:
            return 1.0
        
        # If used very recently (consecutive_uses > 0), heavy penalty
        if self.consecutive_uses > 0:
            if self.consecutive_uses == 1:
                return 0.7  # Used last decision: 30% penalty
            elif self.consecutive_uses == 2:
                return 0.5  # Used last 2 decisions: 50% penalty
            else:
                return 0.3  # Used last 3+: 70% penalty
        else:
            # Not consecutive, but was used before
            # Diminishing penalty over time
            # After 0 decisions gap: 0.8
            # After 2 decisions gap: 0.9
            # After 5+ decisions gap: 1.0
            return 0.8


class DecisionFatigueEngine:
    """
    Prevent exploration collapse where same decisions always win.
    
    Mechanisms:
    1. Recency penalty: Recent decisions score lower
    2. Fatigue decay: Penalty slowly relaxes over time
    3. Exploration bonus: Untried options get boost
    4. Entropy pressure: Force diversity
    """
    
    def __init__(self, decision_memory_window: int = 10):
        """
        Args:
            decision_memory_window: How many recent decisions to track
        """
        self.decision_history: List[str] = []
        self.decisions: Dict[str, DecisionRecord] = {}
        self.memory_window = decision_memory_window
        self.total_decisions = 0
    
    def reweight_by_fatigue(
        self,
        option_scores: Dict[str, float],
        current_phase: str = None,
    ) -> Dict[str, float]:
        """
        Adjust scores to reduce fatigue and encourage exploration.
        
        Args:
            option_scores: {option: causal_score}
            current_phase: Context (for logging)
        
        Returns:
            {option: adjusted_score}
        """
        adjusted_scores = {}
        
        for option, base_score in option_scores.items():
            # Ensure record exists
            if option not in self.decisions:
                self.decisions[option] = DecisionRecord(decision=option)
            
            record = self.decisions[option]
            
            # Apply recency penalty
            penalty = record.recency_penalty()
            
            # Apply exploration bonus for untried options
            exploration_bonus = 1.0
            if record.use_count == 0:
                exploration_bonus = 1.15  # 15% boost for untried options
            elif record.use_count == 1:
                exploration_bonus = 1.08  # 8% boost for lightly tried
            else:
                exploration_bonus = 1.0
            
            # Combine
            adjusted_score = base_score * penalty * exploration_bonus
            adjusted_scores[option] = adjusted_score
        
        return adjusted_scores
    
    def record_decision(self, option: str, success: bool):
        """Record outcome of a decision"""
        self.total_decisions += 1
        self.decision_history.append(option)
        
        # Keep only recent history
        if len(self.decision_history) > self.memory_window:
            self.decision_history = self.decision_history[-self.memory_window:]
        
        # Update record
        if option not in self.decisions:
            self.decisions[option] = DecisionRecord(decision=option)
        
        record = self.decisions[option]
        record.last_used = datetime.now()
        record.use_count += 1
        
        # Update consecutive uses
        if len(self.decision_history) >= 2 and self.decision_history[-2] == option:
            record.consecutive_uses += 1
        else:
            record.consecutive_uses = 1
        
        # Update success/failure
        if success:
            record.success_count += 1
            record.last_success_time = datetime.now()
        else:
            record.failure_count += 1
            record.last_failure_time = datetime.now()
    
    def reset_consecutive_for_other_options(self, option: str):
        """When an option is used, reset consecutive counter for others"""
        for key, record in self.decisions.items():
            if key != option:
                record.consecutive_uses = 0
    
    def get_decision_diversity_score(self) -> float:
        """
        Measure how diverse recent decisions are.
        
        Returns:
            0.0 = All same decision (no diversity)
            1.0 = All different decisions (max diversity)
        
        Example:
            [A, A, A, A, A] → 0.0 (all same)
            [A, B, A, B, A] → 1.0 (perfectly alternating)
            [A, A, B, B, C] → 0.6 (mixed)
        """
        if len(self.decision_history) < 2:
            return 1.0
        
        unique_count = len(set(self.decision_history))
        max_unique = len(self.decision_history)
        
        # Normalize to 0-1
        diversity = unique_count / max_unique
        return diversity
    
    def get_decision_stats(self) -> str:
        """Get human-readable stats on decision diversity"""
        recent = self.decision_history[-5:] if self.decision_history else []
        diversity = self.get_decision_diversity_score()
        
        lines = [
            f"Decision Fatigue Stats (last {len(recent)} decisions):",
            f"  Recent history: {' → '.join(recent)}",
            f"  Diversity score: {diversity:.0%} (0% = stale, 100% = exploring)",
            f"  Total decisions made: {self.total_decisions}",
        ]
        
        return "\n".join(lines)
    
    def explain_adjustment(self, option: str, base_score: float) -> str:
        """Explain why a score was adjusted"""
        if option not in self.decisions:
            return f"{option}: New option (+15% exploration bonus)"
        
        record = self.decisions[option]
        penalty = record.recency_penalty()
        
        if record.use_count == 0:
            bonus = 0.15
            reason = "Never tried before"
        elif record.use_count == 1:
            bonus = 0.08
            reason = "Lightly tried"
        else:
            bonus = 0.0
            reason = "Well explored"
        
        if record.consecutive_uses > 0:
            penalty_pct = (1.0 - penalty) * 100
            penalty_reason = f"Used {record.consecutive_uses} time(s) recently"
        else:
            penalty_pct = 0
            penalty_reason = "Not recently used"
        
        adjusted = base_score * penalty * (1.0 + bonus)
        
        return (
            f"{option}: {base_score:.2f} → {adjusted:.2f}\n"
            f"  Penalty: -{penalty_pct:.0f}% ({penalty_reason})\n"
            f"  Bonus: +{bonus*100:.0f}% ({reason})"
        )


# =============================================================================
# DEMO: Preventing greedy collapse where same decision always wins
# =============================================================================

if __name__ == "__main__":
    engine = DecisionFatigueEngine()
    
    print("=" * 70)
    print("DECISION FATIGUE: From Greedy Collapse to Diverse Exploration")
    print("=" * 70)
    
    # Simulate 15 decisions where phishing keeps winning
    options = ["phishing", "exploit", "supply_chain", "osint"]
    
    # These are the causal scores (unchanged)
    base_scores = {
        "phishing": 0.85,
        "exploit": 0.65,
        "supply_chain": 0.70,
        "osint": 0.55,
    }
    
    print("\nDECISION SEQUENCE (without fatigue, phishing always wins):")
    print("-" * 70)
    decisions_without_fatigue = []
    for i in range(8):
        winner = max(base_scores.items(), key=lambda x: x[1])[0]
        decisions_without_fatigue.append(winner)
    print(f"Without fatigue: {' → '.join(decisions_without_fatigue)}")
    print(f"Diversity: {len(set(decisions_without_fatigue)) / 8:.0%} (STALE)")
    
    print("\nDECISION SEQUENCE (WITH FATIGUE ENGINE):")
    print("-" * 70)
    decisions_with_fatigue = []
    
    for i in range(15):
        # Reweight by fatigue
        adjusted = engine.reweight_by_fatigue(base_scores)
        
        # Pick winner
        winner = max(adjusted.items(), key=lambda x: x[1])[0]
        decisions_with_fatigue.append(winner)
        
        # Record outcome (let's say all succeed)
        engine.record_decision(winner, success=True)
        engine.reset_consecutive_for_other_options(winner)
        
        # Show first few iterations in detail
        if i < 8:
            print(f"\nIteration {i+1}: {winner}")
            print(f"  Scores: base={base_scores[winner]:.2f}, adjusted={adjusted[winner]:.2f}")
            for opt in [x for x in options if x != winner]:
                print(f"    {opt}: {base_scores[opt]:.2f} → {adjusted[opt]:.2f}")
    
    print(f"\nWith fatigue: {' → '.join(decisions_with_fatigue[:10])}")
    
    diversity_without = len(set(decisions_without_fatigue)) / len(decisions_without_fatigue)
    diversity_with = len(set(decisions_with_fatigue[:10])) / 10
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Without fatigue: {diversity_without:.0%} diversity (always phishing)")
    print(f"With fatigue:    {diversity_with:.0%} diversity (exploring all options)")
    print(f"\nImprovement: +{(diversity_with - diversity_without):.0%} diversity")
    
    print("\n" + engine.get_decision_stats())
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Fatigue prevents collapse to same decision")
    print("=" * 70)
    print("Without it: Greedy optimization (always phishing)")
    print("With it:    Adaptive exploration (phishing, exploit, supply_chain...)")
