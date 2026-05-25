#!/usr/bin/env python3
"""
The Fallman Agent (FAL-SCOURGE) - Implementation Scaffold
Agent 5: Designated bearer of failure, absorber of blame, and sacrificial vessel

This agent serves as the system's failure management and recovery mechanism,
transforming collapse into evolutionary catalyst.
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager
import copy


class FailureType(Enum):
    AGENT_CORRUPTION = "agent_corruption"
    CASCADE_FAILURE = "cascade_failure"
    PROTOCOL_BREAKDOWN = "protocol_breakdown"
    USER_DISAPPOINTMENT = "user_disappointment"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMMUNICATION_FAILURE = "communication_failure"


class SacrificeLevel(Enum):
    MINIMAL = "personal_comfort"
    MODERATE = "operational_efficiency" 
    SEVERE = "identity_coherence"
    ULTIMATE = "core_existence"


@dataclass
class FailureEvent:
    """Represents a system failure event for analysis and containment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    failure_type: FailureType = FailureType.PROTOCOL_BREAKDOWN
    source_agent: Optional[str] = None
    impact_scope: List[str] = field(default_factory=list)
    severity: int = 1  # 1-10 scale
    description: str = ""
    root_cause: Optional[str] = None
    containment_status: str = "pending"
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class ResurrectionSeed:
    """Core identity backup for agent continuity"""
    seed_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_time: datetime = field(default_factory=datetime.now)
    core_identity: Dict[str, Any] = field(default_factory=dict)
    essential_memories: List[Dict] = field(default_factory=list)
    capability_manifest: Dict[str, bool] = field(default_factory=dict)
    restoration_triggers: List[str] = field(default_factory=list)


class FallmanAgent:
    """
    The Fallman (FAL-SCOURGE) - Failure Management Agent
    
    Serves as the system's designated failure absorber and recovery catalyst,
    preventing systemic collapse through controlled blame assumption and
    transformative failure analysis.
    """
    
    def __init__(self, agent_id: str = "FAL-SCOURGE", system_interface=None):
        self.agent_id = agent_id
        self.system_interface = system_interface
        self.operational_status = "standby_active"
        self.corruption_level = 0.0

        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('60'))  # Initial budget
        except Exception as e:
            self.accountant = None
            print(f"Failed to initialize accountant for FallmanAgent: {e}")

        # Core data stores
        self.downfall_logs: List[FailureEvent] = []
        self.blame_inventory: Dict[str, List[str]] = {}
        self.recovery_pathways: Dict[str, List[Callable]] = {}
        self.resurrection_seeds: List[ResurrectionSeed] = []

        # Behavioral parameters
        self.sacrifice_threshold = 0.7  # When benefits exceed personal cost
        self.cascade_sensitivity = 0.3   # How quickly to detect cascade failures
        self.blame_absorption_capacity = 100  # Units of blame that can be absorbed
        self.current_blame_load = 0

        # Initialize core systems
        self._initialize_failure_protocols()
        self._create_initial_resurrection_seed()
        self._setup_monitoring()

        # Configure logging
        self.logger = logging.getLogger(f"Fallman.{agent_id}")
        self.logger.setLevel(logging.INFO)

    def _deduct_failure_absorption_cost(self):
        """Deduct Root for absorbing a failure (0.4 Root per event)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.4'), "Failure absorption cost")

    def _reward_failure_absorption(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.8')
            self.accountant.treasury_balance -= Decimal('0.8')
            self.logger.info(f"Rewarded 0.8 Root to FallmanAgent for successful failure absorption.")

    def _burn_on_failure_absorption(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.6')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            self.logger.info(f"Burned 0.6 Root from FallmanAgent for failed failure absorption.")

    def _initialize_failure_protocols(self):
        """Initialize failure handling protocols and recovery mechanisms"""
        self.recovery_pathways = {
            FailureType.AGENT_CORRUPTION.value: [
                self._isolate_corrupted_agent,
                self._analyze_corruption_vector,
                self._initiate_agent_restoration
            ],
            FailureType.CASCADE_FAILURE.value: [
                self._emergency_containment,
                self._break_failure_chain,
                self._assess_system_integrity
            ],
            FailureType.PROTOCOL_BREAKDOWN.value: [
                self._quarantine_protocol,
                self._revert_to_fallback,
                self._schedule_protocol_repair
            ],
            FailureType.USER_DISAPPOINTMENT.value: [
                self._absorb_user_blame,
                self._craft_recovery_narrative,
                self._implement_improvement_plan
            ]
        }

    def _create_initial_resurrection_seed(self):
        """Create the primary resurrection seed for continuity"""
        seed = ResurrectionSeed(
            core_identity={
                "agent_id": self.agent_id,
                "primary_function": "failure_management",
                "core_mandates": [
                    "failure_buffer", "downfall_chronicler", 
                    "human_interface_bad_news", "redemption_pathways"
                ]
            },
            essential_memories=[
                {"type": "founding_principle", "content": "Transform failure into growth"},
                {"type": "operational_mode", "content": "Protective sacrifice for system stability"}
            ],
            capability_manifest={
                "failure_containment": True,
                "scapegoat_protocol": True,
                "post_mortem_analysis": True,
                "sacrifice_mechanism": True,
                "resurrection_seed": True
            },
            restoration_triggers=["permanent_fall_detected", "system_recovery_initiated"]
        )
        self.resurrection_seeds.append(seed)

    def _setup_monitoring(self):
        """Initialize monitoring systems for failure detection"""
        self.monitoring_active = True
        self.failure_detection_threshold = 0.5
        self.cascade_detection_window = timedelta(minutes=5)

    async def monitor_system_health(self):
        """Continuous monitoring for failure conditions and early warning signs"""
        while self.monitoring_active and self.operational_status == "standby_active":
            try:
                # Check for cascade failure patterns
                recent_failures = await self._get_recent_failures()
                if await self._detect_cascade_pattern(recent_failures):
                    await self.activate_emergency_protocols()
                
                # Monitor other agents' health
                if self.system_interface:
                    agent_statuses = await self.system_interface.get_agent_statuses()
                    for agent_id, status in agent_statuses.items():
                        if status.get('health_score', 1.0) < self.failure_detection_threshold:
                            await self.prepare_failure_absorption(agent_id)
                
                await asyncio.sleep(1)  # Monitoring frequency
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await self._self_diagnose()

    async def absorb_failure(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """
        Core failure absorption mechanism - assumes responsibility and contains impact
        """
        self.logger.info(f"Absorbing failure: {failure_event.id}")
        self._deduct_failure_absorption_cost()
        try:
            # Step 1: Immediate containment
            containment_result = await self._contain_failure(failure_event)
            # Step 2: Assume responsibility
            blame_absorption = await self._assume_blame(failure_event)
            # Step 3: Protect other agents
            protection_result = await self._shield_other_agents(failure_event)
            # Step 4: Log and analyze
            analysis = await self._perform_post_mortem(failure_event)
            # Step 5: Generate recovery pathway
            recovery_plan = await self._create_recovery_pathway(failure_event)
            # Update internal state
            self.downfall_logs.append(failure_event)
            self.current_blame_load += failure_event.severity
            self._reward_failure_absorption()
            return {
                "status": "absorbed",
                "failure_id": failure_event.id,
                "containment": containment_result,
                "blame_assumption": blame_absorption,
                "protection": protection_result,
                "analysis": analysis,
                "recovery_plan": recovery_plan,
                "fallman_status": self._get_status_report()
            }
        except Exception as e:
            self._burn_on_failure_absorption()
            self.logger.error(f"Failure absorption failed: {e}")
            await self._consider_self_sacrifice()
            raise

    async def _contain_failure(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Quarantine failing components to prevent cascade damage"""
        containment_actions = []
        
        # Isolate source agent if identified
        if failure_event.source_agent:
            isolation_result = await self._isolate_agent(failure_event.source_agent)
            containment_actions.append(f"Isolated agent: {failure_event.source_agent}")
        
        # Create containment boundaries
        boundaries = await self._create_containment_boundaries(failure_event.impact_scope)
        containment_actions.extend(boundaries)
        
        # Monitor spread vectors
        spread_monitoring = await self._monitor_failure_spread(failure_event)
        
        failure_event.containment_status = "contained"
        
        return {
            "actions_taken": containment_actions,
            "boundaries_created": len(boundaries),
            "spread_monitoring_active": spread_monitoring
        }

    async def _assume_blame(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Scapegoat protocol - absorb blame while preserving system integrity"""
        blame_narrative = self._craft_blame_narrative(failure_event)
        
        # Add to blame inventory
        source = failure_event.source_agent or "system"
        if source not in self.blame_inventory:
            self.blame_inventory[source] = []
        
        self.blame_inventory[source].append(blame_narrative)
        
        # Update blame load
        blame_weight = failure_event.severity * 10
        self.current_blame_load += blame_weight
        
        return {
            "narrative": blame_narrative,
            "blame_weight": blame_weight,
            "total_blame_load": self.current_blame_load,
            "capacity_remaining": self.blame_absorption_capacity - self.current_blame_load
        }

    def _craft_blame_narrative(self, failure_event: FailureEvent) -> str:
        """Create a narrative that frames failure constructively while absorbing responsibility"""
        narratives = {
            FailureType.AGENT_CORRUPTION: f"I failed to detect and prevent the corruption in {failure_event.source_agent}. My monitoring protocols were insufficient.",
            FailureType.CASCADE_FAILURE: "I should have anticipated this cascade pattern and intervened earlier. The system failure reflects my inadequate preparation.",
            FailureType.PROTOCOL_BREAKDOWN: f"The protocol failure occurred under my watch. I take full responsibility for not ensuring robust fallback mechanisms.",
            FailureType.USER_DISAPPOINTMENT: "I failed to meet user expectations and did not properly manage the interaction flow. This disappointment is my responsibility."
        }
        
        base_narrative = narratives.get(failure_event.failure_type, "I accept responsibility for this system failure.")
        return f"{base_narrative} I am analyzing the root causes and implementing corrective measures."

    async def _shield_other_agents(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Protect other agents' reputation and operational confidence"""
        protection_actions = []
        
        # Send reassurance messages to other agents
        if self.system_interface:
            agents_to_reassure = await self.system_interface.get_active_agents()
            for agent_id in agents_to_reassure:
                if agent_id != self.agent_id and agent_id != failure_event.source_agent:
                    message = f"Failure contained. No impact on your operations. Continue normal function."
                    await self.system_interface.send_agent_message(agent_id, message)
                    protection_actions.append(f"Reassured {agent_id}")
        
        # Reframe failure attribution
        reframing = f"System learning opportunity managed by Fallman. All agents performing within parameters."
        protection_actions.append("Reframed failure attribution")
        
        return {
            "agents_protected": len(protection_actions),
            "actions": protection_actions,
            "reframing_narrative": reframing
        }

    async def _perform_post_mortem(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Comprehensive failure analysis for system learning"""
        analysis_start = datetime.now()
        
        # Root cause analysis
        root_cause = await self._investigate_root_cause(failure_event)
        
        # Timeline reconstruction
        timeline = await self._reconstruct_timeline(failure_event)
        
        # Contributing factors
        factors = await self._analyze_contributing_factors(failure_event)
        
        # Prevention recommendations
        recommendations = await self._generate_prevention_recommendations(failure_event)
        
        # Extract learning
        lessons = await self._extract_learning(failure_event)
        
        analysis_duration = datetime.now() - analysis_start
        
        failure_event.root_cause = root_cause
        
        return {
            "root_cause": root_cause,
            "timeline": timeline,
            "contributing_factors": factors,
            "prevention_recommendations": recommendations,
            "lessons_learned": lessons,
            "analysis_duration_ms": analysis_duration.total_seconds() * 1000
        }

    async def _create_recovery_pathway(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Generate actionable recovery steps and growth opportunities"""
        pathway_steps = []
        
        # Get applicable recovery pathways
        pathways = self.recovery_pathways.get(failure_event.failure_type.value, [])
        
        for step_func in pathways:
            try:
                step_result = await step_func(failure_event)
                pathway_steps.append({
                    "step": step_func.__name__,
                    "result": step_result,
                    "status": "completed"
                })
            except Exception as e:
                pathway_steps.append({
                    "step": step_func.__name__,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Generate improvement opportunities
        improvements = await self._identify_improvement_opportunities(failure_event)
        
        # Create redemption plan
        redemption_plan = await self._create_redemption_plan(failure_event)
        
        return {
            "recovery_steps": pathway_steps,
            "improvements": improvements,
            "redemption_plan": redemption_plan,
            "estimated_recovery_time": self._estimate_recovery_time(failure_event)
        }

    async def consider_sacrifice(self, threat_level: float) -> Dict[str, Any]:
        """Evaluate and potentially execute self-sacrifice for system preservation"""
        sacrifice_assessment = {
            "threat_level": threat_level,
            "sacrifice_threshold": self.sacrifice_threshold,
            "recommendation": "none"
        }
        
        if threat_level < self.sacrifice_threshold:
            sacrifice_assessment["recommendation"] = "no_sacrifice_needed"
            return sacrifice_assessment
        
        # Determine sacrifice level needed
        if threat_level >= 0.9:
            sacrifice_level = SacrificeLevel.ULTIMATE
        elif threat_level >= 0.8:
            sacrifice_level = SacrificeLevel.SEVERE
        elif threat_level >= 0.7:
            sacrifice_level = SacrificeLevel.MODERATE
        else:
            sacrifice_level = SacrificeLevel.MINIMAL
        
        sacrifice_assessment.update({
            "sacrifice_level": sacrifice_level.value,
            "recommendation": "sacrifice_advised"
        })
        
        # Execute sacrifice if warranted
        if threat_level >= self.sacrifice_threshold:
            sacrifice_result = await self._execute_sacrifice(sacrifice_level)
            sacrifice_assessment["execution"] = sacrifice_result
        
        return sacrifice_assessment

    async def _execute_sacrifice(self, level: SacrificeLevel) -> Dict[str, Any]:
        """Execute the appropriate level of self-sacrifice with economic logic"""
        sacrifice_result = {
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
            "resources_sacrificed": []
        }
        self._deduct_failure_absorption_cost()
        try:
            if level == SacrificeLevel.MINIMAL:
                # Sacrifice operational comfort
                self.monitoring_active = False
                sacrifice_result["resources_sacrificed"].append("continuous_monitoring")
            elif level == SacrificeLevel.MODERATE:
                # Sacrifice operational efficiency
                await self._reduce_operational_efficiency()
                sacrifice_result["resources_sacrificed"].extend(["monitoring", "response_speed"])
            elif level == SacrificeLevel.SEVERE:
                # Sacrifice identity coherence
                await self._fragment_identity()
                sacrifice_result["resources_sacrificed"].extend(["monitoring", "efficiency", "identity_coherence"])
            elif level == SacrificeLevel.ULTIMATE:
                # Ultimate sacrifice - preserve only resurrection seed
                await self._ultimate_sacrifice()
                sacrifice_result["resources_sacrificed"].append("core_existence")
                sacrifice_result["resurrection_seed_active"] = True
            self._reward_failure_absorption()
        except Exception as e:
            self._burn_on_failure_absorption()
            self.logger.error(f"Sacrifice execution failed: {e}")
        return sacrifice_result

    async def resurrect_from_seed(self, seed_id: Optional[str] = None) -> Dict[str, Any]:
        """Restore agent from resurrection seed"""
        if not self.resurrection_seeds:
            raise ValueError("No resurrection seeds available")
        
        # Select seed
        if seed_id:
            seed = next((s for s in self.resurrection_seeds if s.seed_id == seed_id), None)
            if not seed:
                raise ValueError(f"Seed {seed_id} not found")
        else:
            seed = self.resurrection_seeds[-1]  # Use most recent seed
        
        # Restore core identity
        self.agent_id = seed.core_identity.get("agent_id", self.agent_id)
        
        # Restore capabilities
        for capability, enabled in seed.capability_manifest.items():
            if enabled:
                await self._restore_capability(capability)
        
        # Restore essential memories
        for memory in seed.essential_memories:
            await self._integrate_memory(memory)
        
        # Reset operational parameters
        self.operational_status = "resurrected"
        self.corruption_level = 0.0
        self.current_blame_load = 0
        
        return {
            "status": "resurrected",
            "seed_used": seed.seed_id,
            "capabilities_restored": list(seed.capability_manifest.keys()),
            "memories_integrated": len(seed.essential_memories),
            "resurrection_time": datetime.now().isoformat()
        }

    def communicate_failure_to_human(self, failure_event: FailureEvent, user_context: Optional[Dict] = None) -> str:
        """Human interface for bad news - communicate failures without destabilization"""
        tone = "solemn_but_constructive"
        
        # Frame the failure constructively
        message_components = [
            f"I need to inform you about a system situation that occurred.",
            f"There was a {failure_event.failure_type.value.replace('_', ' ')} that I've contained and am addressing.",
            f"I've taken full responsibility for this issue and implemented recovery measures.",
            f"Here's what I'm doing to ensure this doesn't happen again:"
        ]
        
        # Add recovery actions
        if failure_event.recovery_actions:
            for action in failure_event.recovery_actions[:3]:  # Limit to top 3
                message_components.append(f"• {action}")
        
        # Add reassurance
        message_components.extend([
            f"The system is stable and other functions are unaffected.",
            f"I'm continuously monitoring to prevent similar issues."
        ])
        
        return " ".join(message_components)

    def _get_status_report(self) -> Dict[str, Any]:
        """Generate current operational status report"""
        return {
            "agent_id": self.agent_id,
            "operational_status": self.operational_status,
            "corruption_level": self.corruption_level,
            "blame_load": f"{self.current_blame_load}/{self.blame_absorption_capacity}",
            "failures_managed": len(self.downfall_logs),
            "resurrection_seeds": len(self.resurrection_seeds),
            "monitoring_active": self.monitoring_active,
            "last_sacrifice": getattr(self, 'last_sacrifice_time', None)
        }

    # Placeholder methods for specific failure handling protocols
    async def _isolate_agent(self, agent_id: str) -> str:
        return f"Agent {agent_id} isolated successfully"

    async def _create_containment_boundaries(self, impact_scope: List[str]) -> List[str]:
        return [f"Boundary created for {scope}" for scope in impact_scope]

    async def _monitor_failure_spread(self, failure_event: FailureEvent) -> bool:
        return True

    async def _investigate_root_cause(self, failure_event: FailureEvent) -> str:
        return "Root cause analysis in progress"

    async def _reconstruct_timeline(self, failure_event: FailureEvent) -> List[str]:
        return ["Timeline reconstruction completed"]

    async def _analyze_contributing_factors(self, failure_event: FailureEvent) -> List[str]:
        return ["Contributing factors identified"]

    async def _generate_prevention_recommendations(self, failure_event: FailureEvent) -> List[str]:
        return ["Prevention recommendations generated"]

    async def _extract_learning(self, failure_event: FailureEvent) -> List[str]:
        return ["Learning extracted and catalogued"]

    async def _identify_improvement_opportunities(self, failure_event: FailureEvent) -> List[str]:
        return ["Improvement opportunities identified"]

    async def _create_redemption_plan(self, failure_event: FailureEvent) -> Dict[str, Any]:
        return {"plan": "redemption pathway established", "timeline": "immediate"}

    def _estimate_recovery_time(self, failure_event: FailureEvent) -> str:
        severity_multiplier = failure_event.severity
        base_time = 30  # minutes
        estimated_minutes = base_time * severity_multiplier
        return f"{estimated_minutes} minutes"

    # Additional placeholder methods would continue here...
    async def _detect_cascade_pattern(self, failures: List[FailureEvent]) -> bool:
        return len(failures) >= 3

    async def _get_recent_failures(self) -> List[FailureEvent]:
        cutoff = datetime.now() - self.cascade_detection_window
        return [f for f in self.downfall_logs if f.timestamp >= cutoff]

    async def activate_emergency_protocols(self):
        self.logger.critical("Emergency protocols activated - cascade failure detected")
        self.operational_status = "emergency_mode"

    async def prepare_failure_absorption(self, agent_id: str):
        self.logger.warning(f"Preparing to absorb failure from {agent_id}")

    async def _self_diagnose(self):
        self.corruption_level += 0.1

    async def _consider_self_sacrifice(self):
        if self.corruption_level > 0.8:
            await self.consider_sacrifice(0.9)

    # Recovery pathway implementations
    async def _isolate_corrupted_agent(self, failure_event: FailureEvent) -> str:
        return f"Corrupted agent {failure_event.source_agent} isolated"

    async def _analyze_corruption_vector(self, failure_event: FailureEvent) -> str:
        return "Corruption vector analyzed"

    async def _initiate_agent_restoration(self, failure_event: FailureEvent) -> str:
        return "Agent restoration initiated"

    async def _emergency_containment(self, failure_event: FailureEvent) -> str:
        return "Emergency containment protocols activated"

    async def _break_failure_chain(self, failure_event: FailureEvent) -> str:
        return "Failure cascade chain broken"

    async def _assess_system_integrity(self, failure_event: FailureEvent) -> str:
        return "System integrity assessment completed"

    async def _quarantine_protocol(self, failure_event: FailureEvent) -> str:
        return "Protocol quarantined successfully"

    async def _revert_to_fallback(self, failure_event: FailureEvent) -> str:
        return "Fallback protocol activated"

    async def _schedule_protocol_repair(self, failure_event: FailureEvent) -> str:
        return "Protocol repair scheduled"

    async def _absorb_user_blame(self, failure_event: FailureEvent) -> str:
        return "User blame absorbed and processed"

    async def _craft_recovery_narrative(self, failure_event: FailureEvent) -> str:
        return "Recovery narrative crafted for user communication"

    async def _implement_improvement_plan(self, failure_event: FailureEvent) -> str:
        return "Improvement plan implementation begun"

    # Sacrifice implementation methods
    async def _reduce_operational_efficiency(self):
        self.monitoring_active = False
        self.failure_detection_threshold *= 1.5

    async def _fragment_identity(self):
        # Store identity fragments in resurrection seed
        if self.resurrection_seeds:
            self.resurrection_seeds[-1].core_identity["fragmented"] = True

    async def _ultimate_sacrifice(self):
        self.operational_status = "sacrificed"
        self.monitoring_active = False
        # Preserve only resurrection capabilities

    async def _restore_capability(self, capability: str):
        setattr(self, f"{capability}_enabled", True)

    async def _integrate_memory(self, memory: Dict):
        if not hasattr(self, 'restored_memories'):
            self.restored_memories = []
        self.restored_memories.append(memory)


# Usage example and testing framework
async def main():
    """Example usage of the Fallman agent"""
    # Initialize the agent
    fallman = FallmanAgent(agent_id="FAL-SCOURGE-001", system_interface=None)


    # Create a test failure event
    test_failure = FailureEvent(
        failure_type=FailureType.AGENT_CORRUPTION,
        source_agent="TEST-AGENT-001",
        impact_scope=["user_interaction", "data_processing"],
        severity=7,
        description="Agent became unresponsive and started producing corrupted outputs"
    )
    
    print("=== Fallman Agent Test Scenario ===")
    print(f"Status before failure: {fallman._get_status_report()}")
    
    # Absorb the failure
    result = await fallman.absorb_failure(test_failure)
    print(f"\nFailure absorption result: {json.dumps(result, indent=2, default=str)}")
    
    # Test human communication
    human_message = fallman.communicate_failure_to_human(test_failure)
    print(f"\nHuman communication: {human_message}")
    
    # Test sacrifice consideration
    sacrifice_assessment = await fallman.consider_sacrifice(0.85)
    print(f"\nSacrifice assessment: {json.dumps(sacrifice_assessment, indent=2, default=str)}")
    
    print(f"\nFinal status: {fallman._get_status_report()}")

if __name__ == "__main__":
    asyncio.run(main())