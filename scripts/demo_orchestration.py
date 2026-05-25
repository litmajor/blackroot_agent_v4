#!/usr/bin/env python3
"""
Step 2: Agent Coordination Pipeline - End-to-End Demo

Demonstrates:
1. MissionOrchestrator auto-generating sub-missions (SCAN → ESCALATE → EXTRACT → EXFIL)
2. Typed, versioned agent handoffs
3. Checkpoint/resume with exponential backoff
4. Autonomous mission execution without manual C2 orchestration

Usage:
    python demo_orchestration.py --goal "compromiseHost(192.168.1.100)"
    python demo_orchestration.py --goal "compromiseHost(10.0.0.5)" --max-retries 5
"""

import sys
import logging
import argparse
import time
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s'
)
logger = logging.getLogger('DEMO')

# Import orchestration components
try:
    from orchestration import (
        MissionOrchestrator,
        HandoffResult,
        HandoffChain,
        ScanHandoff,
        EscalateHandoff,
        ExtractHandoff,
        ExfilHandoff,
        CheckpointManager,
        ResumableExecution
    )
    ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"[!] Orchestration module not available: {e}")
    ORCHESTRATION_AVAILABLE = False


class MockScoutAgent:
    """Mock ScoutAgent for demo (simulates real scanning)"""
    
    @staticmethod
    def execute(stage, params: Dict[str, Any]) -> HandoffResult:
        """Simulate network scanning"""
        logger.info(f"[🔍] ScoutAgent: Scanning target {params.get('target')}")
        
        # Simulate scan
        time.sleep(1)
        
        target = params.get('target', 'unknown')
        
        # Generate mock scan results
        handoff = ScanHandoff(
            target=target,
            discovered_hosts=[
                f"{target.rsplit('.', 1)[0]}.1",
                f"{target.rsplit('.', 1)[0]}.2",
                f"{target.rsplit('.', 1)[0]}.3",
                target
            ],
            open_ports={
                target: [22, 80, 443, 3389],
                f"{target.rsplit('.', 1)[0]}.1": [22, 22],
                f"{target.rsplit('.', 1)[0]}.2": [80, 443],
                f"{target.rsplit('.', 1)[0]}.3": [139, 445],
            },
            services={
                target: {22: "SSH", 80: "HTTP", 443: "HTTPS", 3389: "RDP"},
                f"{target.rsplit('.', 1)[0]}.1": {22: "SSH"},
                f"{target.rsplit('.', 1)[0]}.2": {80: "HTTP", 443: "HTTPS"},
                f"{target.rsplit('.', 1)[0]}.3": {139: "NetBIOS", 445: "SMB"},
            },
            os_guesses={
                target: ("Windows", 0.8),
                f"{target.rsplit('.', 1)[0]}.1": ("Linux", 0.9),
                f"{target.rsplit('.', 1)[0]}.2": ("Linux", 0.85),
                f"{target.rsplit('.', 1)[0]}.3": ("Windows", 0.75),
            },
            zombie_candidates=[target, f"{target.rsplit('.', 1)[0]}.3"],
            confidence=0.95,
            version="1.0"
        )
        
        logger.info(f"[✅] SCAN: {len(handoff.discovered_hosts)} hosts, {len(handoff.zombie_candidates)} candidates")
        
        return HandoffResult(
            stage_name="SCAN",
            success=True,
            data=handoff.to_dict(),
            confidence=0.95,
            version="1.0"
        )


class MockPrivescAgent:
    """Mock PrivescAgent for demo (simulates escalation)"""
    
    @staticmethod
    def execute(stage, params: Dict[str, Any]) -> HandoffResult:
        """Simulate privilege escalation"""
        logger.info(f"[🔓] PrivescAgent: Escalating on {params.get('target')}")
        
        # Simulate escalation
        time.sleep(2)
        
        target = params.get('target', 'unknown')
        
        # Use scan results from context
        scan_data = params.get('discovered_hosts', [])
        
        handoff = EscalateHandoff(
            target=target,
            escalation_method="GodPotato",
            success=True,
            privilege_level="SYSTEM",
            tokens={
                'admin_token': f'TOKEN_{target.replace(".", "_")}',
                'system_token': 'SYSTEM_TOKEN_12345'
            },
            persistence_installed=True,
            confidence=0.85,
            version="1.0"
        )
        
        logger.info(f"[✅] ESCALATE: Privilege level = {handoff.privilege_level}")
        
        return HandoffResult(
            stage_name="ESCALATE",
            success=True,
            data=handoff.to_dict(),
            confidence=0.85,
            version="1.0"
        )


class MockExtractorAgent:
    """Mock ExtractorAgent for demo (simulates data extraction)"""
    
    @staticmethod
    def execute(stage, params: Dict[str, Any]) -> HandoffResult:
        """Simulate credential extraction"""
        logger.info(f"[💾] ExtractorAgent: Extracting from {params.get('target')}")
        
        # Simulate extraction
        time.sleep(3)
        
        target = params.get('target', 'unknown')
        
        handoff = ExtractHandoff(
            target=target,
            credentials={
                'LSASS': ['Domain\\Admin:Password123', 'Domain\\User:Pass456'],
                'Registry': ['Administrator:Hash789']
            },
            sensitive_files=[
                f'C:\\Users\\Admin\\Documents\\secret.txt',
                f'C:\\Users\\Admin\\AppData\\Local\\Temp\\key.pem'
            ],
            database_creds={
                'MySQL': 'root:db_password',
                'MSSQL': 'sa:mssql_pass'
            },
            private_keys=[
                'ssh_rsa_key_admin',
                'tls_cert_key'
            ],
            encryption_key='AES_KEY_12345',
            total_size_bytes=1048576,  # 1 MB
            confidence=0.90,
            version="1.0"
        )
        
        logger.info(f"[✅] EXTRACT: {handoff.total_size_bytes} bytes extracted")
        
        return HandoffResult(
            stage_name="EXTRACT",
            success=True,
            data=handoff.to_dict(),
            confidence=0.90,
            version="1.0"
        )


class MockExfiltratorAgent:
    """Mock ExfiltratorAgent for demo (simulates data exfiltration)"""
    
    @staticmethod
    def execute(stage, params: Dict[str, Any]) -> HandoffResult:
        """Simulate data exfiltration"""
        logger.info(f"[📤] ExfiltratorAgent: Exfiltrating from {params.get('target')}")
        
        # Simulate exfiltration
        time.sleep(2)
        
        target = params.get('target', 'unknown')
        
        handoff = ExfilHandoff(
            target=target,
            exfil_method="HTTPS_C2",
            exfiltrated_bytes=1048576,
            chunks_sent=4,
            status="completed",
            c2_callback=f"https://c2.evil.com/callback?target={target}",
            confidence=0.88,
            version="1.0"
        )
        
        logger.info(f"[✅] EXFIL: {handoff.exfiltrated_bytes} bytes sent via {handoff.exfil_method}")
        
        return HandoffResult(
            stage_name="EXFIL",
            success=True,
            data=handoff.to_dict(),
            confidence=0.88,
            version="1.0"
        )


def run_orchestration_demo(goal: str, target: str = None, max_retries: int = 3):
    """Run full orchestration demo"""
    
    if not ORCHESTRATION_AVAILABLE:
        logger.error("[❌] Orchestration module not available")
        return
    
    print("\n" + "="*70)
    print("STEP 2: AGENT COORDINATION PIPELINE - ORCHESTRATION DEMO")
    print("="*70 + "\n")
    
    # Initialize orchestrator
    orchestrator = MissionOrchestrator()
    
    # Register mock agents
    orchestrator.register_agent_executor("ScoutAgent", MockScoutAgent.execute)
    orchestrator.register_agent_executor("PrivescAgent", MockPrivescAgent.execute)
    orchestrator.register_agent_executor("ExtractorAgent", MockExtractorAgent.execute)
    orchestrator.register_agent_executor("ExfiltratorAgent", MockExfiltratorAgent.execute)
    
    logger.info(f"[🎯] Registered 4 mock agents for orchestration")
    
    # Create mission from goal
    mission = orchestrator.create_mission(goal, target)
    
    # Configure retry limits
    for stage in mission.stages:
        stage.max_retries = max_retries
    
    logger.info(f"\n[📋] Mission Stages:")
    for i, stage in enumerate(mission.stages, 1):
        deps = f" (depends on: {', '.join(stage.dependencies)})" if stage.dependencies else ""
        logger.info(f"   {i}. {stage.name}: {stage.agent_type}{deps}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(use_redis=False)  # Use memory-only for demo
    
    logger.info(f"\n[🚀] STARTING MISSION EXECUTION...\n")
    print("-"*70)
    
    # Execute mission stages with checkpoint tracking
    results: Dict[str, Any] = {}
    
    for stage_index, stage in enumerate(mission.stages):
        logger.info(f"\n[→] STAGE {stage_index + 1}/{len(mission.stages)}: {stage.name}")
        logger.info(f"   Agent: {stage.agent_type}")
        
        # Create checkpoint
        checkpoint_manager.create_checkpoint(mission.mission_id, stage.name, stage_index)
        checkpoint_manager.mark_stage_started(mission.mission_id, stage.name)
        
        # Execute stage
        result = orchestrator._execute_stage_with_retry(stage, mission, timeout=300)
        results[stage.name] = result
        
        if result.success:
            logger.info(f"[✅] {stage.name} succeeded (confidence: {result.confidence})")
            checkpoint_manager.mark_stage_completed(mission.mission_id, stage.name, result.data)
        else:
            logger.error(f"[❌] {stage.name} failed: {result.error}")
            checkpoint_manager.mark_stage_failed(
                mission.mission_id,
                stage.name,
                result.error or "Unknown error",
                stage.attempted,
                stage.max_retries
            )
            break
    
    # Display results and handoff chain
    print("\n" + "-"*70)
    print("ORCHESTRATION RESULTS\n")
    
    logger.info(f"[📊] Mission Status:")
    logger.info(f"   Mission ID: {mission.mission_id[:8]}...")
    logger.info(f"   Goal: {mission.goal}")
    logger.info(f"   Completed: {mission.completed_at}")
    
    print()
    
    # Show handoff chain
    chain = HandoffChain()
    for stage_name in ["SCAN", "ESCALATE", "EXTRACT", "EXFIL"]:
        if stage_name in results:
            result = results[stage_name]
            if result.success:
                if stage_name == "SCAN":
                    chain.add_scan_result(result.data)
                elif stage_name == "ESCALATE":
                    chain.add_escalate_result(result.data)
                elif stage_name == "EXTRACT":
                    chain.add_extract_result(result.data)
                elif stage_name == "EXFIL":
                    chain.add_exfil_result(result.data)
    
    print("HANDOFF CHAIN:")
    print(chain.audit_log())
    
    print("\n" + "-"*70)
    print("CHECKPOINT AUDIT TRAIL\n")
    
    # Show checkpoint history
    checkpoints = checkpoint_manager.get_checkpoint_history(mission.mission_id)
    for cp in checkpoints:
        status_icon = {
            'created': '📍',
            'started': '▶️',
            'completed': '✅',
            'failed': '❌'
        }.get(cp.state, '?')
        
        logger.info(f"{status_icon} {cp.stage_name} (Attempt {cp.attempt}): {cp.state}")
        if cp.error:
            logger.info(f"   Error: {cp.error}")
        if cp.recovery_hint:
            logger.info(f"   Recovery: {cp.recovery_hint}")
    
    # Final summary
    print("\n" + "="*70)
    completion_pct = (len([r for r in results.values() if r.success]) / len(mission.stages)) * 100
    if len([r for r in results.values() if r.success]) == len(mission.stages):
        print("✅ MISSION COMPLETED SUCCESSFULLY")
    else:
        print(f"⚠️ MISSION PARTIALLY COMPLETED ({completion_pct:.0f}%)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Agent Coordination Pipeline Demo")
    parser.add_argument('--goal', default="compromiseHost(192.168.1.100)",
                       help="High-level mission goal")
    parser.add_argument('--target', default=None,
                       help="Target IP/host (extracted from goal if not provided)")
    parser.add_argument('--max-retries', type=int, default=3,
                       help="Maximum retries per stage")
    
    args = parser.parse_args()
    
    run_orchestration_demo(args.goal, args.target, args.max_retries)


if __name__ == "__main__":
    main()
