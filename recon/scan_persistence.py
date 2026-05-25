#!/usr/bin/env python3
"""
Scan Persistence Layer (Redis + BlackVault)
- Saves scan results to Redis (ephemeral, no disk forensics) with 48hr TTL
- Archives to BlackVault (encrypted, versioned) for long-term retention
- Loads historical scans for comparison from Redis
- Computes drift between scans
- Enables anomaly detection based on real historical data
- Zero disk forensic footprint during operation
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import redis

logger = logging.getLogger(__name__)

# Attempt to import BlackVault for archival
try:
    from black_vault import BlackVault
    BLACKVAULT_AVAILABLE = True
except ImportError:
    BLACKVAULT_AVAILABLE = False
    logger.warning("[!] BlackVault not available; archival disabled (Redis only)")


@dataclass
class ScanRecord:
    """Single scan snapshot"""
    timestamp: str  # ISO format
    subnet: str
    hosts: Dict  # {ip: DiscoveredHost as dict}
    total_hosts_discovered: int
    total_open_ports: int
    total_filtered_ports: int
    scan_duration_seconds: float


class ScanPersistence:
    """
    Manage scan history via Redis + optional BlackVault.
    
    Storage Strategy:
    1. Redis (primary): Current scans, 48hr TTL, instantly available to agents
       Key: scan:{subnet_sanitized}:{timestamp}
       TTL: 48 hours (auto-cleanup, no disk evidence)
    
    2. BlackVault (archive): Optional long-term encrypted storage
       Key: artifact:scan_archive:{subnet}:{timestamp}
       Encrypted: AES-256, versioned, audit trail
    
    Zero disk writes during normal operation (forensic stealth).
    """
    
    REDIS_TTL_SECONDS = 172800  # 48 hours
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    
    def __init__(self, archive_to_blackvault: bool = False):
        """
        Args:
            archive_to_blackvault: If True, also store in encrypted BlackVault
        """
        try:
            self.redis_client = redis.Redis(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"[✓] Redis persistence initialized: {self.REDIS_HOST}:{self.REDIS_PORT}")
        except Exception as e:
            self.redis_available = False
            logger.error(f"[!] Redis unavailable: {e}; reverting to in-memory only")
            self.redis_client = None
        
        self.archive_to_vault = archive_to_blackvault and BLACKVAULT_AVAILABLE
        if self.archive_to_vault:
            try:
                self.vault = BlackVault()
                logger.info("[✓] BlackVault archival enabled (encrypted long-term storage)")
            except Exception as e:
                self.archive_to_vault = False
                logger.warning(f"[!] BlackVault init failed: {e}")
        
        # In-memory fallback (if Redis unavailable)
        self.memory_cache: Dict[str, List[ScanRecord]] = {}
    
    def _sanitize_subnet(self, subnet: str) -> str:
        """Convert subnet to Redis key-safe format"""
        return subnet.replace('/', '_').replace('.', '_')
    
    def _serialize_record(self, record: ScanRecord) -> str:
        """Convert ScanRecord to JSON string"""
        return json.dumps(asdict(record), default=str)
    
    def _deserialize_record(self, data: str) -> ScanRecord:
        """Convert JSON string to ScanRecord"""
        data_dict = json.loads(data)
        return ScanRecord(**data_dict)
    
    def save_scan(self, subnet: str, hosts: Dict, scan_duration: float) -> str:
        """
        Save scan results to Redis (no disk forensics).
        
        Args:
            subnet: e.g., "192.168.1.0/24"
            hosts: {ip: DiscoveredHost objects}
            scan_duration: time taken to scan
        
        Returns:
            Redis key identifier
        """
        # Convert DiscoveredHost objects to dicts
        hosts_dict = {}
        for ip, host in hosts.items():
            if hasattr(host, '__dict__'):
                hosts_dict[ip] = host.__dict__
            else:
                hosts_dict[ip] = host
        
        # Create record
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        record = ScanRecord(
            timestamp=timestamp,
            subnet=subnet,
            hosts=hosts_dict,
            total_hosts_discovered=len(hosts),
            total_open_ports=sum(len(h.get('open_ports', [])) for h in hosts_dict.values()),
            total_filtered_ports=sum(len(h.get('filtered_ports', [])) for h in hosts_dict.values()),
            scan_duration_seconds=scan_duration
        )
        
        # Primary: Save to Redis
        if self.redis_available:
            subnet_safe = self._sanitize_subnet(subnet)
            redis_key = f"scan:{subnet_safe}:{timestamp}"
            serialized = self._serialize_record(record)
            
            try:
                self.redis_client.setex(redis_key, self.REDIS_TTL_SECONDS, serialized)
                logger.info(f"[💾] Scan persisted to Redis (TTL: 48h): {redis_key}")
                result_key = redis_key
            except Exception as e:
                logger.error(f"[!] Redis save failed: {e}; using memory cache")
                if subnet not in self.memory_cache:
                    self.memory_cache[subnet] = []
                self.memory_cache[subnet].append(record)
                result_key = f"memory:{timestamp}"
        else:
            # Fallback: In-memory only (if Redis down)
            if subnet not in self.memory_cache:
                self.memory_cache[subnet] = []
            self.memory_cache[subnet].append(record)
            logger.info(f"[💾] Scan persisted to memory (Redis unavailable): {timestamp}")
            result_key = f"memory:{timestamp}"
        
        # Secondary: Archive to BlackVault (if enabled)
        if self.archive_to_vault:
            try:
                vault_key = f"scan_archive:{subnet}:{timestamp}"
                self.vault.store_artifact(vault_key, serialized, ttl=2592000)  # 30 days
                logger.info(f"[🔒] Scan archived to BlackVault: {vault_key}")
            except Exception as e:
                logger.warning(f"[!] BlackVault archive failed: {e}")
        
        return result_key
    
    def load_scan(self, subnet: str, scan_index: int = 0) -> Optional[ScanRecord]:
        """
        Load a scan from Redis.
        
        Args:
            subnet: e.g., "192.168.1.0/24"
            scan_index: 0 = most recent, 1 = 2nd most recent, etc.
        
        Returns:
            ScanRecord or None if not found
        """
        history = self.get_scan_history(subnet, limit=scan_index + 1)
        if not history or len(history) <= scan_index:
            return None
        return history[scan_index]
    
    def get_scan_history(self, subnet: str, limit: int = 10) -> List[ScanRecord]:
        """Get recent scans for a subnet from Redis"""
        results = []
        
        # Try Redis first
        if self.redis_available:
            try:
                subnet_safe = self._sanitize_subnet(subnet)
                pattern = f"scan:{subnet_safe}:*"
                keys = self.redis_client.keys(pattern)
                
                # Sort by timestamp (descending, most recent first)
                def extract_timestamp(key):
                    parts = key.split(':')
                    return parts[-1] if len(parts) > 2 else ''
                
                keys_sorted = sorted(keys, key=extract_timestamp, reverse=True)[:limit]
                
                for key in keys_sorted:
                    try:
                        serialized = self.redis_client.get(key)
                        if serialized:
                            record = self._deserialize_record(serialized)
                            results.append(record)
                    except Exception as e:
                        logger.warning(f"[!] Failed to deserialize {key}: {e}")
                
                if results:
                    logger.debug(f"[✓] Loaded {len(results)} scans from Redis for {subnet}")
                    return results
            except Exception as e:
                logger.warning(f"[!] Redis query failed: {e}; trying memory cache")
        
        # Fallback: In-memory cache
        if subnet in self.memory_cache:
            results = sorted(self.memory_cache[subnet], key=lambda r: r.timestamp, reverse=True)[:limit]
            logger.debug(f"[✓] Loaded {len(results)} scans from memory for {subnet}")
            return results
        
        return []
    
    def compute_drift(self, subnet: str, current_hosts: Dict) -> Dict:
        """
        Compare current scan against previous scan from Redis.
        
        Returns:
            {
                'previous_scan_timestamp': str,
                'new_hosts': [ips],
                'missing_hosts': [ips],
                'port_changes': {ip: {'new': [ports], 'closed': [ports]}},
                'service_changes': {ip: {'new': [services], 'lost': [services]}},
                'drift_score': float (0.0-1.0, 0=identical, 1=completely different)
            }
        """
        previous = self.load_scan(subnet)
        if not previous:
            return {
                'previous_scan_timestamp': None,
                'new_hosts': list(current_hosts.keys()),
                'missing_hosts': [],
                'port_changes': {},
                'service_changes': {},
                'drift_score': 1.0  # No previous data = max drift
            }
        
        current_ips = set(current_hosts.keys())
        previous_ips = set(previous.hosts.keys())
        
        new_hosts = list(current_ips - previous_ips)
        missing_hosts = list(previous_ips - current_ips)
        
        port_changes = {}
        service_changes = {}
        
        # Compare hosts that exist in both
        for ip in current_ips & previous_ips:
            current_host = current_hosts[ip]
            previous_host = previous.hosts[ip]
            
            # Port changes
            current_open = set(current_host.get('open_ports', []) if hasattr(current_host, 'get') else current_host.__dict__.get('open_ports', []))
            previous_open = set(previous_host.get('open_ports', []))
            
            new_ports = current_open - previous_open
            closed_ports = previous_open - current_open
            
            if new_ports or closed_ports:
                port_changes[ip] = {
                    'new': sorted(list(new_ports)),
                    'closed': sorted(list(closed_ports))
                }
            
            # Service changes (from banners)
            current_services = set()
            previous_services = set()
            
            if hasattr(current_host, '__dict__'):
                current_banners = current_host.__dict__.get('banners', {})
            else:
                current_banners = current_host.get('banners', {})
            
            for port, banner in current_banners.items():
                if hasattr(banner, 'service_name'):
                    current_services.add(banner.service_name)
                elif isinstance(banner, dict):
                    current_services.add(banner.get('service_name', 'unknown'))
            
            for port, banner in previous_host.get('banners', {}).items():
                if isinstance(banner, dict):
                    previous_services.add(banner.get('service_name', 'unknown'))
                else:
                    previous_services.add(getattr(banner, 'service_name', 'unknown'))
            
            new_services = current_services - previous_services
            lost_services = previous_services - current_services
            
            if new_services or lost_services:
                service_changes[ip] = {
                    'new': sorted(list(new_services)),
                    'lost': sorted(list(lost_services))
                }
        
        # Calculate drift score (0-1, higher = more change)
        total_hosts = max(len(current_ips), len(previous_ips))
        if total_hosts == 0:
            drift_score = 0.0
        else:
            changed_hosts = len(new_hosts) + len(missing_hosts) + len(port_changes) + len(service_changes)
            drift_score = min(1.0, changed_hosts / total_hosts)
        
        return {
            'previous_scan_timestamp': previous.timestamp,
            'new_hosts': new_hosts,
            'missing_hosts': missing_hosts,
            'port_changes': port_changes,
            'service_changes': service_changes,
            'drift_score': drift_score,
            'total_hosts_previous': len(previous_ips),
            'total_hosts_current': len(current_ips)
        }



class DriftAnomalyDetector:
    """
    Detect anomalies based on historical scan drift.
    """
    
    DRIFT_THRESHOLDS = {
        'very_low': 0.05,      # 0-5% change = normal
        'low': 0.15,           # 5-15% = minor change
        'medium': 0.35,        # 15-35% = moderate change
        'high': 0.60,          # 35-60% = significant change
        'very_high': 1.0       # 60%+ = major incident
    }
    
    @staticmethod
    def classify_drift(drift_score: float) -> Tuple[str, int]:
        """
        Classify drift into severity level.
        
        Returns:
            (category: str, severity: int 1-5)
        """
        if drift_score <= DriftAnomalyDetector.DRIFT_THRESHOLDS['very_low']:
            return 'normal', 1
        elif drift_score <= DriftAnomalyDetector.DRIFT_THRESHOLDS['low']:
            return 'minor_change', 2
        elif drift_score <= DriftAnomalyDetector.DRIFT_THRESHOLDS['medium']:
            return 'moderate_change', 3
        elif drift_score <= DriftAnomalyDetector.DRIFT_THRESHOLDS['high']:
            return 'significant_change', 4
        else:
            return 'major_incident', 5
    
    @staticmethod
    def generate_anomaly_events(subnet: str, drift: Dict) -> List[Dict]:
        """
        Convert drift data to anomaly events.
        
        Returns list of events like:
        {
            'timestamp': ISO,
            'subnet': subnet,
            'host_ip': ip,
            'event_type': 'port_opened' | 'port_closed' | 'service_appeared' | 'service_disappeared',
            'severity': 1-5,
            'details': {...}
        }
        """
        events = []
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # New hosts
        for ip in drift.get('new_hosts', []):
            events.append({
                'timestamp': timestamp,
                'subnet': subnet,
                'host_ip': ip,
                'event_type': 'host_appeared',
                'severity': 3,
                'details': f"New host discovered on network"
            })
        
        # Missing hosts (likely offline)
        for ip in drift.get('missing_hosts', []):
            events.append({
                'timestamp': timestamp,
                'subnet': subnet,
                'host_ip': ip,
                'event_type': 'host_offline',
                'severity': 2,
                'details': f"Host no longer responding"
            })
        
        # Port changes
        for ip, changes in drift.get('port_changes', {}).items():
            for port in changes.get('new', []):
                events.append({
                    'timestamp': timestamp,
                    'subnet': subnet,
                    'host_ip': ip,
                    'event_type': 'port_opened',
                    'severity': 4,
                    'details': f"Port {port} now open (was closed)"
                })
            
            for port in changes.get('closed', []):
                events.append({
                    'timestamp': timestamp,
                    'subnet': subnet,
                    'host_ip': ip,
                    'event_type': 'port_closed',
                    'severity': 2,
                    'details': f"Port {port} now closed (was open)"
                })
        
        # Service changes
        for ip, changes in drift.get('service_changes', {}).items():
            for svc in changes.get('new', []):
                events.append({
                    'timestamp': timestamp,
                    'subnet': subnet,
                    'host_ip': ip,
                    'event_type': 'service_appeared',
                    'severity': 4,
                    'details': f"Service '{svc}' now present"
                })
            
            for svc in changes.get('lost', []):
                events.append({
                    'timestamp': timestamp,
                    'subnet': subnet,
                    'host_ip': ip,
                    'event_type': 'service_disappeared',
                    'severity': 3,
                    'details': f"Service '{svc}' no longer detected"
                })
        
        return events


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    persistence = ScanPersistence()
    
    # Example: "load latest scan"
    history = persistence.get_scan_history("192.168.1.0/24")
    if history:
        print(f"\n[INFO] Found {len(history)} scans")
        for record in history:
            print(f"  - {record.timestamp}: {record.total_hosts_discovered} hosts, {record.total_open_ports} open ports")
    else:
        print("[INFO] No historical scans found")
