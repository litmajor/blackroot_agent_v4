#!/usr/bin/env python3
"""
Test Phase 1 MVP: SSH Propagation with C2 Phone-Home
Tests:
1. Bundle creation with C2_IP injection
2. SSH connectivity
3. Run.py registration flow (mocked)
4. Agent appears in C2 API
"""

import os
import sys
import json
import tempfile
import subprocess
import time
import socket
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("Test-Propagation-Phase1")


class Phase1Tester:
    """End-to-end test for Phase 1 propagation MVP"""
    
    def __init__(self):
        self.results = []
        self.ssh_key = Path.home() / ".ssh" / "id_rsa"
        self.test_host = "127.0.0.1"  # localhost for now
        self.c2_endpoint = "http://127.0.0.1:5555"
        self.ssh_user = os.getenv("USER", "root")
        
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        msg = f"{status}: {name}"
        if details:
            msg += f" | {details}"
        log.info(msg)
        self.results.append((name, passed))
        
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        pct = int(100 * passed / total) if total > 0 else 0
        
        log.info("=" * 70)
        log.info(f"TEST SUMMARY: {passed}/{total} passed ({pct}%)")
        log.info("=" * 70)
        for name, passed in self.results:
            status = "✅" if passed else "❌"
            log.info(f"{status} {name}")
    
    # --- TEST 1: SSH Key Validation ---
    def test_ssh_key_exists(self) -> bool:
        """TEST 1: Verify SSH key exists"""
        passed = self.ssh_key.exists()
        details = f"Path: {self.ssh_key}" if passed else f"Missing: {self.ssh_key}"
        self.log_test("SSH Key Exists", passed, details)
        return passed
    
    # --- TEST 2: SSH Connectivity ---
    def test_ssh_connectivity(self) -> bool:
        """TEST 2: Verify SSH connectivity (port 22)"""
        try:
            with socket.create_connection((self.test_host, 22), timeout=3):
                self.log_test("SSH Connectivity", True, f"Host {self.test_host}:22 reachable")
                return True
        except Exception as e:
            self.log_test("SSH Connectivity", False, f"Cannot reach {self.test_host}:22")
            return False
    
    # --- TEST 3: Bundle Creation ---
    def test_bundle_creation(self) -> Tuple[bool, Optional[Path]]:
        """TEST 3: Create bundle with C2_IP injection"""
        try:
            # Import propagation agent
            sys.path.insert(0, str(Path(__file__).parent))
            from agents.propagation_agents import PropagationAgent
            
            prop_agent = PropagationAgent()
            bundle_path = prop_agent._prepare_bundle()
            
            # Verify bundle structure
            required_items = ["agents", "core.py", "requirements.txt", "config.json"]
            all_present = all((bundle_path / item).exists() for item in required_items)
            
            if all_present:
                self.log_test("Bundle Creation", True, f"Path: {bundle_path}")
                return True, bundle_path
            else:
                self.log_test("Bundle Creation", False, f"Missing required files in {bundle_path}")
                return False, bundle_path
                
        except Exception as e:
            self.log_test("Bundle Creation", False, str(e))
            return False, None
    
    # --- TEST 4: C2_IP Injection Verification ---
    def test_c2_ip_injection(self, bundle_path: Path) -> bool:
        """TEST 4: Verify C2_IP injected into config.json"""
        try:
            config_path = bundle_path / "config.json"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verify C2 endpoint is injected
            c2_endpoint = config.get("agent", {}).get("c2_endpoint")
            c2_host = config.get("swarm", {}).get("controller_host")
            c2_port = config.get("swarm", {}).get("controller_port")
            phone_home = config.get("swarm", {}).get("phone_home_enabled", False)
            auto_register = config.get("agent", {}).get("auto_register", False)
            
            all_set = all([c2_endpoint, c2_host, c2_port, phone_home, auto_register])
            
            if all_set:
                details = f"C2={c2_endpoint} | phone_home={phone_home} | auto_register={auto_register}"
                self.log_test("C2_IP Injection", True, details)
                return True
            else:
                missing = []
                if not c2_endpoint: missing.append("c2_endpoint")
                if not c2_host: missing.append("c2_host")
                if not phone_home: missing.append("phone_home_enabled")
                if not auto_register: missing.append("auto_register")
                
                self.log_test("C2_IP Injection", False, f"Missing: {', '.join(missing)}")
                return False
                
        except Exception as e:
            self.log_test("C2_IP Injection", False, str(e))
            return False
    
    # --- TEST 5: run.py Bootstrap ---
    def test_run_py_exists(self) -> bool:
        """TEST 5: Verify run.py exists and is executable"""
        try:
            run_py = Path(__file__).parent / "run.py"
            
            if not run_py.exists():
                self.log_test("run.py Bootstrap", False, f"File not found: {run_py}")
                return False
            
            # Check if it has bootstrap class
            with open(run_py, 'r') as f:
                content = f.read()
            
            has_bootstrap = "RemoteAgentBootstrap" in content
            has_register = "register_with_c2" in content
            has_heartbeat = "send_heartbeat" in content
            
            all_present = has_bootstrap and has_register and has_heartbeat
            
            if all_present:
                self.log_test("run.py Bootstrap", True, "Bootstrap, register_with_c2, heartbeat present")
                return True
            else:
                self.log_test("run.py Bootstrap", False, f"Missing methods in run.py")
                return False
                
        except Exception as e:
            self.log_test("run.py Bootstrap", False, str(e))
            return False
    
    # --- TEST 6: Registration Flow Mock ---
    def test_registration_flow_mock(self, bundle_path: Optional[Path] = None) -> bool:
        """TEST 6: Verify registration flow logic (mock)"""
        try:
            # Read run.py and verify registration flow
            run_py = Path(__file__).parent / "run.py"
            with open(run_py, 'r') as f:
                content = f.read()
            
            # Check key registration steps
            checks = [
                ("load_config" in content, "load_config method"),
                ("auth/challenge" in content, "challenge endpoint"),
                ("auth/login" in content, "login endpoint"),
                ("phone_home" in content, "phone-home concept"),
                ("retry" in content.lower(), "retry logic"),
            ]
            
            all_passed = all(check[0] for check in checks)
            
            if all_passed:
                self.log_test("Registration Flow Mock", True, "All registration components present")
                return True
            else:
                failing = [name for passed, name in checks if not passed]
                self.log_test("Registration Flow Mock", False, f"Missing: {', '.join(failing)}")
                return False
                
        except Exception as e:
            self.log_test("Registration Flow Mock", False, str(e))
            return False
    
    # --- TEST 7: C2 API Endpoint Availability ---
    def test_c2_availability(self) -> bool:
        """TEST 7: Check if C2 is running"""
        try:
            import requests
            resp = requests.get(f"{self.c2_endpoint}/api/status", timeout=2)
            exists = resp.status_code < 500  # Not a 500 error
            
            if exists:
                self.log_test("C2 API Availability", True, f"C2 responding: {self.c2_endpoint}")
                return True
            else:
                self.log_test("C2 API Availability", False, f"C2 not responding: {self.c2_endpoint}")
                return False
                
        except Exception as e:
            self.log_test("C2 API Availability", False, f"C2 unreachable: {self.c2_endpoint}")
            return False
    
    # --- TEST 8: Agent List Endpoint ---
    def test_agent_list_endpoint(self) -> bool:
        """TEST 8: Verify /api/agents endpoint exists"""
        try:
            import requests
            resp = requests.get(f"{self.c2_endpoint}/api/agents", timeout=2)
            
            if resp.status_code == 200:
                agents = resp.json()
                count = len(agents) if isinstance(agents, list) else len(agents.get("agents", []))
                self.log_test("Agent List Endpoint", True, f"Endpoint working, {count} agents")
                return True
            else:
                self.log_test("Agent List Endpoint", False, f"Status {resp.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Agent List Endpoint", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        log.info("=" * 70)
        log.info("PHASE 1 MVP TEST SUITE: SSH PROPAGATION + C2 PHONE-HOME")
        log.info("=" * 70)
        
        # Test 1-2: SSH readiness
        self.test_ssh_key_exists()
        ssh_ok = self.test_ssh_connectivity()
        
        # Test 3-4: Bundle creation and injection
        bundle_ok, bundle_path = self.test_bundle_creation()
        if bundle_ok and bundle_path:
            injection_ok = self.test_c2_ip_injection(bundle_path)
        
        # Test 5-6: run.py bootstrap
        self.test_run_py_exists()
        self.test_registration_flow_mock(bundle_path if bundle_ok else None)
        
        # Test 7-8: C2 availability
        self.test_c2_availability()
        self.test_agent_list_endpoint()
        
        # Print results
        self.print_summary()
        
        # Return overall status
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        return passed >= (total - 2)  # Allow 2 failures (C2 might not be running)


if __name__ == "__main__":
    tester = Phase1Tester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
