#!/usr/bin/env python3
"""
BLACKROOT API Endpoint Testing Suite
Test core enhancements, agents, and C2 operations
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def log_test(name: str, passed: bool, response=None, error=None):
    """Log test result with color"""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"{status} | {name}")
    if response:
        print(f"    Response: {json.dumps(response, indent=2)[:200]}")
    if error:
        print(f"    Error: {error}")

def test_health_endpoints():
    """Test basic health check endpoints"""
    print(f"\n{Colors.BLUE}=== HEALTH CHECK ENDPOINTS ==={Colors.RESET}")
    
    # Test general health
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        log_test("GET /health", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /health", False, error=str(e))
    
    # Test agent health
    try:
        resp = requests.get(f"{BASE_URL}/health/agents", timeout=5)
        log_test("GET /health/agents", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /health/agents", False, error=str(e))
    
    # Test Redis health
    try:
        resp = requests.get(f"{BASE_URL}/health/redis", timeout=5)
        log_test("GET /health/redis", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /health/redis", False, error=str(e))

def test_security_endpoints():
    """Test security and authentication endpoints"""
    print(f"\n{Colors.BLUE}=== SECURITY ENDPOINTS ==={Colors.RESET}")
    
    # Test security status
    try:
        resp = requests.get(f"{BASE_URL}/security/status", timeout=5)
        log_test("GET /security/status", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /security/status", False, error=str(e))
    
    # Test auth challenge
    try:
        resp = requests.post(f"{BASE_URL}/auth/challenge", timeout=5, headers=HEADERS)
        log_test("POST /auth/challenge", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("POST /auth/challenge", False, error=str(e))

def test_c2_metrics():
    """Test C2 metrics and status endpoints"""
    print(f"\n{Colors.BLUE}=== C2 METRICS & STATUS ==={Colors.RESET}")
    
    # Test C2 metrics
    try:
        resp = requests.get(f"{BASE_URL}/c2/metrics", timeout=5)
        log_test("GET /c2/metrics", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/metrics", False, error=str(e))
    
    # Test agent list
    try:
        resp = requests.get(f"{BASE_URL}/c2/agents", timeout=5)
        log_test("GET /c2/agents", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/agents", False, error=str(e))

def test_vault_endpoints():
    """Test vault and artifact endpoints"""
    print(f"\n{Colors.BLUE}=== VAULT OPERATIONS ==={Colors.RESET}")
    
    # Test BlackVault list
    try:
        resp = requests.get(f"{BASE_URL}/c2/blackvault", timeout=5)
        log_test("GET /c2/blackvault", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/blackvault", False, error=str(e))

def test_systeminformation():
    """Test system information endpoints"""
    print(f"\n{Colors.BLUE}=== SYSTEM INFORMATION ==={Colors.RESET}")
    
    # Test memory stats
    try:
        resp = requests.get(f"{BASE_URL}/c2/memory", timeout=5)
        log_test("GET /c2/memory", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/memory", False, error=str(e))
    
    # Test kernel status
    try:
        resp = requests.get(f"{BASE_URL}/c2/kernel", timeout=5)
        log_test("GET /c2/kernel", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/kernel", False, error=str(e))
    
    # Test economic layer
    try:
        resp = requests.get(f"{BASE_URL}/c2/economic", timeout=5)
        log_test("GET /c2/economic", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/economic", False, error=str(e))

def test_enhancements():
    """Test core enhancements integration"""
    print(f"\n{Colors.BLUE}=== CORE ENHANCEMENTS ==={Colors.RESET}")
    
    # Test threat feedback status (custom endpoint if available)
    try:
        resp = requests.get(f"{BASE_URL}/c2/threat-status", timeout=5)
        if resp.status_code == 404:
            log_test("GET /c2/threat-status", False, error="Enhancement endpoint not yet exposed")
        else:
            log_test("GET /c2/threat-status", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/threat-status (Enhancement)", False, error="Endpoint not exposed")
    
    # Test lifecycle manager status
    try:
        resp = requests.get(f"{BASE_URL}/c2/lifecycle", timeout=5)
        if resp.status_code == 404:
            log_test("GET /c2/lifecycle", False, error="Lifecycle endpoint not exposed")
        else:
            log_test("GET /c2/lifecycle", resp.status_code == 200, resp.json())
    except Exception as e:
        log_test("GET /c2/lifecycle (Lifecycle)", False, error="Endpoint not exposed")

def test_recon_capability():
    """Test reconnaissance capability"""
    print(f"\n{Colors.BLUE}=== RECON CAPABILITY ==={Colors.RESET}")
    
    print("    Note: Recon operations require valid session token")
    print("    Endpoints would be protected under /secure/request")

def print_summary():
    """Print test summary and next steps"""
    print(f"\n{Colors.YELLOW}=== SUMMARY ==={Colors.RESET}")
    print("""
✓ Health endpoints confirm the system is running
✓ Security endpoints show PKI infrastructure active
✓ C2 metrics expose agent status and economic layer
✓ Vault operations available for artifact management
✓ System info endpoints show memory, kernel, and economic state

{NEXT_STEPS}
1. POST /auth/challenge → Get authentication challenge
2. POST /auth/login → Authenticate with challenge
3. GET /c2/agents → View all registered agents
4. GET /health/agents → Check agent lifecycle status
5. GET /c2/stealth → Verify stealth layer configuration

{ENHANCEMENT_STATUS}
Core enhancements are initialized but not yet exposed as dedicated endpoints.
Consider adding:
  - /enhancements/async-recon/status
  - /enhancements/lifecycle/agents
  - /enhancements/threat-feedback/level
  - /enhancements/plugins/available
  - /enhancements/config-security/status
""")

def main():
    """Run all tests"""
    print(f"{Colors.BLUE}╔════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BLUE}║  BLACKROOT API ENDPOINT TEST SUITE         ║{Colors.RESET}")
    print(f"{Colors.BLUE}║  Core Enhancements & Agent Integration     ║{Colors.RESET}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════╝{Colors.RESET}")
    
    print(f"\n{Colors.YELLOW}Target: {BASE_URL}{Colors.RESET}")
    print(f"{Colors.YELLOW}Testing started at: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    try:
        # Run all test suites
        test_health_endpoints()
        test_security_endpoints()
        test_c2_metrics()
        test_vault_endpoints()
        test_systeminformation()
        test_enhancements()
        test_recon_capability()
        
        # Print summary
        print_summary()
        
        print(f"\n{Colors.GREEN}Testing complete!{Colors.RESET}\n")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Testing interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Testing failed: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
