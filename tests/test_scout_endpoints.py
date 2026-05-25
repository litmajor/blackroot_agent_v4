#!/usr/bin/env python3
"""
Scout Agent Endpoint Testing Suite
Tests all available C2 API endpoints
"""

import requests
import json
import time
import sys
import subprocess
import threading
from typing import Dict, Any, Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 5

class ScoutEndpointTester:
    """Test Scout agent API endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test_endpoint(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     expected_status: int = 200, description: str = "") -> bool:
        """Test a single endpoint"""
        url = f"{API_BASE_URL}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=API_TIMEOUT)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=API_TIMEOUT)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            success = response.status_code == expected_status
            status = "✅ PASS" if success else "❌ FAIL"
            
            self.results.append({
                "status": status,
                "method": method,
                "endpoint": endpoint,
                "description": description,
                "expected": expected_status,
                "actual": response.status_code,
                "response": response.text[:200] if response.text else ""
            })
            
            if success:
                self.passed += 1
            else:
                self.failed += 1
            
            print(f"{status} | {method:4} | {endpoint:40} | {description:40} | {response.status_code}")
            
            return success
            
        except Exception as e:
            self.results.append({
                "status": "❌ ERROR",
                "method": method,
                "endpoint": endpoint,
                "description": description,
                "error": str(e)
            })
            self.failed += 1
            print(f"❌ ERROR | {method:4} | {endpoint:40} | {description:40} | {str(e)[:50]}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*150)
        print("SCOUT AGENT ENDPOINT TEST SUITE")
        print("="*150 + "\n")
        
        # Health endpoints
        print("\n[HEALTH CHECKS]")
        self.test_endpoint("GET", "/health", description="System health check")
        self.test_endpoint("GET", "/health/agents", description="Agent health status")
        self.test_endpoint("GET", "/health/redis", description="Redis connection status")
        self.test_endpoint("GET", "/health/dlq", description="Dead letter queue status")
        
        # Security endpoints
        print("\n[SECURITY & AUTHENTICATION]")
        self.test_endpoint("POST", "/auth/challenge", 
                          data={"agent_id": "scout-test"}, 
                          expected_status=200,
                          description="Auth challenge request")
        
        self.test_endpoint("GET", "/security/status", description="Security status")
        
        # C2 Command Center endpoints
        print("\n[C2 COMMAND CENTER]")
        self.test_endpoint("GET", "/c2/metrics", description="C2 metrics")
        self.test_endpoint("GET", "/c2/agents", description="List active agents")
        self.test_endpoint("GET", "/c2/wallets", description="Wallet information")
        self.test_endpoint("GET", "/c2/layers", description="System layers")
        self.test_endpoint("GET", "/c2/blackvault", description="BlackVault status")
        self.test_endpoint("GET", "/c2/memory", description="Memory statistics")
        self.test_endpoint("GET", "/c2/kernel", description="Kernel status")
        self.test_endpoint("GET", "/c2/economic", description="Economic metrics")
        self.test_endpoint("GET", "/c2/stealth", description="Stealth status")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*150)
        print(f"TEST SUMMARY: {self.passed} PASSED | {self.failed} FAILED")
        print("="*150 + "\n")
        
        if self.failed > 0:
            print("\n[FAILED TESTS]")
            for result in self.results:
                if "FAIL" in result.get("status", "") or "ERROR" in result.get("status", ""):
                    print(f"  {result['method']} {result['endpoint']}: {result.get('error', result.get('actual'))}")

def wait_for_api(timeout: int = 30):
    """Wait for API to be ready"""
    print(f"\n⏳ Waiting for API to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!\n")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(1)
    
    print(f"❌ API failed to start within {timeout}s")
    return False

def main():
    """Main test runner"""
    
    # Check if API is already running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is already running, starting tests...\n")
    except requests.exceptions.ConnectionError:
        print("⚠️  API not running. Please start it with:")
        print("    python main.py --use-scout --api-mode")
        print("\nWaiting for API to start...")
        
        if not wait_for_api():
            print("❌ Could not connect to API")
            sys.exit(1)
    
    # Run test suite
    tester = ScoutEndpointTester()
    tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)

if __name__ == "__main__":
    main()
