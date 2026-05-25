#!/usr/bin/env python3
"""Test BlackVault retrieval fix"""

from black_vault import BlackVault
import os

print("[*] Testing BlackVault Retrieval Fix\n")

# Initialize vault
vault = BlackVault()

# Test 1: Store artifact
test_data = b'This is a test artifact for the vault fix'
vault.store('test_artifact_001', test_data)
print("[+] Stored test artifact: test_artifact_001")
print(f"    Data: {test_data[:30]}...")

# Test 2: Get the vault path
vault_path = vault._vault_path('test_artifact_001')
print(f"[*] Vault path: {vault_path}")
print(f"    File exists: {os.path.exists(vault_path)}")

# Test 3: Immediate retrieval (same vault instance)
try:
    retrieved = vault.retrieve('test_artifact_001')
    if retrieved == test_data:
        print("[+] Retrieval Test 1 PASSED: Same instance retrieval works!")
    else:
        print("[-] Retrieval Test 1 FAILED: Data mismatch")
except Exception as e:
    print(f"[-] Retrieval Test 1 FAILED: {e}")

# Test 4: New vault instance retrieval (critical test - different key state)
print("\n[*] Creating new vault instance (simulating fresh load)...")
vault2 = BlackVault()
vault2_path = vault2._vault_path('test_artifact_001')
print(f"[*] New instance vault path: {vault2_path}")
print(f"    Paths match: {vault_path == vault2_path}")

try:
    retrieved2 = vault2.retrieve('test_artifact_001')
    if retrieved2 == test_data:
        print("[+] Retrieval Test 2 PASSED: New instance retrieval works!")
    else:
        print("[-] Retrieval Test 2 FAILED: Data mismatch in new instance")
except Exception as e:
    print(f"[-] Retrieval Test 2 FAILED: {e}")

# Test 5: Multiple artifacts
print("\n[*] Testing multiple artifacts...")
test_artifacts = {
    'credentials_set_1': b'username:password123',
    'api_key_token': b'sk-abc123def456',
    'secret_config': b'{"api": "endpoint", "key": "value"}',
}

for name, data in test_artifacts.items():
    vault.store(name, data)
    print(f"[+] Stored: {name}")

print("\n[*] Retrieving all artifacts...")
success_count = 0
for name, expected_data in test_artifacts.items():
    try:
        retrieved = vault.retrieve(name)
        if retrieved == expected_data:
            print(f"[+] Retrieved: {name} ✓")
            success_count += 1
        else:
            print(f"[-] Retrieved: {name} ✗ (data mismatch)")
    except Exception as e:
        print(f"[-] Retrieved: {name} ✗ ({e})")

print(f"\n[*] Success rate: {success_count}/{len(test_artifacts)} artifacts retrieved")

if success_count == len(test_artifacts):
    print("\n✓ VAULT RETRIEVAL FIX VALIDATED - ALL TESTS PASSED")
else:
    print(f"\n✗ Some retrieval tests failed")

# Cleanup
print("\n[*] Cleaning up test artifacts...")
for name in test_artifacts.keys():
    vault.delete(name)
vault.delete('test_artifact_001')
print("[+] Cleanup complete")
