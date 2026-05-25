#!/usr/bin/env python3
"""Test extraction search without CLI overhead"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault
import json

print("1. Initializing...")
v = BlackVault()
g = GhostLayer(vault=v)

print("2. Running extraction...")
result = g.scan_local_credentials()
print(f"   Result type: {type(result)}")

print("3. Checking phases...")
for phase in ['critical_findings', 'phase_1_extracted', 'phase_2_extracted']:
    phase_data = result.get(phase, {})
    print(f"   {phase}: {type(phase_data)} with {len(phase_data)} keys")

print("4. Getting browser_data...")
browser_data = result.get('phase_2_extracted', {}).get('browser_data', [])
print(f"   browser_data type: {type(browser_data)}, len: {len(browser_data) if hasattr(browser_data, '__len__') else '?'}")

print("5. Filtering for payment cards...")
payment_count = 0
for item in browser_data[:10]:  # Just check first 10
    if isinstance(item, dict) and item.get('type') == 'payment':
        payment_count += 1
        print(f"   Found payment: {item.get('cardholder', 'unknown')}")

print(f"   Total payment items checked: {payment_count}")

print("\n[OK] Test complete")
