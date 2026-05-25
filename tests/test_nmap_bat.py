#!/usr/bin/env python3
"""Test nmap.bat subprocess execution."""

import subprocess
import json
import os

def test_nmap_bat():
    nmap_bat = os.path.abspath('nmap.bat')
    cmd = [nmap_bat, '-sV', '-O', '192.168.1.0/24']
    
    print(f'[*] Testing nmap.bat subprocess')
    print(f'[*] Command: {cmd}')
    print(f'[*] nmap.bat exists: {os.path.exists(nmap_bat)}')
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        print(f'[+] Return code: {result.returncode}')
        
        if result.returncode == 0:
            print(f'[+] Stdout length: {len(result.stdout)} bytes')
            
            try:
                data = json.loads(result.stdout)
                num_hosts = len(data.get('scan', {}))
                print(f'[+] Parsed JSON with {num_hosts} hosts')
                
                for ip in list(data['scan'].keys())[:5]:
                    tcp = data['scan'][ip].get('tcp', {})
                    ports = list(tcp.keys())
                    print(f'    {ip}: {len(ports)} ports - {",".join(ports[:3])}...')
                
                print()
                print('[✅] SUCCESS: nmap.bat works correctly!')
                return True
            except json.JSONDecodeError as e:
                print(f'[-] JSON parse error: {e}')
                print(f'[-] Output preview: {result.stdout[:300]}...')
                return False
        else:
            print(f'[-] Subprocess failed')
            print(f'[-] Stderr: {result.stderr}')
            return False
            
    except subprocess.TimeoutExpired:
        print(f'[-] Timeout')
        return False
    except Exception as e:
        print(f'[-] Error: {e}')
        return False

if __name__ == '__main__':
    success = test_nmap_bat()
    exit(0 if success else 1)
