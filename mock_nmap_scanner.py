#!/usr/bin/env python3
"""
Mock nmap scanner for testing network_mapper.py
Simulates real nmap output in a format that python-nmap can parse.
"""

import sys
import json
import random
import re
from datetime import datetime

def parse_args():
    """Parse mock nmap command line arguments."""
    args = sys.argv[1:]
    
    params = {
        'hosts': None,
        'arguments': '',
    }
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg.startswith('-') and not arg[1].isdigit():
            # Flag argument
            if arg in ['-sV', '-O', '-A', '-sS', '-sT']:
                params['arguments'] += arg + ' '
        else:
            # Host specification
            params['hosts'] = arg
        
        i += 1
    
    return params

def generate_mock_scan_results(hosts_spec):
    """Generate mock nmap scan results."""
    import ipaddress
    
    # Parse CIDR notation
    try:
        if '/' in hosts_spec:
            network = ipaddress.IPv4Network(hosts_spec, strict=False)
            host_list = list(network.hosts())[:12]  # Limit to 12 for perf
        else:
            host_list = [ipaddress.IPv4Address(hosts_spec)]
    except:
        host_list = []
    
    results = {
        'nmap': {
            'command': 'nmap [MOCK] ' + hosts_spec,
            'args': '',
            'start': str(datetime.now()),
            'startstr': datetime.now().isoformat(),
            'version': '7.93',
            'xmloutputversion': '1.05'
        },
        'scan': {},
        'scanstats': {
            'timestr': datetime.now().isoformat(),
            'elapsed': '5.23',
            'summary': f'Nmap done at {datetime.now().isoformat()}'
        }
    }
    
    common_ports = {
        22: ('ssh', 'OpenSSH 7.4'),
        80: ('http', 'Apache httpd 2.4'),
        443: ('https', 'nginx 1.16'),
        445: ('microsoft-ds', 'Windows'),
        3306: ('mysql', 'MySQL 5.7'),
        3389: ('ms-wbt-server', 'Windows RDP'),
        5432: ('postgresql', 'PostgreSQL'),
        5985: ('wsman', 'Windows WinRM'),
    }
    
    os_types = [
        'Microsoft Windows 10 Pro',
        'Microsoft Windows Server 2019',
        'Linux 4.15 - 5.6',
        'Ubuntu 20.04 LTS',
        'CentOS Linux 8'
    ]
    
    for ip in host_list:
        ip_str = str(ip)
        
        # Generate random open ports
        num_ports = random.randint(3, 7)
        selected_ports = random.sample(list(common_ports.keys()), min(num_ports, len(common_ports)))
        
        scan_data = {
            'status': {'state': 'up', 'reason': 'echo-reply'},
            'hostnames': [{'name': f'host-{random.randint(1, 999)}', 'type': 'user'}],
            'osmatch': [],
            'tcp': {}
        }
        
        # Add services
        for port in selected_ports:
            service_name, service_version = common_ports[port]
            scan_data['tcp'][port] = {
                'state': 'open',
                'reason': 'syn-ack',
                'name': service_name,
                'product': service_name,
                'version': service_version,
                'extrainfo': 'protocol 2.0'
            }
        
        # OS detection
        if random.random() > 0.3:
            os_type = random.choice(os_types)
            scan_data['osmatch'].append({
                'name': os_type,
                'accuracy': str(random.randint(85, 99)),
                'line': '1'
            })
        
        results['scan'][ip_str] = scan_data
    
    return results

def main():
    params = parse_args()
    
    if not params['hosts']:
        print('Error: No hosts specified', file=sys.stderr)
        sys.exit(1)
    
    # Generate and output results in JSON format (what python-nmap expects)
    results = generate_mock_scan_results(params['hosts'])
    
    # python-nmap will parse this, so we output in a format it understands
    # The library expects XML, but we'll output JSON which it can convert
    print(json.dumps(results))
    
    sys.exit(0)

if __name__ == '__main__':
    main()
