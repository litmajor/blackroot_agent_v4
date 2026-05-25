#!/usr/bin/env python3
"""Display complete network map from nmap.bat scan."""

from recon.network_mapper import RealNetworkMapper

def print_full_network_map():
    mapper = RealNetworkMapper(vault=None, config={'scan_timeout': 30})
    nodes = mapper.scan_subnet('192.168.1.0/24')
    
    print('=' * 100)
    print('COMPLETE NETWORK MAP - REAL NMAP.BAT SCAN')
    print('=' * 100)
    print()
    print(f'Source: {mapper.get_last_scan_source().upper()}')
    print(f'Subnet: 192.168.1.0/24')
    print(f'Total Hosts: {len(nodes)}')
    print()
    
    # Full detailed map
    print('FULL NETWORK TOPOLOGY:')
    print('-' * 100)
    header = f"{'IP Address':<18} {'Status':<20} {'OS':<30} {'Ports':<30} {'Score':<8}"
    print(header)
    print('-' * 100)
    
    for ip, profile in sorted(nodes.items()):
        status = profile.status
        os_name = profile.os if profile.os else 'Unknown'
        ports = ', '.join(map(str, profile.open_ports[:5]))
        if len(profile.open_ports) > 5:
            ports += '...'
        score = profile.zombie_candidate_score
        print(f'{ip:<18} {status:<20} {os_name:<30} {ports:<30} {score:<8.3f}')
    
    print()
    print('ZOMBIE CANDIDATES (Ranked by Propagation Score):')
    print('-' * 100)
    header2 = f"{'Rank':<6} {'IP Address':<18} {'Score':<10} {'Isolation':<12} {'Reputation':<12} {'Ports':<8} Status"
    print(header2)
    print('-' * 100)
    
    candidates = mapper.score_zombie_candidates(min_score=0.45)
    if candidates:
        for rank, (ip, profile, score) in enumerate(candidates, 1):
            iso = profile.isolation_score
            rep = profile.service_reputation
            ports = len(profile.open_ports)
            status = profile.status
            print(f'{rank:<6} {ip:<18} {score:<10.3f} {iso:<12.3f} {rep:<12.3f} {ports:<8} {status}')
    else:
        print('(No candidates above min_score threshold)')
    
    print()
    print('SERVICES DETECTED:')
    print('-' * 100)
    
    service_map = {}
    for ip, profile in sorted(nodes.items()):
        for service in profile.services:
            service_key = f'{service.name} (:{service.port})'
            if service_key not in service_map:
                service_map[service_key] = []
            service_map[service_key].append(ip)
    
    for service, ips in sorted(service_map.items()):
        ip_list = ', '.join(ips[:3])
        if len(ips) > 3:
            ip_list += ' ...'
        print(f'{service:<20} Found on {len(ips)} hosts: {ip_list}')
    
    print()
    print('=' * 100)
    print(f'SCAN COMPLETE - {len(nodes)} hosts mapped via nmap.bat')
    print('=' * 100)

if __name__ == '__main__':
    print_full_network_map()
