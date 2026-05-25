#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩸 shogun_network.py — Sovereign Network Continuous Awareness Integration
Ultra-passive network topology mapper for Blackroot
"""

import os
import json
import time
import uuid
import hashlib
import random
import socket
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from enum import Enum
from cryptography.fernet import Fernet

# Scapy is optional; guard all imports
SHOGUN_AVAILABLE = False
ARP = None
Ether = None
srp = None
sr1 = None
sniff = None
get_if_hwaddr = None
IP = None
TCP = None
UDP = None
ICMP = None
Raw = None
BOOTP = None
DHCP = None
DNS = None
DNSQR = None
DNSRR = None
RandShort = None
RandIP = None
send = None
conf = None
RandString = None
MacLookup = None
VendorNotFoundError = None
netifaces = None

try:
    import netifaces as _netifaces
    from scapy.all import (  # type: ignore
        ARP as _ARP, Ether as _Ether, srp as _srp, sr1 as _sr1, sniff as _sniff,
        get_if_hwaddr as _get_if_hwaddr,
        IP as _IP, TCP as _TCP, UDP as _UDP, ICMP as _ICMP, Raw as _Raw,
        BOOTP as _BOOTP, DHCP as _DHCP, DNS as _DNS, DNSQR as _DNSQR, DNSRR as _DNSRR,
        RandShort as _RandShort, RandIP as _RandIP, send as _send, conf as _conf, RandString as _RandString
    )
    from mac_vendor_lookup import MacLookup as _MacLookup, VendorNotFoundError as _VendorNotFoundError  # type: ignore
    
    # Assign to module-level variables
    netifaces = _netifaces  # type: ignore
    ARP = _ARP  # type: ignore[assignment]
    Ether = _Ether  # type: ignore[assignment]
    srp = _srp  # type: ignore[assignment]
    sr1 = _sr1  # type: ignore[assignment]
    sniff = _sniff  # type: ignore[assignment]
    get_if_hwaddr = _get_if_hwaddr  # type: ignore[assignment]
    IP = _IP  # type: ignore[assignment]
    TCP = _TCP  # type: ignore[assignment]
    UDP = _UDP  # type: ignore[assignment]
    ICMP = _ICMP  # type: ignore[assignment]
    Raw = _Raw  # type: ignore[assignment]
    BOOTP = _BOOTP  # type: ignore[assignment]
    DHCP = _DHCP  # type: ignore[assignment]
    DNS = _DNS  # type: ignore[assignment]
    DNSQR = _DNSQR  # type: ignore[assignment]
    DNSRR = _DNSRR  # type: ignore[assignment]
    RandShort = _RandShort  # type: ignore
    RandIP = _RandIP  # type: ignore
    send = _send  # type: ignore
    conf = _conf  # type: ignore
    RandString = _RandString  # type: ignore
    MacLookup = _MacLookup  # type: ignore
    VendorNotFoundError = _VendorNotFoundError  # type: ignore
    
    SHOGUN_AVAILABLE = True
except ImportError as e:
    SHOGUN_AVAILABLE = False
    logging.warning(f"Shogun dependencies not available: {e}")

log = logging.getLogger("SHOGUN-NET")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

SHOGUN_ID = hashlib.sha256(b"Blackroot-Shogun").hexdigest()[:16]
PASSIVE_DURATION_BASE = 900      # 15 min base
PASSIVE_JITTER = 600              # ±10 min
ACTIVE_MAX_HOSTS = 8
PORTS_PER_HOST = 8
ZOMBIE_MIN_OBS = 5                # min IP ID samples
ZOMBIE_MAX_PPS = 0.3              # packets per second threshold
BANNER_PROB = 0.12
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3389, 8080]

# ────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────

class NodeStatus(Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    ZOMBIE_CANDIDATE = "ZOMBIE_CANDIDATE"
    UNKNOWN = "UNKNOWN"

@dataclass
class NetworkNode:
    ip: str
    mac: str
    vendor: str
    hostnames: Set[str] = field(default_factory=set)
    services: List[str] = field(default_factory=list)
    open_ports: List[int] = field(default_factory=list)
    os_hints: List[str] = field(default_factory=list)
    dhcp_info: Dict = field(default_factory=dict)
    last_seen: str = ""
    source: str = "passive"
    reliability_score: int = 0
    influence_rating: int = 0
    volatility_index: int = 0
    status: NodeStatus = NodeStatus.UNKNOWN
    pps_estimate: float = 1.0
    zombie_confidence: float = 0.0
    
    def to_dict(self):
        data = asdict(self)
        data["status"] = self.status.value
        data["hostnames"] = list(self.hostnames)
        data["os_hints"] = list(set(self.os_hints))  # dedupe
        return data

@dataclass
class NetworkTopology:
    nodes: Dict[str, NetworkNode] = field(default_factory=dict)
    connections: List[Dict] = field(default_factory=list)
    flows: List[Dict] = field(default_factory=list)
    zombie_candidates: List[Dict] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "nodes": {ip: node.to_dict() for ip, node in self.nodes.items()},
            "connections": self.connections,
            "flows": self.flows,
            "zombie_candidates": self.zombie_candidates,
            "meta": self.meta
        }

# ────────────────────────────────────────────────
# SHOGUN NETWORK MAPPER
# ────────────────────────────────────────────────

class ShogunNetworkMapper:
    def __init__(self, vault=None):
        self.vault = vault
        self.node_db: Dict[str, NetworkNode] = {}
        self.ipid_samples: Dict[str, List[Tuple[float, int]]] = {}
        self.ipid_last: Dict[str, Tuple[float, int]] = {}
        self.lock = threading.Lock()
        
        self.mac_lookup = None
        if SHOGUN_AVAILABLE and MacLookup is not None:
            try:
                self.mac_lookup = MacLookup()
                self.mac_lookup.load_vendors()
            except Exception:
                self.mac_lookup = None
    
    def get_vendor(self, mac: str) -> str:
        """Lookup MAC vendor with fallback."""
        if not mac or not self.mac_lookup:
            return "Unknown"
        try:
            return self.mac_lookup.lookup(mac.upper().replace(":", "")[:6])
        except Exception:
            return "Unknown"
    
    def get_subnet(self) -> str:
        """Auto-detect local subnet."""
        if not SHOGUN_AVAILABLE or netifaces is None:
            return "192.168.1.0/24"
        try:
            gw = netifaces.gateways()['default'][netifaces.AF_INET]
            iface = gw[1]
            addr = netifaces.ifaddresses(iface)[netifaces.AF_INET][0]
            return '.'.join(addr['addr'].split('.')[:3]) + '.0/24'
        except Exception:
            return "192.168.1.0/24"
    
    def passive_listen(self, duration: int = 900) -> Dict[str, NetworkNode]:
        """
        Ultra-passive network sniffer.
        Collects: ARP, DHCP, mDNS, LLMNR, NBNS, SSDP, SNMP, TCP fingerprints, IP ID samples
        """
        if not SHOGUN_AVAILABLE or sniff is None or IP is None:
            log.warning("Shogun dependencies not available; returning empty")
            return {}
        
        log.info(f"[🎤] Passive listening for {duration}s (ARP/DHCP/mDNS/LLMNR/NBNS/SSDP/SNMP/IPID)")
        discovered = {}
        pkt_count = {}
        
        def handler(pkt):
            try:
                now_iso = datetime.utcnow().isoformat()
                src_ip = None
                
                if IP in pkt:
                    src_ip = pkt[IP].src
                
                if src_ip and (src_ip.startswith("127.") or src_ip == "0.0.0.0"):
                    return
                
                # ─── IP ID monitoring for zombies ───
                if IP in pkt and src_ip:
                    ipid = pkt[IP].id
                    ts = time.time()
                    
                    with self.lock:
                        self.ipid_samples.setdefault(src_ip, []).append((ts, ipid))
                        if len(self.ipid_samples[src_ip]) > 30:
                            self.ipid_samples[src_ip].pop(0)
                        self.ipid_last[src_ip] = (ts, ipid)
                        pkt_count[src_ip] = pkt_count.get(src_ip, 0) + 1
                
                # ─── ARP ───
                if ARP in pkt and pkt[ARP].op in (1, 2):
                    ip = pkt[ARP].psrc
                    mac = pkt[ARP].hwsrc.upper()
                    if ip not in discovered:
                        discovered[ip] = NetworkNode(
                            ip=ip,
                            mac=mac,
                            vendor=self.get_vendor(mac),
                            last_seen=now_iso,
                            source="passive"
                        )
                    else:
                        discovered[ip].last_seen = now_iso
                
                # ─── DHCP ───
                if DHCP in pkt and BOOTP in pkt:
                    bootp = pkt[BOOTP]
                    dhcp = pkt[DHCP]
                    mac = ':'.join(f'{b:02x}' for b in bootp.chaddr[:6]).upper()
                    ip_assigned = bootp.yiaddr if bootp.yiaddr != "0.0.0.0" else None
                    
                    opts = dict(dhcp.options[1:]) if len(dhcp.options) > 1 else {}
                    hostname = opts.get("hostname", b"").decode(errors="ignore").strip()
                    vendor_class = opts.get("vendor_class_id", b"").decode(errors="ignore")
                    
                    key = ip_assigned or mac
                    if key not in discovered:
                        discovered[key] = NetworkNode(ip=key, mac=mac, vendor=self.get_vendor(mac))
                    
                    if hostname:
                        discovered[key].hostnames.add(hostname)
                    discovered[key].dhcp_info["vendor_class"] = vendor_class
                
                # ─── mDNS / LLMNR ───
                if DNS in pkt and UDP in pkt and pkt[UDP].dport in (5353, 5355):
                    try:
                        if DNSQR in pkt:
                            qname = pkt[DNSQR].qname.decode(errors='ignore').rstrip('.').lower()
                            if "._" in qname or ".local" in qname:
                                parts = qname.split('.')
                                hostname = parts[0]
                                if src_ip and src_ip not in discovered:
                                    discovered[src_ip] = NetworkNode(ip=src_ip, mac="", vendor="")
                                if src_ip:
                                    discovered[src_ip].hostnames.add(hostname)
                    except Exception:
                        pass
                
                # ─── NBNS (NetBIOS) ───
                if UDP in pkt and pkt[UDP].dport == 137:
                    if Raw in pkt:
                        payload = pkt[Raw].load
                        if len(payload) > 10 and payload[0:2] == b'\x01\x00':
                            try:
                                name = payload[13:13 + payload[12]].decode(errors='ignore').rstrip()
                                if src_ip and src_ip not in discovered:
                                    discovered[src_ip] = NetworkNode(ip=src_ip, mac="", vendor="")
                                if src_ip:
                                    discovered[src_ip].hostnames.add(name)
                            except Exception:
                                pass
                
                # ─── SSDP ───
                if UDP in pkt and pkt[UDP].dport == 1900:
                    if Raw in pkt:
                        try:
                            payload = pkt[Raw].load.decode(errors='ignore')
                            if "M-SEARCH" in payload or "NOTIFY" in payload:
                                server_line = [l for l in payload.split('\r\n') if l.startswith("SERVER:")]
                                if server_line and src_ip:
                                    server = server_line[0].split(":", 1)[1].strip()
                                    if src_ip not in discovered:
                                        discovered[src_ip] = NetworkNode(ip=src_ip, mac="", vendor="")
                                    discovered[src_ip].services.append(f"SSDP:{server}")
                        except Exception:
                            pass
                
                # ─── SNMP traps ───
                if UDP in pkt and pkt[UDP].dport == 162 and src_ip:
                    if src_ip not in discovered:
                        discovered[src_ip] = NetworkNode(ip=src_ip, mac="", vendor="")
                    discovered[src_ip].services.append("SNMP-trap")
                
                # ─── TCP SYN/SYN-ACK fingerprinting ───
                if TCP in pkt and src_ip:
                    tcp = pkt[TCP]
                    ttl = pkt[IP].ttl
                    win = tcp.window
                    
                    if src_ip not in discovered:
                        discovered[src_ip] = NetworkNode(ip=src_ip, mac="", vendor="")
                    
                    node = discovered[src_ip]
                    
                    if ttl == 64 or ttl == 63:
                        if "Likely Linux/Unix" not in node.os_hints:
                            node.os_hints.append("Likely Linux/Unix")
                    elif ttl == 128 or ttl == 127:
                        if "Likely Windows" not in node.os_hints:
                            node.os_hints.append("Likely Windows")
                    elif ttl == 255 or ttl == 254:
                        if "Likely Cisco/router" not in node.os_hints:
                            node.os_hints.append("Likely Cisco/router")
                    
                    if win > 60000 and "Large window → modern stack" not in node.os_hints:
                        node.os_hints.append("Large window → modern stack")
                    
                    # SYN-ACK → port is open
                    if tcp.flags & 0x10:  # ACK
                        dport = tcp.dport
                        if dport not in node.open_ports:
                            node.open_ports.append(dport)
            
            except Exception as e:
                log.debug(f"Packet handler error: {e}")
        
        try:
            sniff(
                filter="arp or ip",
                prn=handler,
                timeout=duration,
                store=0
            )
        except PermissionError:
            log.error("Shogun requires root/admin privileges for packet sniffing")
            return {}
        except Exception as e:
            log.error(f"Sniff error: {e}")
            return {}
        
        # Post-process zombie candidates
        self._update_zombie_candidates(pkt_count)
        
        return discovered
    
    def _update_zombie_candidates(self, pkt_count: Dict[str, int]):
        """Identify idle hosts via IP ID analysis."""
        with self.lock:
            candidates = []
            
            for ip, samples in self.ipid_samples.items():
                if len(samples) < ZOMBIE_MIN_OBS:
                    continue
                
                samples_sorted = sorted(samples, key=lambda x: x[0])
                
                deltas = []
                for i in range(1, len(samples_sorted)):
                    dt = samples_sorted[i][0] - samples_sorted[i - 1][0]
                    did = (samples_sorted[i][1] - samples_sorted[i - 1][1]) % 65536
                    if dt > 0:
                        pps = 1.0 / dt
                        deltas.append((did, pps))
                
                if not deltas:
                    continue
                
                avg_delta = sum(d[0] for d in deltas) / len(deltas)
                avg_pps = sum(d[1] for d in deltas) / len(deltas)
                
                if 0.9 < avg_delta < 1.2 and avg_pps < ZOMBIE_MAX_PPS:
                    confidence = 1.0 - min(avg_pps, 0.3)
                    candidates.append({
                        "ip": ip,
                        "confidence": confidence,
                        "last_seen": datetime.utcfromtimestamp(samples_sorted[-1][0]).isoformat(),
                        "pps_est": round(avg_pps, 3),
                        "sample_count": len(samples)
                    })
            
            log.info(f"[🧟] Zombie candidates detected: {len(candidates)}")
            for c in sorted(candidates, key=lambda x: x["confidence"], reverse=True)[:5]:
                log.info(f"  {c['ip']} — conf:{c['confidence']:.2f} pps:{c['pps_est']}")
            
            return candidates
    
    def grab_banner(self, ip: str, port: int, timeout: int = 4) -> str:
        """Grab service banner (conservative)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((ip, port))
            time.sleep(random.uniform(0.6, 2.8))
            
            if port in {80, 443, 8080, 8443}:
                s.send(b"HEAD / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            
            data = s.recv(512).decode('utf-8', errors='ignore').strip()
            s.close()
            return data[:90] if len(data.strip()) > 8 else ""
        except:
            return ""
    
    def build_topology(self, nodes: Dict[str, NetworkNode]) -> NetworkTopology:
        """Assemble network topology from discovered nodes."""
        topology = NetworkTopology(nodes=nodes)
        
        # Add basic connection inference
        for ip, node in nodes.items():
            if "Likely router" in " ".join(node.os_hints) or "Cisco" in node.vendor:
                topology.flows.append({
                    "source": ip,
                    "target": "gateway",
                    "purpose": "network_control",
                    "frequency": "continuous"
                })
        
        return topology
    
    def score_nodes(self, topology: NetworkTopology) -> NetworkTopology:
        """Apply scoring/status to nodes."""
        for ip, node in topology.nodes.items():
            reliability = min(80 + random.randint(-20, 20), 100)
            influence = min(20 + len(node.open_ports) * 5 + random.randint(0, 30), 100)
            volatility = 50
            
            if len(node.open_ports) > 3:
                volatility += 20
            
            node.reliability_score = reliability
            node.influence_rating = influence
            node.volatility_index = volatility
            
            # Status classification
            pps = node.pps_estimate
            if 0.05 < pps < 0.3:
                node.status = NodeStatus.ZOMBIE_CANDIDATE
                node.zombie_confidence = 1.0 - pps
            elif pps < 0.05:
                node.status = NodeStatus.IDLE
            else:
                node.status = NodeStatus.ACTIVE
        
        return topology
    
    def save_topology(self, topology: NetworkTopology, vault=None) -> str:
        """Save topology to vault and return artifact ID."""
        topology.meta = {
            "version": "shogun-1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "shogun_id": SHOGUN_ID,
            "node_count": len(topology.nodes),
            "zombie_candidates": len(topology.zombie_candidates)
        }
        
        topo_dict = topology.to_dict()
        
        if vault:
            artifact_id = f"shogun_topology_{uuid.uuid4().hex[:8]}"
            vault.store(artifact_id, json.dumps(topo_dict).encode(), ttl=604800)
            log.info(f"[🔒] Topology saved to vault: {artifact_id}")
            return artifact_id
        
        return ""
    
    def get_zombie_candidates(self) -> List[Dict]:
        """Return currently tracked zombie candidates."""
        candidates = []
        with self.lock:
            for ip, samples in self.ipid_samples.items():
                if len(samples) >= ZOMBIE_MIN_OBS:
                    samples_sorted = sorted(samples, key=lambda x: x[0])
                    deltas = []
                    for i in range(1, len(samples_sorted)):
                        dt = samples_sorted[i][0] - samples_sorted[i - 1][0]
                        did = (samples_sorted[i][1] - samples_sorted[i - 1][1]) % 65536
                        if dt > 0:
                            pps = 1.0 / dt
                            deltas.append((did, pps))
                    
                    if deltas:
                        avg_delta = sum(d[0] for d in deltas) / len(deltas)
                        avg_pps = sum(d[1] for d in deltas) / len(deltas)
                        if 0.9 < avg_delta < 1.2 and avg_pps < ZOMBIE_MAX_PPS:
                            candidates.append({
                                "ip": ip,
                                "confidence": 1.0 - avg_pps,
                                "pps_est": avg_pps,
                                "samples": len(samples)
                            })
        
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)
