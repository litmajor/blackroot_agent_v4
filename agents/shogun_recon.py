#!/usr/bin/env python3
"""
SHOGUN RECON AGENT
Integrates ultra-passive network topology mapping into Blackroot
"""

import json
import logging
import time
import threading
import random
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, TYPE_CHECKING

from agents.base import BaseAgent
from agent_core_anatomy import AgentID, AgentStatus, Mission

if TYPE_CHECKING:
    from recon.shogun_network import ShogunNetworkMapper

# Runtime import with fallback
_ShogunNetworkMapper = None
SHOGUN_AVAILABLE = False

try:
    from recon.shogun_network import ShogunNetworkMapper as _ShogunNetworkMapper
    SHOGUN_AVAILABLE = True
except ImportError:
    pass

log = logging.getLogger("SHOGUN-RECON")


class ShogunReconAgent(BaseAgent):
    """
    Ultra-passive network topology and zombie candidate discovery.
    Operates at Layer 2-4 with minimal detection risk.
    """
    
    def should_activate(self, config):
        """Check if Shogun should run."""
        return "ShogunRecon" in config.get("agent_whitelist", [])
    
    def __init__(self, agent_id: AgentID = AgentID(value="SHOGUN-RECON")):
        super().__init__(str(agent_id))
        self.agent_id = agent_id
        self.mapper: Optional["ShogunNetworkMapper"] = None
        self.vault = None
        self.swarm = None
        self.last_topology: Optional[Dict[str, Any]] = None
        self.topology_history = []
        
        # Economic integration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('100'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for ShogunReconAgent: {e}")
        
        if SHOGUN_AVAILABLE and _ShogunNetworkMapper is not None:
            self.mapper = _ShogunNetworkMapper(vault=self.vault)
        else:
            log.warning("Shogun network mapper not available; running in stub mode")
    
    def attach_kernel(self, kernel):
        """Attach to kernel and gain access to vault/swarm."""
        super().attach_kernel(kernel)
        self.vault = getattr(kernel, "black_vault", None)
        self.swarm = getattr(kernel, "swarm", None)
        
        if self.mapper and self.vault:
            self.mapper.vault = self.vault
        
        log.info(f"[🩸] ShogunReconAgent attached to kernel")
    
    def _deduct_recon_cost(self):
        """Deduct Root for network reconnaissance (0.3 Root per scan)."""
        if self.accountant:
            from decimal import Decimal
            try:
                self.accountant.process_agent_spend(
                    str(self.agent_id), 
                    Decimal('0.3'), 
                    "Network topology scan"
                )
            except Exception as e:
                log.debug(f"Cost deduction error: {e}")
    
    def _reward_recon_success(self, node_count: int):
        """Reward for discovering network nodes."""
        if self.accountant:
            from decimal import Decimal
            try:
                reward = Decimal(str(min(node_count * 0.1, 5.0)))
                self.accountant.agent_wallets[str(self.agent_id)].balance += reward
                self.accountant.treasury_balance -= reward
                log.debug(f"Rewarded {reward} Root for discovering {node_count} nodes")
            except Exception as e:
                log.debug(f"Reward error: {e}")
    
    def run(self, mission: Optional[Mission] = None):
        """Main Shogun reconnaissance loop."""
        if mission is None:
            mission = Mission(name="default", objectives=[], parameters={})
        super().run()
        log.info("[🩸] Shogun reconnaissance started")
        
        if not SHOGUN_AVAILABLE or self.mapper is None:
            log.warning("Shogun dependencies unavailable; agent will simulate operation")
            self._run_stub_mode()
            return
        
        config = getattr(self.kernel, "config", {}) if self.kernel else {}
        shogun_cfg = config.get("shogun", {})
        
        listen_duration = shogun_cfg.get("passive_duration", 900)
        cycle_interval = shogun_cfg.get("cycle_interval", 3600)
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                log.info(f"[🩸] Shogun cycle {cycle_count} starting...")
                
                # Charge for operation
                self._deduct_recon_cost()
                
                # Passive listen phase
                log.info(f"[🎤] Passive listening for {listen_duration}s...")
                discovered_nodes = self.mapper.passive_listen(listen_duration) if self.mapper else {}
                
                log.info(f"[✓] Discovered {len(discovered_nodes)} nodes")
                
                if discovered_nodes and self.mapper:
                    # Build topology
                    topology = self.mapper.build_topology(discovered_nodes)
                    
                    # Score nodes
                    scored_topology = self.mapper.score_nodes(topology)
                    
                    # Save to vault
                    artifact_id = self.mapper.save_topology(scored_topology, self.vault)
                    
                    # Get zombie candidates
                    zombies = self.mapper.get_zombie_candidates()
                    scored_topology.zombie_candidates = zombies
                    
                    # Update last topology
                    self.last_topology = scored_topology.to_dict()
                    self.topology_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "artifact_id": artifact_id,
                        "node_count": len(discovered_nodes),
                        "zombie_count": len(zombies)
                    })
                    
                    # Keep only last 24 topologies
                    if len(self.topology_history) > 24:
                        self.topology_history.pop(0)
                    
                    # Publish to swarm mesh if available
                    if self.swarm and hasattr(self.swarm, 'redis'):
                        try:
                            self.swarm.redis.publish(
                                "shogun_topology",
                                json.dumps(self.last_topology, default=str)
                            )
                            log.debug(f"Published topology to swarm mesh")
                        except Exception as e:
                            log.debug(f"Failed to publish to swarm: {e}")
                    
                    # Reward for discovery
                    self._reward_recon_success(len(discovered_nodes))
                    
                    # Log zombie candidates
                    if zombies:
                        log.info(f"[🧟] {len(zombies)} zombie candidates identified:")
                        for z in zombies[:3]:
                            log.info(f"    {z['ip']} @ {z['confidence']:.2f} confidence")
                else:
                    log.warning("[!] No nodes discovered in this cycle")
                
                # Sleep before next cycle
                actual_interval = random.randint(cycle_interval, cycle_interval + 1800)
                log.info(f"[🩸] Cycle complete. Sleeping for {actual_interval//60:.0f} minutes")
                time.sleep(actual_interval)
            
            except KeyboardInterrupt:
                log.info("ShogunReconAgent interrupted")
                break
            except PermissionError:
                log.error("Shogun requires root/admin privileges. Retrying in 5 minutes...")
                time.sleep(300)
            except Exception as e:
                log.error(f"[!] ShogunReconAgent error: {e}", exc_info=True)
                time.sleep(300)
    
    def _run_stub_mode(self):
        """Real network scanning mode - Phase 2 replacement for stub mode."""
        log.info("[🔍] Real network mapper activated (Phase 2 implementation)")
        
        try:
            from recon.network_mapper import RealNetworkMapper
        except ImportError:
            log.error("[❌] network_mapper not available; falling back to simulation")
            self._run_fallback_simulation()
            return
        
        config = getattr(self.kernel, "config", {}) if self.kernel else {}
        shogun_cfg = config.get("shogun", {})
        
        # Configuration
        scan_subnets = shogun_cfg.get("scan_subnets", ["192.168.1.0/24"])
        cycle_interval = shogun_cfg.get("cycle_interval", 3600)
        aggressive_scan = shogun_cfg.get("aggressive_scan", False)
        min_zombie_score = shogun_cfg.get("min_zombie_score", 0.6)
        
        cycle_count = 0
        mapper = None
        
        while True:
            try:
                cycle_count += 1
                self._deduct_recon_cost()
                
                # Initialize mapper on first run
                if mapper is None:
                    mapper = RealNetworkMapper(vault=self.vault, config=shogun_cfg)
                    log.info(f"[🗺️] Network mapper initialized")
                
                # Scan subnets
                all_nodes = {}
                for subnet in scan_subnets:
                    log.info(f"[🔍] Starting scan of {subnet}...")
                    try:
                        nodes = mapper.scan_subnet(subnet, aggressive=aggressive_scan)
                        all_nodes.update(nodes)
                    except Exception as e:
                        log.warning(f"[⚠️] Scan failed for {subnet}: {e}")
                        continue
                
                if not all_nodes:
                    log.warning("[!] No nodes discovered in this cycle")
                    # Sleep and retry
                    sleep_duration = random.randint(cycle_interval, cycle_interval + 1800)
                    time.sleep(sleep_duration)
                    continue
                
                # Score zombie candidates
                candidates = mapper.score_zombie_candidates(
                    nodes=all_nodes,
                    min_score=min_zombie_score
                )
                
                # Detect honeypots
                honeypots = mapper.detect_honeypots(nodes=all_nodes)
                
                # Build topology response
                self.last_topology = {
                    "nodes": {
                        ip: node.to_dict()
                        for ip, node in all_nodes.items()
                    },
                    "zombie_candidates": [
                        {
                            "ip": ip,
                            "confidence": min(1.0, score),
                            "profile": node.to_dict(),
                            "reason": "Isolated, low-defense system"
                        }
                        for ip, node, score in candidates
                    ],
                    "honeypots": list(honeypots),
                    "scan_timestamp": datetime.utcnow().isoformat(),
                    "total_hosts": len(all_nodes),
                    "scan_subnets": scan_subnets
                }
                
                self.topology_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_count": len(all_nodes),
                    "zombie_count": len(candidates),
                    "honeypot_count": len(honeypots)
                })
                
                # Keep history limited
                if len(self.topology_history) > 24:
                    self.topology_history.pop(0)
                
                # Publish to swarm mesh if available
                if self.swarm and hasattr(self.swarm, 'redis'):
                    try:
                        self.swarm.redis.publish(
                            "shogun_topology",
                            json.dumps(self.last_topology, default=str)
                        )
                        log.debug(f"[📡] Published topology to swarm mesh")
                    except Exception as e:
                        log.debug(f"[⚠️] Failed to publish to swarm: {e}")
                
                # Reward for discovery
                self._reward_recon_success(len(all_nodes))
                
                # Log zombie candidates
                if candidates:
                    log.info(f"[🧟] {len(candidates)} zombie candidates identified:")
                    for ip, node, score in candidates[:5]:
                        log.info(f"    {ip} @ {score:.2f} score ({node.status})")
                
                # Log honeypots
                if honeypots:
                    log.info(f"[🍯] {len(honeypots)} honeypots detected: {list(honeypots)[:3]}")
                
                # Sleep before next cycle
                actual_interval = random.randint(cycle_interval, cycle_interval + 1800)
                log.info(f"[🩸] Scan cycle complete. Sleeping for {actual_interval//60:.0f} minutes")
                time.sleep(actual_interval)
            
            except KeyboardInterrupt:
                log.info("[🛑] ShogunReconAgent interrupted")
                break
            except PermissionError:
                log.error("[❌] Nmap requires elevated privileges. Retrying in 5 minutes...")
                time.sleep(300)
            except Exception as e:
                log.error(f"[!] Error in network mapper: {e}", exc_info=True)
                time.sleep(300)
    
    def _run_fallback_simulation(self):
        """Fallback simulation when real scanning unavailable."""
        log.warning("[⚠️] Fallback to simulation mode")
        
        config = getattr(self.kernel, "config", {}) if self.kernel else {}
        shogun_cfg = config.get("shogun", {})
        cycle_interval = shogun_cfg.get("cycle_interval", 3600)
        
        while True:
            try:
                self._deduct_recon_cost()
                
                # Simulate discovery
                node_count = random.randint(5, 15)
                log.info(f"[🎤] [SIM] Simulated discovery of {node_count} nodes")
                
                self.last_topology = {
                    "nodes": {
                        f"192.168.1.{10+i}": {
                            "ip": f"192.168.1.{10+i}",
                            "status": random.choice(["ACTIVE", "IDLE", "ZOMBIE_CANDIDATE"]),
                            "vendor": random.choice(["Apple", "Cisco", "Windows", "Linux", "Unknown"]),
                            "open_ports": random.sample(range(22, 5000), random.randint(1, 5))
                        }
                        for i in range(node_count)
                    },
                    "zombie_candidates": [
                        {
                            "ip": f"192.168.1.{15+i}",
                            "confidence": random.uniform(0.8, 0.99),
                            "pps_est": random.uniform(0.01, 0.2)
                        }
                        for i in range(random.randint(1, 3))
                    ]
                }
                
                self._reward_recon_success(node_count)
                
                time.sleep(random.randint(cycle_interval, cycle_interval + 1800))
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"[!] Error in fallback simulation: {e}")
                time.sleep(300)
    
    def get_topology(self) -> Optional[Dict[str, Any]]:
        """Return last collected topology."""
        return self.last_topology
    
    def get_zombie_candidates(self) -> list:
        """Return current zombie candidates."""
        if self.mapper:
            return self.mapper.get_zombie_candidates()
        return []
    
    def get_node_by_ip(self, ip: str) -> Optional[Dict]:
        """Query node by IP address."""
        if self.last_topology and "nodes" in self.last_topology:
            return self.last_topology["nodes"].get(ip)
        return None
    
    def filter_nodes_by_status(self, status: str) -> list:
        """Get all n nodes with specific status."""
        if not self.last_topology or "nodes" not in self.last_topology:
            return []
        
        return [
            node for node in self.last_topology["nodes"].values()
            if node.get("status") == status
        ]
