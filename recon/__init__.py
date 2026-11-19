"""
RECONNAISSANCE & THREAT INTELLIGENCE MODULES
Phase 2 implementation of network discovery and threat analysis
"""

from .network_mapper import RealNetworkMapper, NodeProfile, ServiceInfo
from .threat_intelligence import (
    ThreatIntelligenceLoader,
    ThreatType,
    ThreatSignature,
    CVERecord,
    MaliciousIPRecord,
    get_threat_loader
)
from .network_mapper_integration import EnhancedNetworkMapper, integrate_persistence_with_network_mapper
from .scan_persistence import ScanPersistence, DriftAnomalyDetector
from .resilience import CircuitBreaker, make_canary, ChecksumCache
from .passive_plus import PassivePlus

__all__ = [
    # Core mapper
    'RealNetworkMapper',
    'NodeProfile',
    'ServiceInfo',
    # Enhanced mapper (with real persistence baselines)
    'EnhancedNetworkMapper',
    'integrate_persistence_with_network_mapper',
    # Threat intelligence
    'ThreatIntelligenceLoader',
    'ThreatType',
    'ThreatSignature',
    'CVERecord',
    'MaliciousIPRecord',
    'get_threat_loader',
    # Persistence & drift
    'ScanPersistence',
    'DriftAnomalyDetector',
    # Resilience primitives
    'CircuitBreaker',
    'make_canary',
    'ChecksumCache',
    # Passive recon helpers
    'PassivePlus',
]
