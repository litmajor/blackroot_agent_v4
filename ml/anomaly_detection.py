#!/usr/bin/env python3
"""
ML-BASED ANOMALY DETECTION - Phase 2.3 Implementation
Advanced behavioral anomaly detection using Isolation Forest and statistical methods.
Trains on network behavior patterns to identify honeypots and anomalous systems.
"""

import logging
import json
import pickle
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

# Try to import sklearn; fail gracefully if not available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available; using fallback anomaly detection")
    # Type stubs for when sklearn not available
    if TYPE_CHECKING:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler


@dataclass
class BehavioralFeatures:
    """Extracted behavioral features from a node profile"""
    port_count: float
    service_count: float
    os_accuracy: float
    response_time_variance: float
    service_diversity: float  # 0-1 based on service types
    fingerprint_entropy: float
    open_port_distribution: float  # How spread out ports are
    service_port_consistency: float  # How well services match expected ports
    behavioral_anomaly_score: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models"""
        return np.array([
            self.port_count,
            self.service_count,
            self.os_accuracy,
            self.response_time_variance,
            self.service_diversity,
            self.fingerprint_entropy,
            self.open_port_distribution,
            self.service_port_consistency,
        ], dtype=np.float32)
    
    @staticmethod
    def from_node_profile(node_profile: Any) -> "BehavioralFeatures":
        """Extract features from a NodeProfile object"""
        ports = node_profile.open_ports if hasattr(node_profile, 'open_ports') else []
        services = node_profile.services if hasattr(node_profile, 'services') else {}
        fingerprint = node_profile.system_fingerprint if hasattr(node_profile, 'system_fingerprint') else ""
        response_times = node_profile.response_timing if hasattr(node_profile, 'response_timing') else {}
        os_accuracy = node_profile.os_accuracy if hasattr(node_profile, 'os_accuracy') else 0.5
        
        # Calculate features
        port_count = float(len(ports))
        service_count = float(len(services))
        
        # Service diversity (how varied the services are)
        service_diversity = len(set(services.values())) / max(len(services), 1)
        
        # Fingerprint entropy
        fingerprint_entropy = calculate_entropy(fingerprint) if fingerprint else 0.0
        
        # Port distribution (how spread out are the ports)
        port_distribution = calculate_port_distribution(ports) if ports else 0.0
        
        # Response time variance
        response_variance = np.var(list(response_times.values())) if response_times else 0.0
        response_variance = min(response_variance, 10000.0) / 10000.0  # Normalize
        
        # Service-port consistency
        consistency = calculate_port_service_consistency(ports, services)
        
        return BehavioralFeatures(
            port_count=float(min(port_count, 65535.0) / 65535.0),  # Normalize to 0-1
            service_count=float(min(service_count, 100.0) / 100.0),
            os_accuracy=float(os_accuracy),
            response_time_variance=float(response_variance),
            service_diversity=float(service_diversity),
            fingerprint_entropy=float(fingerprint_entropy),
            open_port_distribution=float(port_distribution),
            service_port_consistency=float(consistency)
        )


class MLAnomalyDetector:
    """
    ML-based anomaly detection using Isolation Forest.
    Identifies honeypots and anomalous systems based on behavioral patterns.
    """
    
    def __init__(self, model_path: str = "./models/anomaly_detector.pkl"):
        self.model_path = model_path
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = [
            'port_count', 'service_count', 'os_accuracy', 'response_time_variance',
            'service_diversity', 'fingerprint_entropy', 'port_distribution',
            'port_service_consistency'
        ]
        self.is_trained = False
        self.training_samples = 0
        
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or create new untrained model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
                    self.model = state.get('model')
                    self.scaler = state.get('scaler')
                    self.is_trained = state.get('is_trained', False)
                    self.training_samples = state.get('training_samples', 0)
                log.info(f"Loaded anomaly detector model from {self.model_path}")
            except Exception as e:
                log.warning(f"Failed to load anomaly model: {e}; initializing new model")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new, untrained Isolation Forest model"""
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=0.1,  # ~10% of data likely to be anomalies
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
            log.info("Created new Isolation Forest anomaly detector")
        else:
            log.warning("sklearn not available; anomaly detection will use statistical fallback")
    
    def train(self, node_profiles: List[Any], labels: Optional[List[int]] = None):
        """
        Train the anomaly detector on normal node profiles.
        
        Args:
            node_profiles: List of normal NodeProfile objects to train on
            labels: Optional labels (1 for anomaly, -1 for normal) for semi-supervised learning
        """
        if not node_profiles:
            log.warning("No profiles provided for training")
            return
        
        # Extract features
        features = []
        for profile in node_profiles:
            bf = BehavioralFeatures.from_node_profile(profile)
            features.append(bf.to_vector())
        
        X = np.array(features, dtype=np.float32)
        
        if not SKLEARN_AVAILABLE:
            log.warning("sklearn not available; cannot perform ML training")
            self.training_samples = len(node_profiles)
            return
        
        try:
            # Ensure scaler and model exist
            if self.scaler is None and SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()

            # Scale features (guarding against scaler being None)
            if self.scaler is not None:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            # Train model - IsolationForest expects X only; ignore labels if provided
            if self.model is None:
                log.warning("No ML model available; skipping ML training")
            else:
                if labels is not None:
                    log.info("Labels provided; labels will be ignored for IsolationForest training")
                self.model.fit(X_scaled)

            self.is_trained = True
            self.training_samples = len(node_profiles)

            # Save model
            self._save_model()

            log.info(f"Trained anomaly detector on {len(node_profiles)} profiles")
        except Exception as e:
            log.error(f"Error training anomaly detector: {e}")
    
    def detect_anomaly(self, node_profile: Any) -> Tuple[float, str]:
        """
        Detect if a node profile is anomalous.
        
        Returns:
            (anomaly_score: 0.0-1.0, reason: str)
            Score > 0.5 likely indicates honeypot/anomaly
        """
        if not self.is_trained and SKLEARN_AVAILABLE:
            log.warning("Model not trained; using statistical fallback")
            return self._detect_anomaly_statistical(node_profile)
        
        try:
            bf = BehavioralFeatures.from_node_profile(node_profile)
            X = bf.to_vector().reshape(1, -1)
            
            if SKLEARN_AVAILABLE and self.model and self.scaler:
                X_scaled = self.scaler.transform(X)
                
                # Get anomaly score (-1 for anomalies, 1 for normal)
                prediction = self.model.predict(X_scaled)[0]
                
                # Get raw score (lower = more anomalous)
                raw_score = self.model.score_samples(X_scaled)[0]

                # Convert to 0-1 scale (0 = normal, 1 = anomalous)
                # Cast to float to avoid returning numpy scalar types
                anomaly_score = float(1.0 / (1.0 + np.exp(raw_score)))  # Sigmoid
                
                if prediction == -1:
                    reason = "Isolation Forest flagged as anomaly"
                else:
                    reason = "Normal behavior pattern"
                
                return float(anomaly_score), reason
            else:
                return self._detect_anomaly_statistical(node_profile)
        
        except Exception as e:
            log.error(f"Error in anomaly detection: {e}")
            return self._detect_anomaly_statistical(node_profile)
    
    def _detect_anomaly_statistical(self, node_profile: Any) -> Tuple[float, str]:
        """Fallback statistical anomaly detection"""
        bf = BehavioralFeatures.from_node_profile(node_profile)
        anomaly_score = 0.0
        reasons = []
        
        # High port count (>100 open ports is suspicious)
        if bf.port_count > 0.8:
            anomaly_score += 0.25
            reasons.append("too_many_open_ports")
        
        # High service count (>20 services is suspicious)
        if bf.service_count > 0.8:
            anomaly_score += 0.25
            reasons.append("too_many_services")
        
        # Perfect or near-perfect fingerprint (artificial)
        if bf.fingerprint_entropy < 0.1:
            anomaly_score += 0.2
            reasons.append("low_fingerprint_entropy")
        
        # Unusual service-port mappings
        if bf.service_port_consistency < 0.3:
            anomaly_score += 0.2
            reasons.append("unusual_port_mappings")
        
        # Very high response time variance (inconsistent)
        if bf.response_time_variance > 0.8:
            anomaly_score += 0.1
            reasons.append("inconsistent_responses")
        
        anomaly_score = float(min(anomaly_score, 1.0))
        reason = f"Statistical anomaly: {', '.join(reasons)}" if reasons else "Normal"

        return anomaly_score, reason
    
    def batch_detect(self, node_profiles: List[Any]) -> Dict[str, Any]:
        """Detect anomalies in a batch of profiles"""
        results = []
        for profile in node_profiles:
            score, reason = self.detect_anomaly(profile)
            results.append({
                'ip': getattr(profile, 'ip_address', 'unknown'),
                'anomaly_score': score,
                'reason': reason,
                'is_anomaly': score > 0.5
            })
        
        avg = np.mean([r['anomaly_score'] for r in results]) if results else 0.0
        return {
            'results': results,
            'total': len(node_profiles),
            'anomalies_detected': sum(1 for r in results if r['is_anomaly']),
            'average_score': float(avg)
        }
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            
            state = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'training_samples': self.training_samples,
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(state, f)
            
            log.info(f"Saved anomaly detector model to {self.model_path}")
        except Exception as e:
            log.error(f"Error saving anomaly model: {e}")


class StatisticalAnomalyDetector:
    """
    Fallback statistical anomaly detection when ML models unavailable.
    Uses z-score and interquartile range methods.
    """
    
    def __init__(self):
        self.baseline_features: Dict[str, Dict] = {}
        self.is_trained = False
    
    def train(self, node_profiles: List[Any]):
        """Train statistical models from normal profiles"""
        features_by_name = {name: [] for name in ['port_count', 'service_count', 'os_accuracy']}
        
        for profile in node_profiles:
            bf = BehavioralFeatures.from_node_profile(profile)
            features_by_name['port_count'].append(bf.port_count)
            features_by_name['service_count'].append(bf.service_count)
            features_by_name['os_accuracy'].append(bf.os_accuracy)
        
        # Calculate statistics
        for name, values in features_by_name.items():
            self.baseline_features[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25)
            }
        
        self.is_trained = True
        log.info("Statistical anomaly detector trained")
    
    def detect_anomaly(self, node_profile: Any) -> Tuple[float, str]:
        """Detect anomalies using statistical methods"""
        if not self.is_trained:
            return 0.0, "Statistical detector not trained"
        
        bf = BehavioralFeatures.from_node_profile(node_profile)
        anomaly_score = 0.0
        reasons = []
        
        # Check each feature for outliers
        for feature_name, value in {
            'port_count': bf.port_count,
            'service_count': bf.service_count,
            'os_accuracy': bf.os_accuracy
        }.items():
            if feature_name in self.baseline_features:
                stats = self.baseline_features[feature_name]
                z_score = abs((value - stats['mean']) / (stats['std'] + 1e-6))
                
                if z_score > 3:  # More than 3 standard deviations = anomalous
                    anomaly_score += 0.25
                    reasons.append(f"{feature_name}_outlier")
                elif z_score > 2:  # 2-3 standard deviations = suspicious
                    anomaly_score += 0.10
        
        anomaly_score = min(anomaly_score, 1.0)
        reason = f"Statistical anomaly: {', '.join(reasons)}" if reasons else "Normal"
        
        return anomaly_score, reason


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string"""
    if not data:
        return 0.0
    
    entropy = 0.0
    for char in set(data):
        p = data.count(char) / len(data)
        entropy -= p * np.log2(p)
    
    # Normalize to 0-1
    return entropy / np.log2(max(len(set(data)), 2))


def calculate_port_distribution(ports: List[int]) -> float:
    """
    Calculate how spread out ports are.
    Honeypots often have clustered ports, real systems are more distributed.
    """
    if not ports or len(ports) < 2:
        return 0.5
    
    sorted_ports = sorted(ports)
    gaps = [sorted_ports[i+1] - sorted_ports[i] for i in range(len(sorted_ports)-1)]
    
    if not gaps:
        return 0.5
    
    avg_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    
    # High std deviation = well distributed; low = clustered
    distribution_score = std_gap / (avg_gap + std_gap + 1e-6)
    
    return float(min(distribution_score, 1.0))


def calculate_port_service_consistency(ports: List[int], services: Dict[int, str]) -> float:
    """
    Calculate how well services match their standard ports.
    Real systems usually have services on standard ports.
    """
    standard_ports = {
        22: 'ssh', 25: 'smtp', 53: 'dns', 80: 'http', 110: 'pop3',
        143: 'imap', 443: 'https', 445: 'smb', 3306: 'mysql', 5432: 'postgres'
    }
    
    if not services:
        return 0.5
    
    matches = 0
    for port, service in services.items():
        if port in standard_ports:
            if service.lower() in standard_ports[port].lower():
                matches += 1
    
    consistency = matches / len(services)
    return consistency


# Public API
_ml_detector: Optional[MLAnomalyDetector] = None
_statistical_detector: Optional[StatisticalAnomalyDetector] = None


def get_ml_detector() -> MLAnomalyDetector:
    """Get or create the global ML anomaly detector"""
    global _ml_detector
    if _ml_detector is None:
        _ml_detector = MLAnomalyDetector()
    return _ml_detector


def get_statistical_detector() -> StatisticalAnomalyDetector:
    """Get or create the global statistical anomaly detector"""
    global _statistical_detector
    if _statistical_detector is None:
        _statistical_detector = StatisticalAnomalyDetector()
    return _statistical_detector
