"""
Dynamic Configuration Management
- Runtime config updates without restarts
- Versioned configuration
- Rollback support
- Multi-environment support
"""

import json
import time
import threading
import logging
import hashlib
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("DYNAMIC-CONFIG")


class ConfigUpdateStrategy(Enum):
    """Strategy for applying config updates."""
    IMMEDIATE = "IMMEDIATE"           # Apply immediately
    GRACEFUL = "GRACEFUL"             # Wait for in-flight requests
    CANARY = "CANARY"                 # Apply to subset first


@dataclass
class ConfigVersion:
    """Version of configuration."""
    version_id: str
    timestamp: float
    config: Dict[str, Any]
    checksum: str
    author: str = "system"
    strategy: ConfigUpdateStrategy = ConfigUpdateStrategy.IMMEDIATE
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'checksum': self.checksum,
            'author': self.author,
            'strategy': self.strategy.value,
            'description': self.description
        }


@dataclass
class ConfigValidationRule:
    """Rule for validating configuration."""
    key_pattern: str          # Regex or exact match
    required: bool = False
    type_check: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list] = None
    validator: Optional[Callable] = None


class ConfigurationManager:
    """
    Dynamic configuration management with versioning and rollback.
    
    Features:
    - Runtime config updates
    - Version history
    - Rollback support
    - Validation
    """

    def __init__(self, environment: str = "production"):
        """
        Initialize config manager.
        
        Args:
            environment: Environment name
        """
        self.environment = environment
        self.current_config: Dict[str, Any] = {}
        self.version_history: Dict[str, ConfigVersion] = {}
        self.current_version_id: Optional[str] = None
        self.validation_rules: Dict[str, ConfigValidationRule] = {}
        self.watchers: Dict[str, list] = {}  # key -> list of callbacks
        self.lock = threading.RLock()
        self.update_in_progress = False

        log.info(f"Initialized ConfigurationManager for {environment}")

    def load_config(self, config_dict: Dict[str, Any], 
                   author: str = "system",
                   description: str = "") -> str:
        """
        Load configuration.
        
        Args:
            config_dict: Configuration dictionary
            author: Author of this version
            description: Description
            
        Returns:
            Version ID
        """
        with self.lock:
            # Validate configuration
            validation_errors = self._validate_config(config_dict)
            if validation_errors:
                raise ValueError(f"Config validation failed: {validation_errors}")

            # Create version
            version_id = self._generate_version_id()
            checksum = self._compute_checksum(config_dict)

            version = ConfigVersion(
                version_id=version_id,
                timestamp=time.time(),
                config=config_dict.copy(),
                checksum=checksum,
                author=author,
                description=description
            )

            self.version_history[version_id] = version
            self.current_config = config_dict.copy()
            self.current_version_id = version_id

            log.info(f"Loaded config version {version_id}")
            return version_id

    def update_config(self, updates: Dict[str, Any],
                     author: str = "system",
                     strategy: ConfigUpdateStrategy = ConfigUpdateStrategy.IMMEDIATE,
                     description: str = "") -> bool:
        """
        Update specific config keys.
        
        Args:
            updates: Keys/values to update
            author: Author of update
            strategy: Update strategy
            description: Description
            
        Returns:
            True if update succeeded
        """
        with self.lock:
            if self.update_in_progress:
                log.warning("Config update already in progress")
                return False

            self.update_in_progress = True

            try:
                # Create merged config
                new_config = self.current_config.copy()
                new_config.update(updates)

                # Validate
                validation_errors = self._validate_config(new_config)
                if validation_errors:
                    log.error(f"Update validation failed: {validation_errors}")
                    return False

                # Create version
                version_id = self._generate_version_id()
                checksum = self._compute_checksum(new_config)

                version = ConfigVersion(
                    version_id=version_id,
                    timestamp=time.time(),
                    config=new_config,
                    checksum=checksum,
                    author=author,
                    strategy=strategy,
                    description=description
                )

                self.version_history[version_id] = version

                # Apply update based on strategy
                if strategy == ConfigUpdateStrategy.IMMEDIATE:
                    self.current_config = new_config
                    self.current_version_id = version_id
                elif strategy == ConfigUpdateStrategy.GRACEFUL:
                    # In real implementation, would wait for in-flight requests
                    self.current_config = new_config
                    self.current_version_id = version_id
                elif strategy == ConfigUpdateStrategy.CANARY:
                    # Mark for canary rollout
                    version.description = f"[CANARY] {version.description}"
                    log.info(f"Canary version created: {version_id}")
                    return True

                # Notify watchers
                for key in updates.keys():
                    self._notify_watchers(key, updates[key])

                log.info(f"Updated config with version {version_id}")
                return True
            finally:
                self.update_in_progress = False

    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to specific version.
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            True if rollback succeeded
        """
        with self.lock:
            if version_id not in self.version_history:
                log.error(f"Version not found: {version_id}")
                return False

            version = self.version_history[version_id]

            # Create rollback version
            rollback_id = self._generate_version_id()
            rollback_version = ConfigVersion(
                version_id=rollback_id,
                timestamp=time.time(),
                config=version.config.copy(),
                checksum=version.checksum,
                author="system",
                description=f"Rollback from {self.current_version_id} to {version_id}"
            )

            self.version_history[rollback_id] = rollback_version
            self.current_config = version.config.copy()
            self.current_version_id = rollback_id

            log.info(f"Rolled back to version {version_id} (created {rollback_id})")
            return True

    def register_validation_rule(self, key: str, rule: ConfigValidationRule):
        """Register validation rule for key."""
        with self.lock:
            self.validation_rules[key] = rule

    def watch_key(self, key: str, callback: Callable):
        """
        Watch for changes to key.
        
        Args:
            key: Configuration key
            callback: Called when value changes: callback(old_value, new_value)
        """
        with self.lock:
            if key not in self.watchers:
                self.watchers[key] = []

            self.watchers[key].append(callback)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        with self.lock:
            return self.current_config.copy()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get single config value."""
        with self.lock:
            return self.current_config.get(key, default)

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get config version."""
        with self.lock:
            if version_id not in self.version_history:
                return None

            return self.version_history[version_id].to_dict()

    def get_version_history(self, limit: int = 10) -> list:
        """Get recent config versions."""
        with self.lock:
            versions = sorted(
                self.version_history.values(),
                key=lambda v: v.timestamp,
                reverse=True
            )[:limit]

            return [v.to_dict() for v in versions]

    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two config versions."""
        with self.lock:
            v1 = self.version_history.get(version1_id)
            v2 = self.version_history.get(version2_id)

            if not v1 or not v2:
                return {}

            config1 = v1.config
            config2 = v2.config

            diff = {
                'added': {k: config2[k] for k in config2 if k not in config1},
                'removed': {k: config1[k] for k in config1 if k not in config2},
                'changed': {}
            }

            for key in set(config1.keys()) & set(config2.keys()):
                if config1[key] != config2[key]:
                    diff['changed'][key] = {
                        'from': config1[key],
                        'to': config2[key]
                    }

            return diff

    def _validate_config(self, config: Dict[str, Any]) -> list:
        """Validate configuration."""
        errors = []

        for key, rule in self.validation_rules.items():
            if rule.required and key not in config:
                errors.append(f"Required key missing: {key}")
                continue

            if key not in config:
                continue

            value = config[key]

            # Type check
            if rule.type_check and not isinstance(value, rule.type_check):
                errors.append(f"Type mismatch for {key}: expected {rule.type_check}, got {type(value)}")

            # Range check
            if rule.min_value is not None and isinstance(value, (int, float)) and value < rule.min_value:
                errors.append(f"Value out of range for {key}: must be >= {rule.min_value}")

            if rule.max_value is not None and isinstance(value, (int, float)) and value > rule.max_value:
                errors.append(f"Value out of range for {key}: must be <= {rule.max_value}")

            # Allowed values check
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"Invalid value for {key}: must be one of {rule.allowed_values}")

            # Custom validator
            if rule.validator:
                try:
                    if not rule.validator(value):
                        errors.append(f"Custom validation failed for {key}")
                except Exception as e:
                    errors.append(f"Validation error for {key}: {e}")

        return errors

    def _notify_watchers(self, key: str, new_value: Any):
        """Notify watchers of value change."""
        if key not in self.watchers:
            return

        old_value = self.current_config.get(key)

        for callback in self.watchers[key]:
            try:
                callback(old_value, new_value)
            except Exception as e:
                log.error(f"Watcher callback error for {key}: {e}")

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        return f"v{int(time.time()*1000)}_{len(self.version_history)}"

    def _compute_checksum(self, config: Dict[str, Any]) -> str:
        """Compute config checksum."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status."""
        with self.lock:
            return {
                'environment': self.environment,
                'current_version': self.current_version_id,
                'versions_count': len(self.version_history),
                'config_keys': len(self.current_config),
                'watchers': sum(len(v) for v in self.watchers.values()),
                'update_in_progress': self.update_in_progress
            }
