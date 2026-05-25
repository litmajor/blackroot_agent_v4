"""
PHASE 3.3: DAEMON REGISTRY & ORCHESTRATION

Centralized management of all TaskRunner daemons across Blackroot modules.

Features:
- Unified view of all running daemons
- Dynamic runner lifecycle management
- Cross-module error propagation
- Statistics aggregation
- Graceful error recovery

Architecture:
  DaemonRegistry (Singleton)
  ├─ Registered Daemons (Modules)
  │  ├─ EnforcerAgent (1 daemon)
  │  ├─ SwarmMesh (6 daemons)
  │  ├─ TargetingEngine (3 daemons)
  │  └─ ... (Future modules)
  │
  ├─ Error Propagation Network
  │  ├─ Dependency graph
  │  ├─ Cascade detection
  │  └─ Recovery strategies
  │
  └─ Monitoring Interface
     ├─ Global statistics
     ├─ Health checks
     └─ Alerting
"""

import logging
import threading
from typing import Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from core.task_runner import TaskRunner

log = logging.getLogger("DAEMON-REGISTRY")


class DaemonHealth(Enum):
    """Health status of individual daemon."""
    HEALTHY = "HEALTHY"           # Running normally, no errors
    DEGRADED = "DEGRADED"         # Running but with backoff
    CRITICAL = "CRITICAL"         # High error rate
    STOPPED = "STOPPED"           # Not running
    FAILED = "FAILED"             # Initialization failed


class CascadeStrategy(Enum):
    """How to handle cross-daemon errors."""
    NONE = "NONE"                 # No propagation
    REDUCE_INTENSITY = "REDUCE_INTENSITY"  # Reduce work volume
    THROTTLE = "THROTTLE"         # Increase backoff
    ISOLATE = "ISOLATE"           # Stop dependent daemons
    ALERT = "ALERT"               # Log warning only


@dataclass
class DaemonMetrics:
    """Per-daemon statistics."""
    daemon_name: str
    module_name: str
    interval: float              # Default interval in seconds
    current_interval: float      # Current interval (with backoff)
    execution_count: int = 0     # Total executions
    error_count: int = 0         # Total errors
    last_execution: Optional[datetime] = None
    last_error: Optional[datetime] = None
    last_error_message: Optional[str] = None
    uptime_seconds: float = 0.0  # Time running without error
    health_status: DaemonHealth = DaemonHealth.HEALTHY
    error_rate: float = 0.0      # Errors per execution (%)


@dataclass
class ModuleRegistry:
    """Registry entry for a module with TaskRunner pool."""
    module_name: str             # e.g., "EnforcerAgent", "SwarmMesh"
    module_instance: object      # Reference to actual agent/engine
    task_runners: Dict[str, TaskRunner] = field(default_factory=dict)  # Runners by name
    shutdown_callback: Optional[Callable] = None  # Custom shutdown handler
    metrics: Dict[str, DaemonMetrics] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # Module names this depends on


class DaemonRegistry:
    """
    Centralized daemon registry and orchestrator.
    
    Singleton pattern - use DaemonRegistry.instance() to access.
    """
    
    _instance: Optional['DaemonRegistry'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize registry."""
        self.modules: Dict[str, ModuleRegistry] = {}  # Modules by name
        self.error_callbacks: Dict[str, List[Callable]] = {}  # Error handlers
        self.cascade_rules: Dict[Tuple[str, str], CascadeStrategy] = {}  # (source, target) -> strategy
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        log.info("DaemonRegistry initialized")
    
    @classmethod
    def instance(cls) -> 'DaemonRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    # =====================
    # Registration API
    # =====================
    
    def register_module(
        self,
        module_name: str,
        module_instance: object,
        task_runners: Dict[str, TaskRunner],
        dependencies: Optional[Set[str]] = None,
        shutdown_callback: Optional[Callable] = None
    ) -> None:
        """
        Register a module with its TaskRunner pool.
        
        Args:
            module_name: Unique module identifier
            module_instance: Reference to the agent/engine
            task_runners: Dict mapping daemon name to TaskRunner
            dependencies: Set of module names this depends on
            shutdown_callback: Callable for custom shutdown
        """
        with self._lock:
            if module_name in self.modules:
                log.warning(f"Module '{module_name}' already registered; overwriting")
            
            module_reg = ModuleRegistry(
                module_name=module_name,
                module_instance=module_instance,
                task_runners=task_runners.copy(),
                shutdown_callback=shutdown_callback,
                dependencies=dependencies or set()
            )
            
            # Initialize metrics for each daemon
            for daemon_name, runner in task_runners.items():
                metrics = DaemonMetrics(
                    daemon_name=daemon_name,
                    module_name=module_name,
                    interval=runner._default_interval,
                    current_interval=runner._default_interval
                )
                module_reg.metrics[daemon_name] = metrics
            
            self.modules[module_name] = module_reg
            log.info(f"Registered module '{module_name}' with {len(task_runners)} daemons")
    
    def unregister_module(self, module_name: str) -> bool:
        """
        Unregister and stop all daemons in a module.
        
        Args:
            module_name: Module to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if module_name not in self.modules:
                log.warning(f"Module '{module_name}' not found in registry")
                return False
            
            module_reg = self.modules[module_name]
            
            # Call shutdown callback if provided
            if module_reg.shutdown_callback:
                try:
                    module_reg.shutdown_callback()
                except Exception as e:
                    log.error(f"Error in shutdown callback for '{module_name}': {e}")
            
            # Stop all runners
            for daemon_name, runner in module_reg.task_runners.items():
                try:
                    runner.stop(timeout=5.0)
                    log.debug(f"Stopped {module_name}/{daemon_name}")
                except Exception as e:
                    log.warning(f"Error stopping {module_name}/{daemon_name}: {e}")
            
            del self.modules[module_name]
            log.info(f"Unregistered module '{module_name}'")
            return True
    
    # =====================
    # Query API
    # =====================
    
    def get_all_modules(self) -> List[str]:
        """Get list of all registered module names."""
        with self._lock:
            return list(self.modules.keys())
    
    def get_all_daemons(self) -> Dict[str, List[str]]:
        """Get all daemons grouped by module.
        
        Returns:
            Dict mapping module name to list of daemon names
        """
        with self._lock:
            return {
                module_name: list(module_reg.task_runners.keys())
                for module_name, module_reg in self.modules.items()
            }
    
    def get_daemon_runner(self, module_name: str, daemon_name: str) -> Optional[TaskRunner]:
        """Get a specific TaskRunner instance.
        
        Args:
            module_name: Module containing daemon
            daemon_name: Name of daemon
            
        Returns:
            TaskRunner instance or None if not found
        """
        with self._lock:
            if module_name not in self.modules:
                return None
            module_reg = self.modules[module_name]
            return module_reg.task_runners.get(daemon_name)
    
    def get_daemon_metrics(self, module_name: str, daemon_name: str) -> Optional[DaemonMetrics]:
        """Get metrics for a specific daemon.
        
        Args:
            module_name: Module containing daemon
            daemon_name: Name of daemon
            
        Returns:
            DaemonMetrics or None if not found
        """
        with self._lock:
            if module_name not in self.modules:
                return None
            module_reg = self.modules[module_name]
            return module_reg.metrics.get(daemon_name)
    
    def get_module_metrics(self, module_name: str) -> Optional[Dict[str, DaemonMetrics]]:
        """Get all metrics for a module.
        
        Args:
            module_name: Module name
            
        Returns:
            Dict mapping daemon name to metrics, or None if module not found
        """
        with self._lock:
            if module_name not in self.modules:
                return None
            return self.modules[module_name].metrics.copy()
    
    def get_global_statistics(self) -> Dict:
        """Get aggregate statistics across all modules/daemons.
        
        Returns:
            Dict with global metrics
        """
        with self._lock:
            total_modules = len(self.modules)
            total_daemons = sum(len(m.task_runners) for m in self.modules.values())
            total_executions = sum(
                sum(m.execution_count for m in module_reg.metrics.values())
                for module_reg in self.modules.values()
            )
            total_errors = sum(
                sum(m.error_count for m in module_reg.metrics.values())
                for module_reg in self.modules.values()
            )
            
            healthy = sum(
                sum(1 for m in module_reg.metrics.values() if m.health_status == DaemonHealth.HEALTHY)
                for module_reg in self.modules.values()
            )
            degraded = sum(
                sum(1 for m in module_reg.metrics.values() if m.health_status == DaemonHealth.DEGRADED)
                for module_reg in self.modules.values()
            )
            critical = sum(
                sum(1 for m in module_reg.metrics.values() if m.health_status == DaemonHealth.CRITICAL)
                for module_reg in self.modules.values()
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_modules": total_modules,
                "total_daemons": total_daemons,
                "total_executions": total_executions,
                "total_errors": total_errors,
                "global_error_rate": (total_errors / total_executions * 100) if total_executions > 0 else 0,
                "daemon_health": {
                    "healthy": healthy,
                    "degraded": degraded,
                    "critical": critical
                }
            }
    
    # =====================
    # Lifecycle Management
    # =====================
    
    def start_daemon(self, module_name: str, daemon_name: str, run_immediately: bool = False) -> bool:
        """Start a specific daemon.
        
        Args:
            module_name: Module containing daemon
            daemon_name: Name of daemon
            run_immediately: If True, run task once before main loop
            
        Returns:
            True if started, False if not found
        """
        runner = self.get_daemon_runner(module_name, daemon_name)
        if not runner:
            log.warning(f"Daemon not found: {module_name}/{daemon_name}")
            return False
        
        try:
            runner.start(run_immediately=run_immediately)
            log.info(f"Started daemon: {module_name}/{daemon_name}")
            return True
        except Exception as e:
            log.error(f"Error starting {module_name}/{daemon_name}: {e}")
            return False
    
    def stop_daemon(self, module_name: str, daemon_name: str, timeout: float = 5.0) -> bool:
        """Stop a specific daemon.
        
        Args:
            module_name: Module containing daemon
            daemon_name: Name of daemon
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if stopped, False if not found
        """
        runner = self.get_daemon_runner(module_name, daemon_name)
        if not runner:
            log.warning(f"Daemon not found: {module_name}/{daemon_name}")
            return False
        
        try:
            runner.stop(timeout=timeout)
            log.info(f"Stopped daemon: {module_name}/{daemon_name}")
            return True
        except Exception as e:
            log.error(f"Error stopping {module_name}/{daemon_name}: {e}")
            return False
    
    def restart_daemon(self, module_name: str, daemon_name: str) -> bool:
        """Restart a specific daemon.
        
        Args:
            module_name: Module containing daemon
            daemon_name: Name of daemon
            
        Returns:
            True if restarted, False if not found
        """
        if not self.stop_daemon(module_name, daemon_name):
            return False
        
        import time
        time.sleep(0.5)  # Brief pause
        
        return self.start_daemon(module_name, daemon_name)
    
    # =====================
    # Error Propagation
    # =====================
    
    def register_cascade_rule(
        self,
        source_module: str,
        target_module: str,
        strategy: CascadeStrategy
    ) -> None:
        """
        Register a cascade rule: when source module has errors, apply strategy to target.
        
        Args:
            source_module: Module whose errors trigger cascade
            target_module: Module affected by cascade
            strategy: How to respond to source errors
        """
        with self._lock:
            self.cascade_rules[(source_module, target_module)] = strategy
            log.debug(f"Registered cascade: {source_module} -> {target_module} via {strategy.value}")
    
    def register_error_callback(
        self,
        module_name: str,
        callback: Callable[[str, str, str], None]
    ) -> None:
        """
        Register callback for daemon errors.
        
        Args:
            module_name: Module to monitor
            callback: Function(module_name, daemon_name, error_msg) called on error
        """
        if module_name not in self.error_callbacks:
            self.error_callbacks[module_name] = []
        self.error_callbacks[module_name].append(callback)
        log.debug(f"Registered error callback for module '{module_name}'")
    
    def _trigger_cascade(self, source_module: str) -> None:
        """
        When a module experiences errors, apply cascade rules to dependent modules.
        
        Args:
            source_module: Module with errors
        """
        with self._lock:
            for (src, tgt), strategy in self.cascade_rules.items():
                if src != source_module:
                    continue
                
                if tgt not in self.modules:
                    continue
                
                log.warning(f"Cascade triggered: {source_module} -> {tgt} ({strategy.value})")
                
                # Apply strategy
                if strategy == CascadeStrategy.REDUCE_INTENSITY:
                    self._reduce_module_intensity(tgt)
                elif strategy == CascadeStrategy.THROTTLE:
                    self._throttle_module(tgt)
                elif strategy == CascadeStrategy.ISOLATE:
                    self._isolate_module(tgt)
                elif strategy == CascadeStrategy.ALERT:
                    pass  # Just log (already done above)
    
    def _reduce_module_intensity(self, module_name: str) -> None:
        """Reduce work volume for a module (increase intervals by 1.5×)."""
        if module_name not in self.modules:
            return
        
        module_reg = self.modules[module_name]
        for daemon_name, runner in module_reg.task_runners.items():
            # Increase interval moderately
            new_interval = runner._default_interval * 1.5
            runner._default_interval = min(new_interval, 60.0)  # Cap at 60s
            log.debug(f"Reduced intensity: {module_name}/{daemon_name} -> {new_interval:.1f}s")
    
    def _throttle_module(self, module_name: str) -> None:
        """Throttle module by stopping all daemons temporarily."""
        if module_name not in self.modules:
            return
        
        module_reg = self.modules[module_name]
        for daemon_name, runner in module_reg.task_runners.items():
            try:
                runner.stop(timeout=5.0)
                log.warning(f"Throttled daemon: {module_name}/{daemon_name}")
            except Exception as e:
                log.error(f"Error throttling {module_name}/{daemon_name}: {e}")
    
    def _isolate_module(self, module_name: str) -> None:
        """Isolate module by stopping all daemons."""
        if module_name not in self.modules:
            return
        
        module_reg = self.modules[module_name]
        for daemon_name, runner in module_reg.task_runners.items():
            try:
                runner.stop(timeout=5.0)
                log.error(f"Isolated daemon: {module_name}/{daemon_name}")
            except Exception as e:
                log.error(f"Error isolating {module_name}/{daemon_name}: {e}")
    
    # =====================
    # Monitoring
    # =====================
    
    def update_daemon_metrics(
        self,
        module_name: str,
        daemon_name: str,
        execution_time: float,
        error_occurred: bool = False,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update metrics for a daemon (call after each execution).
        
        Args:
            module_name: Module name
            daemon_name: Daemon name
            execution_time: Time to execute task (seconds)
            error_occurred: Whether error happened
            error_message: Error message if error occurred
        """
        with self._lock:
            if module_name not in self.modules:
                return
            
            module_reg = self.modules[module_name]
            if daemon_name not in module_reg.metrics:
                return
            
            metrics = module_reg.metrics[daemon_name]
            metrics.execution_count += 1
            metrics.last_execution = datetime.now()
            
            if error_occurred:
                metrics.error_count += 1
                metrics.last_error = datetime.now()
                metrics.last_error_message = error_message
                metrics.uptime_seconds = 0.0
            else:
                metrics.uptime_seconds += execution_time
            
            # Calculate error rate
            if metrics.execution_count > 0:
                metrics.error_rate = (metrics.error_count / metrics.execution_count) * 100
            
            # Determine health status
            if metrics.error_rate > 50:
                metrics.health_status = DaemonHealth.CRITICAL
                self._trigger_cascade(module_name)
            elif metrics.error_rate > 20:
                metrics.health_status = DaemonHealth.DEGRADED
            else:
                metrics.health_status = DaemonHealth.HEALTHY
            
            # Trigger error callbacks if error occurred
            if error_occurred and module_name in self.error_callbacks:
                for callback in self.error_callbacks[module_name]:
                    try:
                        callback(module_name, daemon_name, error_message or "Unknown error")
                    except Exception as e:
                        log.error(f"Error in callback: {e}")
    
    def get_health_report(self) -> Dict:
        """
        Generate comprehensive health report.
        
        Returns:
            Dict with detailed health information
        """
        with self._lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "modules": {}
            }
            
            for module_name, module_reg in self.modules.items():
                module_report = {
                    "daemon_count": len(module_reg.task_runners),
                    "daemons": {}
                }
                
                for daemon_name, metrics in module_reg.metrics.items():
                    module_report["daemons"][daemon_name] = asdict(metrics)
                    # Convert datetime objects to strings for JSON serialization
                    if metrics.last_execution:
                        module_report["daemons"][daemon_name]["last_execution"] = metrics.last_execution.isoformat()
                    if metrics.last_error:
                        module_report["daemons"][daemon_name]["last_error"] = metrics.last_error.isoformat()
                    module_report["daemons"][daemon_name]["health_status"] = metrics.health_status.value
                
                report["modules"][module_name] = module_report
            
            return report
    
    def shutdown_all(self, timeout: float = 10.0) -> None:
        """
        Gracefully shutdown all registered daemons.
        
        Args:
            timeout: Total timeout for shutdown
        """
        with self._lock:
            modules_to_stop = list(self.modules.keys())
        
        log.info(f"Shutting down {len(modules_to_stop)} modules...")
        
        for module_name in modules_to_stop:
            self.unregister_module(module_name)
        
        log.info("All modules shutdown complete")


# Convenience function for accessing singleton
def get_daemon_registry() -> DaemonRegistry:
    """Get the singleton DaemonRegistry instance."""
    return DaemonRegistry.instance()
