"""
PHASE 3.3: DAEMON REGISTRY INTEGRATION LAYER

Provides utilities to register existing Phase 3.0-3.2 modules with the daemon registry.

Supported Modules:
- EnforcerAgent (Phase 3.0): 1 daemon
- SwarmMesh (Phase 3.1): 6 daemons
- TargetingEngine (Phase 3.2): 3 daemons
"""

import logging
from typing import Optional
from core.daemon_registry import DaemonRegistry, CascadeStrategy

log = logging.getLogger("REGISTRY-INTEGRATION")


class RegistryIntegration:
    """Utilities for integrating Phase 3 modules with DaemonRegistry."""
    
    @staticmethod
    def register_enforcer_agent(enforcer_instance) -> bool:
        """
        Register EnforcerAgent with the daemon registry.
        
        Args:
            enforcer_instance: Instance of EnforcerAgent with task_runners pool
            
        Returns:
            True if registered successfully
        """
        try:
            if not hasattr(enforcer_instance, 'task_runners'):
                log.error("EnforcerAgent missing task_runners attribute")
                return False
            
            registry = DaemonRegistry.instance()
            registry.register_module(
                module_name="EnforcerAgent",
                module_instance=enforcer_instance,
                task_runners=enforcer_instance.task_runners,
                shutdown_callback=lambda: getattr(enforcer_instance, '_shutdown_task_runners', lambda: None)()
            )
            
            log.info("EnforcerAgent registered with DaemonRegistry (1 daemon)")
            return True
        except Exception as e:
            log.error(f"Failed to register EnforcerAgent: {e}")
            return False
    
    @staticmethod
    def register_swarm_mesh(swarm_mesh_instance) -> bool:
        """
        Register SwarmMesh with the daemon registry.
        
        Args:
            swarm_mesh_instance: Instance of SwarmMesh with task_runners pool
            
        Returns:
            True if registered successfully
        """
        try:
            if not hasattr(swarm_mesh_instance, 'task_runners'):
                log.error("SwarmMesh missing task_runners attribute")
                return False
            
            registry = DaemonRegistry.instance()
            registry.register_module(
                module_name="SwarmMesh",
                module_instance=swarm_mesh_instance,
                task_runners=swarm_mesh_instance.task_runners,
                shutdown_callback=lambda: getattr(swarm_mesh_instance, '_shutdown_task_runners', lambda: None)()
            )
            
            log.info(f"SwarmMesh registered with DaemonRegistry ({len(swarm_mesh_instance.task_runners)} daemons)")
            return True
        except Exception as e:
            log.error(f"Failed to register SwarmMesh: {e}")
            return False
    
    @staticmethod
    def register_targeting_engine(targeting_engine_instance) -> bool:
        """
        Register TargetingEngine with the daemon registry.
        
        Args:
            targeting_engine_instance: Instance of TargetingEngine with task_runners pool
            
        Returns:
            True if registered successfully
        """
        try:
            if not hasattr(targeting_engine_instance, 'task_runners'):
                log.error("TargetingEngine missing task_runners attribute")
                return False
            
            registry = DaemonRegistry.instance()
            registry.register_module(
                module_name="TargetingEngine",
                module_instance=targeting_engine_instance,
                task_runners=targeting_engine_instance.task_runners,
                shutdown_callback=lambda: getattr(targeting_engine_instance, '_shutdown_task_runners', lambda: None)(),
                dependencies={"SwarmMesh"}  # TargetingEngine depends on mesh topology
            )
            
            log.info(f"TargetingEngine registered with DaemonRegistry (3 daemons)")
            return True
        except Exception as e:
            log.error(f"Failed to register TargetingEngine: {e}")
            return False
    
    @staticmethod
    def setup_default_cascade_rules() -> None:
        """
        Setup recommended cascade rules between modules.
        
        Rules:
        - TargetingEngine errors reduce SwarmMesh intensity (don't propagate if mesh unstable)
        - SwarmMesh errors isolate TargetingEngine (can't target if mesh broken)
        - EnforcerAgent errors alert but don't cascade (critical but isolated)
        """
        registry = DaemonRegistry.instance()
        
        # TargetingEngine → SwarmMesh: Reduce intensity
        registry.register_cascade_rule(
            source_module="TargetingEngine",
            target_module="SwarmMesh",
            strategy=CascadeStrategy.REDUCE_INTENSITY
        )
        
        # SwarmMesh → TargetingEngine: Isolate
        registry.register_cascade_rule(
            source_module="SwarmMesh",
            target_module="TargetingEngine",
            strategy=CascadeStrategy.ISOLATE
        )
        
        # EnforcerAgent → SwarmMesh: Alert only
        registry.register_cascade_rule(
            source_module="EnforcerAgent",
            target_module="SwarmMesh",
            strategy=CascadeStrategy.ALERT
        )
        
        log.info("Default cascade rules configured")
    
    @staticmethod
    def setup_default_error_callbacks() -> None:
        """
        Setup recommended error callbacks for each module.
        
        Callbacks log critical daemon errors and prepare for recovery.
        """
        registry = DaemonRegistry.instance()
        
        def enforcer_error_handler(module_name: str, daemon_name: str, error_msg: str) -> None:
            log.error(f"⚠️  EnforcerAgent error: {daemon_name} - {error_msg}")
        
        def swarm_error_handler(module_name: str, daemon_name: str, error_msg: str) -> None:
            log.error(f"🌐 SwarmMesh error: {daemon_name} - {error_msg}")
        
        def targeting_error_handler(module_name: str, daemon_name: str, error_msg: str) -> None:
            log.error(f"🎯 TargetingEngine error: {daemon_name} - {error_msg}")
        
        registry.register_error_callback("EnforcerAgent", enforcer_error_handler)
        registry.register_error_callback("SwarmMesh", swarm_error_handler)
        registry.register_error_callback("TargetingEngine", targeting_error_handler)
        
        log.info("Default error callbacks configured")


def initialize_registry_from_kernel(kernel_instance) -> bool:
    """
    Initialize daemon registry from kernel instance.
    
    Attempts to register all Phase 3 modules found in kernel.agents.
    
    Args:
        kernel_instance: Kernel instance with agents dict
        
    Returns:
        True if at least one module registered
    """
    if not hasattr(kernel_instance, 'agents'):
        log.warning("Kernel missing agents attribute")
        return False
    
    agents = kernel_instance.agents
    registered_count = 0
    
    # Try to register EnforcerAgent
    if "ENFORCER" in agents:
        if RegistryIntegration.register_enforcer_agent(agents["ENFORCER"]):
            registered_count += 1
    
    # Try to register SwarmMesh
    if "SWARM-MESH" in agents:
        if RegistryIntegration.register_swarm_mesh(agents["SWARM-MESH"]):
            registered_count += 1
    
    # Try to register TargetingEngine
    if "TARGET-FORGE" in agents:
        if RegistryIntegration.register_targeting_engine(agents["TARGET-FORGE"]):
            registered_count += 1
    
    if registered_count > 0:
        log.info(f"Registered {registered_count} module(s) with DaemonRegistry")
        
        # Setup cascading rules and callbacks
        RegistryIntegration.setup_default_cascade_rules()
        RegistryIntegration.setup_default_error_callbacks()
        return True
    
    return False
