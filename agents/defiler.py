from agents.base import BaseAgent
import os, shutil, tempfile, logging, stat, subprocess
from pathlib import Path

# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

log = logging.getLogger("DEF-GLOOM")


class DefilerAgent(BaseAgent):
    """
    DefilerAgent ("DEF-GLOOM")
    Role: Cleanup and stealth removal (logs, cache, temp artifacts, self-delete).
    """

    def __init__(self, agent_id: AgentID = AgentID(value="DEF-GLOOM")):
        self.agent_id: AgentID = agent_id
        super().__init__(str(self.agent_id))
        # Aggressive vs passive mode
        self.aggressive: bool = getattr(self, "priority", "normal") != "low"
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('50'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for DefilerAgent: {e}")

    def _deduct_cleanup_cost(self):
        """Deduct Root for cleanup operation (0.2 Root per action)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.2'), "Cleanup cost")

    def _reward_cleanup_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.4')
            self.accountant.treasury_balance -= Decimal('0.4')
            log.info(f"Rewarded 0.4 Root to DefilerAgent for successful cleanup.")

    def _burn_on_cleanup_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.3')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            log.info(f"Burned 0.3 Root from DefilerAgent for failed cleanup.")

    def should_activate(self, config: dict) -> bool:
        """Only activates if in whitelist."""
        return self.codename in config.get("agent_whitelist", [])

    # ------------------------------------------------------------------
    def run(self, mission: Mission = Mission(name="default", objectives=[], parameters={})):
        """Run mission logic: passive (light) or aggressive (full cleanup)."""
        super().run()

        if not self.aggressive:
            log.info("Passive mode: minimal cleanup.")
            self._deduct_cleanup_cost()
            try:
                event = Event(
                    event_type="defiler_passive",
                    payload={"agent": str(self.agent_id)},
                    sender_id=str(self.agent_id),
                )
                self._emit_event(event)
                self._reward_cleanup_success()
            except Exception as e:
                self._burn_on_cleanup_failure()
                log.warning(f"Passive cleanup failed: {e}")
            return

        log.info("Full-stealth cleanup initiated.")
        self._deduct_cleanup_cost()
        try:
            self._nuke_logs()
            self._emit_event(Event("logs_nuked", {}, str(self.agent_id)))
            self._reward_cleanup_success()
        except Exception as e:
            self._burn_on_cleanup_failure()
            log.warning(f"Log cleanup failed: {e}")

        self._deduct_cleanup_cost()
        try:
            self._nuke_cache()
            self._emit_event(Event("cache_nuked", {}, str(self.agent_id)))
            self._reward_cleanup_success()
        except Exception as e:
            self._burn_on_cleanup_failure()
            log.warning(f"Cache cleanup failed: {e}")

        self._deduct_cleanup_cost()
        try:
            self._nuke_temp()
            self._emit_event(Event("temp_nuked", {}, str(self.agent_id)))
            self._reward_cleanup_success()
        except Exception as e:
            self._burn_on_cleanup_failure()
            log.warning(f"Temp cleanup failed: {e}")

        self._deduct_cleanup_cost()
        try:
            self._shred_self()
            self._emit_event(Event("self_shredded", {}, str(self.agent_id)))
            self._reward_cleanup_success()
        except Exception as e:
            self._burn_on_cleanup_failure()
            log.warning(f"Self-shred failed: {e}")

    # ------------------------------------------------------------------
    def _emit_event(self, event: Event):
        """Stub for event emission (to integrate with event bus)."""
        log.info(f"Event emitted: {event.event_type} for agent {event.sender_id}")

    # ------------------------------------------------------------------
    def _secure_remove(self, path: Path):
        """Best-effort secure delete."""
        try:
            if path.is_file():
                subprocess.run(["shred", "-uzf", str(path)], check=False)
            elif path.is_dir():
                subprocess.run(["rm", "-rf", str(path)], check=False)
            else:
                path.unlink(missing_ok=True)
        except Exception as e:
            log.warning("Could not shred %s: %s", path, e)

    def _nuke_logs(self):
        targets = ["blackroot.log", "agent.log", "*.log"]
        for pattern in targets:
            for f in Path.cwd().glob(pattern):
                self._secure_remove(f)
        log.info("Log files cleared.")

    def _nuke_cache(self):
        dirs = [
            Path.home() / ".cache" / "blackroot",
            Path.home() / ".blackvault",
            Path(tempfile.gettempdir()) / "blackroot*",
        ]
        for d in dirs:
            for p in Path(d.parent).glob(d.name):
                self._secure_remove(p)
        log.info("Cache dirs removed.")

    def _nuke_temp(self):
        for d in Path(tempfile.gettempdir()).glob("br_*"):
            shutil.rmtree(d, ignore_errors=True)
        log.info("Temp artifacts removed.")

    def _shred_self(self):
        """Attempt to remove the agent script after run."""
        self_path = Path(__file__).resolve()
        try:
            os.chmod(self_path, stat.S_IWRITE | stat.S_IREAD)
            self._secure_remove(self_path)
            log.info("Agent binary self-deleted.")
        except Exception as e:
            log.warning("Self-delete failed: %s", e)
