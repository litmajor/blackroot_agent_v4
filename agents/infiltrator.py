from agents.base import BaseAgent
import platform, socket, time, json, logging, uuid, hashlib
from datetime import datetime
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

log = logging.getLogger("INF-VENOM")

class InfiltratorAgent(BaseAgent):
    def __init__(self, agent_id: AgentID = AgentID(value="INF-VENOM")):
        # Use AgentID for identity
        self.agent_id: AgentID = agent_id
        # For BaseAgent compatibility, set codename to value
        super().__init__(str(self.agent_id))
        self.delay = 2 if getattr(self, "priority", None) == "low" else 0

    # ------------------------------------------------------------------
    def run(self, mission: Mission = Mission(name="default", objectives=[], parameters={})):
        super().run()
        if self.delay:
            log.info("Low-priority mode, delaying %ss...", self.delay)
            time.sleep(self.delay)

        fp = self._fingerprint()
        self._store(fp)
        self._exfil(fp)
        log.info("Host fingerprint complete: %s", fp.get("hostname", "?"))

        # Emit Event for fingerprint (sender_id)
        event = Event(event_type="host_fingerprint", payload=fp, sender_id=self.agent_id)
        self._emit_event(event)

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

    def _emit_event(self, event: Event):
        # Stub for event emission (to be integrated with event bus or kernel)
        log.info(f"Event emitted: {event.event_type} for agent {getattr(event, 'agent_id', None)}")