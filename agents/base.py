import logging, json, uuid
from datetime import datetime
from typing import Any, Dict, Optional

log = logging.getLogger("AGENT-BASE")


class BaseAgent:
    def __init__(self, codename: str):
        self.codename = codename
        self.kernel: Optional[Any] = None
        self.priority = "normal"  # may be overwritten by config later

    # ------------------------------------------------------------------
    def should_activate(self, config: Dict[str, Any]) -> bool:
        """Return True if this agent should start."""
        return self.codename in config.get("agent_whitelist", [])

    def attach_kernel(self, kernel):
        """Wire the agent to the running kernel."""
        self.kernel = kernel
        # allow config to override default priority
        cfg = getattr(kernel, "config", {})
        self.priority = cfg.get("agent_priority", {}).get(self.codename, self.priority)
        self._broadcast("start")

    # ------------------------------------------------------------------
    def run(self):
        """Main agent loop stub; subclasses override."""
        log.debug("Agent %s started with priority %s", self.codename, self.priority)

    # ------------------------------------------------------------------
    def terminate(self):
        """Graceful shutdown."""
        self._broadcast("stop")
        log.debug("Agent %s terminated", self.codename)

    # ------------------------------------------------------------------
    def _broadcast(self, event: str):
        """Send agent lifecycle telemetry via SwarmMesh Redis if available."""
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        if redis:
            payload = {
                "event": event,
                "agent": self.codename,
                "priority": self.priority,
                "ts": datetime.utcnow().isoformat(),
            }
            redis.publish("agent_lifecycle", json.dumps(payload, separators=(",", ":")))