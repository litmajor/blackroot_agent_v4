# cloak_supervisor.py
import json
import logging
from typing import Dict, Any, Optional


class CloakSupervisor:
    """
    Stealth-level watchdog for BLACKROOT.Agent.
    """

    def __init__(self, kernel, stealth_threshold: float = 0.8):
        self.kernel = kernel
        self.logger = logging.getLogger("CloakSupervisor")
        self.stealth_threshold = stealth_threshold

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _compute_stealth_score(self, anomaly_stats: Optional[Dict[str, Any]] = None) -> float:
        """
        Placeholder for a real stealth-score calculation.
        Replace with ML, entropy, timing deltas, etc.
        """
        if anomaly_stats is None:
            return 1.0
        return float(anomaly_stats.get("score", 1.0))

    def _publish_redis(self, payload: Dict[str, Any]) -> None:
        """
        Safe publish wrapper; swallows Redis errors so a dead broker
        does not crash the agent.
        """
        try:
            self.kernel.swarm.redis.publish(self.kernel.swarm.channel, json.dumps(payload))
        except Exception as exc:
            self.logger.debug("Redis publish failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, anomaly_stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        Compute current stealth score, persist metrics, and return
        True if we are still under the configured threshold.
        """
        self.logger.info("Evaluating stealth level …")
        stealth_score = self._compute_stealth_score(anomaly_stats)

        try:
            # Count recon artifacts (lightweight heuristic)
            recon_count = sum(
                1
                for key in self.kernel.black_vault.list_artifacts()
                if key.startswith("recon_")
            )

            metrics = {"score": stealth_score, "recon_count": recon_count}

            # Persist in the vault
            self.kernel.black_vault.store(
                "stealth_metrics",
                json.dumps(metrics, separators=(",", ":")).encode()
            )

            # Broadcast to swarm mesh
            self._publish_redis({"type": "stealth_metrics", "data": metrics})

        except Exception as exc:
            self.logger.error("Failed to store stealth metrics: %s", exc)

        return stealth_score >= self.stealth_threshold