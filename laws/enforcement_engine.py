# enforcement_engine.py
"""
Enforcement Engine (Benign Governance Skeleton)

- Loads a constitution manifest (YAML/JSON).
- Validates and indexes rules and agent roles.
- Periodically monitors agent state (simulated heartbeats / metrics).
- Produces recommended actions (warnings, suspend, require manual review).
- DOES NOT execute network-level or destructive operations.
- Provides optional callbacks that an operator may attach to perform real actions
  (those callbacks are called only after explicit operator wiring).
"""

from __future__ import annotations
import asyncio
import yaml
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
import time
import os

log = logging.getLogger("EnforcementEngine")
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Types & Data Holders
# -----------------------------

@dataclass
class Constitution:
    raw: Dict[str, Any]

    @staticmethod
    def from_file(path: str) -> "Constitution":
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                raw = yaml.safe_load(f)
            else:
                raw = json.load(f)
        return Constitution(raw=raw)

    def get_agent_definitions(self) -> Dict[str, Dict[str, Any]]:
        # Normalize manifest to a mapping of agent_name -> definition
        root = self.raw.get("blackroot_constitution") or self.raw
        agents = {}
        for a in root.get("agents", []):
            name = a.get("name") or a.get("role") or f"agent_{len(agents)}"
            agents[name] = a
        return agents

    def get_policies(self) -> Dict[str, Any]:
        return self.raw.get("policies", {}) or self.raw.get("blackroot_constitution", {}).get("policies", {})


@dataclass
class AgentState:
    agent_id: str
    role: str
    last_seen: float = field(default_factory=time.time)
    reputation: float = 1.0
    balance: float = 0.0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    agent_id: str
    action: str          # e.g., 'none', 'warn', 'suspend', 'flag'
    reason: str
    score: float = 0.0   # numeric severity score


# -----------------------------
# Enforcement Engine
# -----------------------------

class EnforcementEngine:
    def __init__(self,
                 manifest_path: str,
                 check_interval: float = 30.0):
        self.constitution = Constitution.from_file(manifest_path)
        self.policies = self.constitution.get_policies()
        self.agent_defs = self.constitution.get_agent_definitions()
        self.check_interval = check_interval

        # Active state
        self.agents: Dict[str, AgentState] = {}

        # Callbacks (operator provides real-world actions)
        self.on_warning: Optional[Callable[[Recommendation], None]] = None
        self.on_suspend: Optional[Callable[[Recommendation], None]] = None
        self.on_freeze_funds: Optional[Callable[[Recommendation], None]] = None
        self.on_review: Optional[Callable[[Recommendation], None]] = None

        log.info("EnforcementEngine initialized with manifest: %s", manifest_path)

    # -------------------------
    # Agent lifecycle APIs
    # -------------------------
    def register_agent(self, agent_id: str, role: str, metadata: Optional[Dict[str, Any]] = None):
        if agent_id in self.agents:
            log.info("Agent %s already registered, updating role/metadata", agent_id)
            st = self.agents[agent_id]
            st.role = role
            if metadata:
                st.metadata.update(metadata)
            st.last_seen = time.time()
            return st
        st = AgentState(agent_id=agent_id, role=role, metadata=metadata or {})
        self.agents[agent_id] = st
        log.info("Registered agent %s with role %s", agent_id, role)
        return st

    def heartbeat(self, agent_id: str, reputation: Optional[float] = None, balance: Optional[float] = None):
        st = self.agents.get(agent_id)
        if not st:
            # Unknown agent - register as generic
            st = self.register_agent(agent_id, role="unknown")
        st.last_seen = time.time()
        if reputation is not None:
            st.reputation = reputation
        if balance is not None:
            st.balance = balance
        log.debug("Heartbeat: %s rep=%.3f bal=%.3f", agent_id, st.reputation, st.balance)

    # -------------------------
    # Policy evaluation
    # -------------------------
    def evaluate_agent(self, st: AgentState) -> Recommendation:
        """
        Evaluate an agent against constitution/policies and return a recommendation.
        This function is intentionally non-destructive: it only returns a suggestion.
        """
        # Basic checks (these are examples; adapt policies in manifest to change behavior)
        now = time.time()
        age = now - st.last_seen

        # Default: no action
        action = "none"
        reason = "ok"
        score = 0.0

        # Policy: heartbeat timeout -> warn / flag
        heartbeat_seconds = float(self.policies.get("heartbeat_seconds", 120))
        if age > heartbeat_seconds:
            action = "warn"
            reason = f"stale_heartbeat (last_seen {int(age)}s ago)"
            score = min(1.0, age / (heartbeat_seconds * 4))
            return Recommendation(st.agent_id, action, reason, score)

        # Policy: reputation threshold
        min_rep = float(self.policies.get("min_reputation", 0.3))
        if st.reputation < min_rep:
            action = "flag"
            reason = f"low_reputation ({st.reputation:.2f} < {min_rep})"
            score = (min_rep - st.reputation) * 10
            return Recommendation(st.agent_id, action, reason, score)

        # Policy: negative balance -> require review / throttle
        if st.balance < 0.0:
            action = "flag"
            reason = f"negative_balance ({st.balance:.2f})"
            score = min(1.0, abs(st.balance) / 100.0)
            return Recommendation(st.agent_id, action, reason, score)

        # Role-specific constraints (from manifest)
        role_conf = self.agent_defs.get(st.role, {})
        constraints = role_conf.get("constraints", []) if isinstance(role_conf, dict) else []

        # Example: constraint strings might include "cannot_self_elevate" -> map to actions
        if "cannot_self_elevate" in constraints:
            # if metadata shows elevation attempt -> flag (metadata is provided by agent instrumentation)
            if st.metadata.get("last_elevation_attempt"):
                action = "flag"
                reason = "self_elevation_attempt"
                score = 0.7
                return Recommendation(st.agent_id, action, reason, score)

        # No violations -> none
        return Recommendation(st.agent_id, "none", "ok", 0.0)

    # -------------------------
    # Recommendation -> non-actionable enforcement
    # -------------------------
    async def handle_recommendation(self, rec: Recommendation):
        """
        Convert a recommendation to a logged outcome and optionally call operator callbacks.
        The engine itself NEVER performs destructive ops by default.
        """
        log.info("Recommendation for %s: %s (%s) score=%.3f", rec.agent_id, rec.action, rec.reason, rec.score)

        if rec.action == "warn" and self.on_warning:
            try:
                self.on_warning(rec)
            except Exception as e:
                log.exception("on_warning callback raised: %s", e)
        elif rec.action == "suspend" and self.on_suspend:
            try:
                self.on_suspend(rec)
            except Exception as e:
                log.exception("on_suspend callback raised: %s", e)
        elif rec.action == "flag" and self.on_review:
            try:
                self.on_review(rec)
            except Exception as e:
                log.exception("on_review callback raised: %s", e)
        # Freeze funds / other economic actions are only suggested and trigger on_freeze_funds if set
        elif rec.action == "freeze_funds" and self.on_freeze_funds:
            try:
                self.on_freeze_funds(rec)
            except Exception as e:
                log.exception("on_freeze_funds callback raised: %s", e)

    # -------------------------
    # Main loop: periodic checks
    # -------------------------
    async def run_monitor_loop(self, stop_after: Optional[float] = None):
        """
        Run the monitor loop. This is non-destructive: it only emits recommendations.
        If you want destructive behavior, attach explicit callbacks that do the real work.
        """
        start = time.time()
        log.info("Starting enforcement monitor loop (interval=%ss)", self.check_interval)
        try:
            while True:
                now = time.time()
                for st in list(self.agents.values()):
                    rec = self.evaluate_agent(st)
                    await self.handle_recommendation(rec)
                await asyncio.sleep(self.check_interval)
                if stop_after and (time.time() - start) > stop_after:
                    log.info("Stopping monitor loop after %.1fs", stop_after)
                    break
        except asyncio.CancelledError:
            log.info("Monitor loop cancelled")
        except Exception as e:
            log.exception("Monitor loop error: %s", e)

# -----------------------------
# Simple demo / usage example
# -----------------------------

async def demo_run(manifest_path: str):
    # Create engine
    engine = EnforcementEngine(manifest_path, check_interval=5.0)

    # Register a couple of simulated agents
    engine.register_agent("agent-alpha", role="Accountant Agent", metadata={"note": "keeper"})
    engine.register_agent("agent-omega", role="Regent Agent", metadata={"note": "regent fallback"})

    # Set simple callbacks that only log (operator may replace with manual actions)
    def on_warn(rec: Recommendation):
        log.warning("OPERATOR: WARNING for %s -> %s : %s", rec.agent_id, rec.action, rec.reason)

    def on_flag(rec: Recommendation):
        log.warning("OPERATOR: FLAG for %s -> require manual review: %s", rec.agent_id, rec.reason)

    engine.on_warning = on_warn
    engine.on_review = on_flag

    # Simulate heartbeats and a failing agent
    engine.heartbeat("agent-alpha", reputation=0.95, balance=100.0)
    engine.heartbeat("agent-omega", reputation=0.9, balance=50.0)

    # Simulate a poor agent that will be flagged
    engine.register_agent("agent-bad", role="unknown", metadata={"last_elevation_attempt": True})
    engine.heartbeat("agent-bad", reputation=0.1, balance=-5.0)

    # Run the monitor for 20 seconds
    await engine.run_monitor_loop(stop_after=20.0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "-m", default="constitution.yaml",
                        help="Path to constitution manifest (YAML or JSON)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        # write a tiny default manifest if missing
        sample = {
            "blackroot_constitution": {
                "version": "1.0.0",
                "agents": [
                    {"name": "Accountant Agent", "role": "Accountant Agent", "constraints": []},
                    {"name": "Regent Agent", "role": "Regent Agent", "constraints": ["cannot_self_elevate"]},
                ],
                "policies": {
                    "heartbeat_seconds": 10,
                    "min_reputation": 0.2
                }
            }
        }
        with open(args.manifest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(sample, fh)
        log.info("Wrote sample manifest to %s", args.manifest)

    asyncio.run(demo_run(args.manifest))
