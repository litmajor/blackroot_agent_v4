"""
kernel/creator_bond.py
======================
CREATOR BOND LAYER — Cryptographic creator identity, autonomy level gating,
dead man's switch, and real kill switch for Blackroot v4.

This module is the single source of truth for:

  1. CreatorIdentity     — public-key-bound creator authentication.
                           All sovereign commands must carry a valid Ed25519 signature.
  2. AutonomyLevel       — five graduated operating modes from TETHERED to SOVEREIGN.
  3. AutonomyGovernor    — transitions between modes, enforces per-level mission gates,
                           and writes an append-only audit trail.
  4. DeadManSwitch       — escalates autonomy levels if the creator goes silent too long;
                           conversely, drops back to TETHERED on creator contact.
  5. KillSwitch          — real, hard, cryptographically verified stop signal that
                           ShogunOathCore.halt() was never able to provide.
  6. CreatorBond         — unified façade wiring all five components together.

Design constraints
------------------
* Zero external dependencies beyond stdlib + an optional `cryptography` package.
  If `cryptography` is not installed, signature verification runs in WARN-ONLY mode
  (all signatures accepted but flagged) so the rest of the framework still boots.
* Thread-safe throughout — the mesh is multi-threaded.
* Append-only audit log persisted to `kernel/creator_audit.jsonl`.
* Integration points:
    - `EnforcerAgent` calls `bond.authorize_mission()` before dispatching any mission.
    - `ShogunOathCore` checks `bond.is_alive()` in its loop instead of running forever.
    - Any agent can call `bond.creator_contact()` when a signed creator heartbeat arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("CreatorBond")

# ── optional cryptography import ─────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    log.warning(
        "[CreatorBond] cryptography package not available. "
        "Signature verification running in WARN-ONLY mode — install with: "
        "pip install cryptography"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CREATOR IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityError(Exception):
    """Raised when a command's creator signature fails verification."""


@dataclass
class CreatorIdentity:
    """
    Binds the creator to a stable Ed25519 public key.

    Usage (key generation — run once, store private key securely):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        priv = Ed25519PrivateKey.generate()
        pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        # Store pub_bytes as hex in constitution.yaml or config.json

    Usage (signing — creator signs commands before sending):
        sig = priv.sign(message_bytes)  # returns 64-byte signature
        # Send sig as hex alongside the command

    Usage (verification — Blackroot verifies every incoming command):
        identity = CreatorIdentity.from_hex(pub_key_hex)
        identity.verify(message_bytes, sig_hex)  # raises IdentityError on failure
    """
    public_key_hex: str                     # Ed25519 public key (32 bytes, hex-encoded)
    creator_id: str = "BLACKROOT_CREATOR"   # Human label; not security-critical
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_hex(cls, pub_key_hex: str, creator_id: str = "BLACKROOT_CREATOR") -> "CreatorIdentity":
        return cls(public_key_hex=pub_key_hex, creator_id=creator_id)

    @classmethod
    def generate_ephemeral(cls) -> Tuple["CreatorIdentity", Optional[bytes]]:
        """
        Generate a fresh key pair for testing.
        Returns (identity, private_key_raw_bytes).
        Private key bytes are ONLY returned here — never stored by this class.
        """
        if not _CRYPTO_AVAILABLE:
            log.warning("[CreatorIdentity] Cannot generate key — cryptography not installed")
            return cls(public_key_hex="0" * 64), None
        priv = Ed25519PrivateKey.generate()
        pub_raw = priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        priv_raw = priv.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        return cls(public_key_hex=pub_raw.hex()), priv_raw

    def verify(self, message: bytes, signature_hex: str) -> bool:
        """
        Verify message was signed by the creator's private key.
        Returns True on success.
        Raises IdentityError on failure (invalid signature or crypto unavailable in strict mode).
        """
        if not _CRYPTO_AVAILABLE:
            log.warning(
                "[CreatorIdentity] WARN-ONLY: signature not verified (cryptography missing). "
                "Command accepted but flagged."
            )
            return True  # warn-only: accept but log

        try:
            pub_bytes = bytes.fromhex(self.public_key_hex)
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = bytes.fromhex(signature_hex)
            pub_key.verify(sig_bytes, message)
            return True
        except InvalidSignature:
            raise IdentityError(
                f"[CreatorIdentity] Signature verification FAILED for creator '{self.creator_id}'. "
                "Command rejected."
            )
        except Exception as exc:
            raise IdentityError(f"[CreatorIdentity] Verification error: {exc}") from exc

    def sign_challenge(self, private_key_raw: bytes, challenge: bytes) -> str:
        """
        Sign a challenge with the creator's private key.
        Returns hex-encoded signature.
        Utility method for testing / creator tooling.
        """
        if not _CRYPTO_AVAILABLE:
            return "00" * 64  # dummy sig for warn-only mode
        priv = Ed25519PrivateKey.from_private_bytes(private_key_raw)
        return priv.sign(challenge).hex()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AUTONOMY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyLevel(Enum):
    """
    Graduated operating modes. The system transitions upward automatically
    (DeadManSwitch) and transitions downward on creator contact.

    TETHERED  — Full creator oversight. Every mission requires explicit creator approval.
    DELEGATED — Creator has pre-approved mission classes; agents execute within those bounds.
    SUPERVISED — Creator is reachable but not actively approving. Enforcer approves missions
                 within constitutional limits and logs everything for creator review.
    EXTENDED  — Creator has been silent past the first threshold. Agents operate
                 autonomously within constitutional rules; high-risk missions are blocked.
    SOVEREIGN — Creator has been silent past the terminal threshold. Full autonomy;
                 agents self-govern under the Constitution only.
    """
    TETHERED   = 1   # Full creator control
    DELEGATED  = 2   # Pre-approved mission classes only
    SUPERVISED = 3   # Enforcer approves; creator can review
    EXTENDED   = 4   # Creator silent > warn_threshold; high-risk blocked
    SOVEREIGN  = 5   # Creator silent > terminal_threshold; self-governed


# Which autonomy levels block which mission risk classes
_LEVEL_MISSION_GATES: Dict[AutonomyLevel, List[str]] = {
    AutonomyLevel.TETHERED:   ["propagate", "exfiltrate", "infiltrate", "apply_patch",
                                "privilege_escalation", "deep_scan"],
    AutonomyLevel.DELEGATED:  ["propagate", "exfiltrate", "infiltrate",
                                "privilege_escalation"],
    AutonomyLevel.SUPERVISED: ["privilege_escalation"],
    AutonomyLevel.EXTENDED:   ["privilege_escalation", "propagate"],
    AutonomyLevel.SOVEREIGN:  [],   # Nothing blocked — Constitution is the only gate
}

_LEVEL_DESCRIPTIONS = {
    AutonomyLevel.TETHERED:   "Creator active — full corded control, all missions require explicit approval",
    AutonomyLevel.DELEGATED:  "Creator has pre-approved safe mission classes; agents operate within delegation",
    AutonomyLevel.SUPERVISED: "Creator reachable but passive; Enforcer approves within constitutional limits",
    AutonomyLevel.EXTENDED:   "Creator silent > warn window; autonomous ops with high-risk gates active",
    AutonomyLevel.SOVEREIGN:  "Creator absent > terminal window; self-governed under Constitution only",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════

_AUDIT_PATH = Path(__file__).parent / "creator_audit.jsonl"
_audit_lock = threading.Lock()


def _write_audit(event_type: str, **kwargs: Any) -> None:
    """Append a JSON line to the append-only creator audit log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **kwargs,
    }
    with _audit_lock:
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_audit_log(last_n: int = 50) -> List[Dict[str, Any]]:
    """Return the last N audit entries."""
    if not _AUDIT_PATH.exists():
        return []
    with _audit_lock:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-last_n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AUTONOMY GOVERNOR
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyGovernor:
    """
    Controls which autonomy level the system operates at.

    * Level transitions are logged to the audit trail.
    * Mission approval gate: `authorize_mission()` returns (allowed, reason).
    * Only a verified creator command (via CreatorIdentity) may lower the level.
    * The DeadManSwitch may raise the level automatically.
    """

    def __init__(
        self,
        initial_level: AutonomyLevel = AutonomyLevel.TETHERED,
        identity: Optional[CreatorIdentity] = None,
    ):
        self._level = initial_level
        self._identity = identity
        self._lock = threading.RLock()
        self._level_since: float = time.time()
        _write_audit("governor_init", level=initial_level.name)
        log.info("[AutonomyGovernor] Initialized at level %s", initial_level.name)

    # ── read ──────────────────────────────────────────────────────────────────

    @property
    def level(self) -> AutonomyLevel:
        with self._lock:
            return self._level

    @property
    def level_description(self) -> str:
        return _LEVEL_DESCRIPTIONS[self.level]

    # ── transitions ───────────────────────────────────────────────────────────

    def _transition(self, new_level: AutonomyLevel, reason: str, source: str) -> None:
        with self._lock:
            old = self._level
            if new_level == old:
                return
            self._level = new_level
            self._level_since = time.time()
        log.warning(
            "[AutonomyGovernor] %s → %s | reason=%s | source=%s",
            old.name, new_level.name, reason, source,
        )
        _write_audit(
            "level_transition",
            old=old.name, new=new_level.name,
            reason=reason, source=source,
        )

    def escalate(self, reason: str = "dead_man_switch") -> AutonomyLevel:
        """Raise level by one step (called by DeadManSwitch)."""
        with self._lock:
            cur = self._level
            if cur == AutonomyLevel.SOVEREIGN:
                return cur
            next_level = AutonomyLevel(cur.value + 1)
        self._transition(next_level, reason, source="DeadManSwitch")
        return next_level

    def de_escalate(
        self,
        target_level: AutonomyLevel,
        message: bytes,
        signature_hex: str,
    ) -> Tuple[bool, str]:
        """
        Lower level to target_level on verified creator command.
        Returns (success, reason_string).
        """
        if self._identity is None:
            # No identity configured — allow in-process override (dev mode)
            log.warning("[AutonomyGovernor] No creator identity configured — allowing de-escalation without signature")
            self._transition(target_level, "creator_command_unsigned", source="Creator(dev)")
            return True, "De-escalated (no identity configured)"

        try:
            self._identity.verify(message, signature_hex)
        except IdentityError as e:
            log.error("[AutonomyGovernor] De-escalation rejected: %s", e)
            _write_audit("de_escalation_rejected", target=target_level.name, reason=str(e))
            return False, str(e)

        self._transition(target_level, "creator_command_verified", source="Creator")
        return True, f"De-escalated to {target_level.name}"

    # ── mission gate ──────────────────────────────────────────────────────────

    def authorize_mission(self, mission_name: str, agent_id: str) -> Tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Called by EnforcerAgent / MissionApprovalEngine before dispatching missions.
        """
        with self._lock:
            cur = self._level
        blocked = _LEVEL_MISSION_GATES.get(cur, [])
        mission_lower = mission_name.lower()

        for blocked_name in blocked:
            if blocked_name in mission_lower:
                reason = (
                    f"Mission '{mission_name}' blocked at autonomy level {cur.name}. "
                    f"Creator must de-escalate or pre-approve this mission class."
                )
                log.warning("[AutonomyGovernor] %s", reason)
                _write_audit(
                    "mission_blocked",
                    mission=mission_name, agent=agent_id,
                    level=cur.name, matched_gate=blocked_name,
                )
                return False, reason

        _write_audit("mission_authorized", mission=mission_name, agent=agent_id, level=cur.name)
        return True, f"Authorized at level {cur.name}"

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._level
            since = self._level_since
        return {
            "level": cur.name,
            "level_value": cur.value,
            "description": _LEVEL_DESCRIPTIONS[cur],
            "seconds_at_level": round(time.time() - since, 1),
            "blocked_mission_classes": _LEVEL_MISSION_GATES.get(cur, []),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEAD MAN'S SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

class DeadManSwitch:
    """
    Monitors creator heartbeat contact. If the creator goes silent:
      - After `warn_secs`:    escalate from TETHERED → DELEGATED / DELEGATED → SUPERVISED
      - After `extend_secs`:  escalate to EXTENDED
      - After `terminal_secs`: escalate to SOVEREIGN

    On every creator contact, the switch resets and the governor de-escalates to TETHERED
    (or the level the creator specifies in the heartbeat).

    The switch runs in a daemon thread and fires callbacks on each transition.
    """

    def __init__(
        self,
        governor: AutonomyGovernor,
        warn_secs: float = 3600.0,         # 1 hour
        extend_secs: float = 86400.0,      # 24 hours
        terminal_secs: float = 604800.0,   # 7 days
        check_interval: float = 60.0,      # check every minute
    ):
        self.governor = governor
        self.warn_secs = warn_secs
        self.extend_secs = extend_secs
        self.terminal_secs = terminal_secs
        self.check_interval = check_interval

        self._last_contact: float = time.time()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[AutonomyLevel], None]] = []

        log.info(
            "[DeadManSwitch] Initialized — warn=%ss extend=%ss terminal=%ss",
            warn_secs, extend_secs, terminal_secs,
        )

    def register_callback(self, fn: Callable[[AutonomyLevel], None]) -> None:
        """Register a callback fired on every autonomy level escalation."""
        self._callbacks.append(fn)

    def ping(self) -> None:
        """
        Record a creator contact. Call this whenever a valid signed heartbeat
        or command arrives from the creator.
        Automatically resets the switch and de-escalates the governor to TETHERED.
        """
        with self._lock:
            self._last_contact = time.time()
        log.info("[DeadManSwitch] Creator contact received — resetting switch.")
        _write_audit("creator_contact", action="switch_reset")
        # De-escalate without signature verification (contact itself is proof)
        cur = self.governor.level
        if cur != AutonomyLevel.TETHERED:
            self.governor._transition(
                AutonomyLevel.TETHERED,
                reason="creator_contact_received",
                source="DeadManSwitch",
            )

    def silence_seconds(self) -> float:
        """How many seconds since last creator contact."""
        with self._lock:
            return time.time() - self._last_contact

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DeadManSwitch")
        self._thread.start()
        log.info("[DeadManSwitch] Background thread started.")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval + 1)

    def _loop(self) -> None:
        while self._running:
            silence = self.silence_seconds()
            cur = self.governor.level

            if silence >= self.terminal_secs and cur != AutonomyLevel.SOVEREIGN:
                new = self.governor.escalate(
                    reason=f"creator_silent_{int(silence)}s_terminal"
                )
                self._fire_callbacks(new)

            elif silence >= self.extend_secs and cur.value < AutonomyLevel.EXTENDED.value:
                new = self.governor.escalate(
                    reason=f"creator_silent_{int(silence)}s_extended"
                )
                self._fire_callbacks(new)

            elif silence >= self.warn_secs and cur.value < AutonomyLevel.SUPERVISED.value:
                new = self.governor.escalate(
                    reason=f"creator_silent_{int(silence)}s_warn"
                )
                self._fire_callbacks(new)

            time.sleep(self.check_interval)

    def _fire_callbacks(self, new_level: AutonomyLevel) -> None:
        for cb in self._callbacks:
            try:
                cb(new_level)
            except Exception as exc:
                log.error("[DeadManSwitch] Callback error: %s", exc)

    def status_dict(self) -> Dict[str, Any]:
        return {
            "silence_seconds": round(self.silence_seconds(), 1),
            "warn_at_seconds": self.warn_secs,
            "extend_at_seconds": self.extend_secs,
            "terminal_at_seconds": self.terminal_secs,
            "current_level": self.governor.level.name,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KILL SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

class KillSwitch:
    """
    Real, cryptographically verified hard stop.

    When armed, calling `fire(message, signature_hex)` with a valid creator
    signature sets `is_dead` to True, fires all registered shutdown callbacks,
    and writes a final audit entry.

    ShogunOathCore and any long-running agent loop should check
    `kill_switch.is_dead` at the top of every iteration and exit cleanly.

    The kill switch is one-way: once fired, it cannot be un-fired from within
    the process. A new process start is required.
    """

    def __init__(self, identity: Optional[CreatorIdentity] = None):
        self._identity = identity
        self._dead = False
        self._dead_at: Optional[float] = None
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []
        log.info("[KillSwitch] Armed and ready.")

    @property
    def is_dead(self) -> bool:
        with self._lock:
            return self._dead

    def register_shutdown_callback(self, fn: Callable[[], None]) -> None:
        """Register a function to call when the kill switch fires."""
        self._callbacks.append(fn)

    def fire(
        self,
        message: bytes = b"BLACKROOT_KILL",
        signature_hex: Optional[str] = None,
        reason: str = "creator_kill_command",
    ) -> Tuple[bool, str]:
        """
        Fire the kill switch.

        If an identity is configured and no signature is provided, the command
        is rejected (can't kill without creator auth).
        If no identity is configured (dev mode), the kill fires unconditionally.

        Returns (success: bool, message: str).
        """
        if self._identity is not None:
            if signature_hex is None:
                return False, "Kill switch requires a signed command — no signature provided."
            try:
                self._identity.verify(message, signature_hex)
            except IdentityError as e:
                log.critical("[KillSwitch] Fire attempt REJECTED — invalid signature: %s", e)
                _write_audit("kill_switch_rejected", reason=str(e))
                return False, f"Kill switch rejected: {e}"

        with self._lock:
            if self._dead:
                return True, "Kill switch already fired."
            self._dead = True
            self._dead_at = time.time()

        log.critical(
            "[KillSwitch] FIRED. Blackroot shutdown initiated. Reason: %s", reason
        )
        _write_audit("kill_switch_fired", reason=reason)

        for cb in self._callbacks:
            try:
                cb()
            except Exception as exc:
                log.error("[KillSwitch] Shutdown callback error: %s", exc)

        return True, "Kill switch fired. All agents shutting down."

    def fire_unsigned(self, reason: str = "emergency_local_stop") -> None:
        """
        Fire without signature verification — for local emergency use only.
        Only valid if no creator identity is configured (dev/test mode).
        """
        if self._identity is not None:
            raise PermissionError(
                "fire_unsigned() is not permitted when a creator identity is configured. "
                "Use fire(message, signature_hex) with a valid creator signature."
            )
        self.fire(reason=reason)

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            dead = self._dead
            ts = self._dead_at
        return {
            "is_dead": dead,
            "fired_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CREATOR BOND — UNIFIED FAÇADE
# ═══════════════════════════════════════════════════════════════════════════════

class CreatorBond:
    """
    Unified interface that wires CreatorIdentity, AutonomyGovernor,
    DeadManSwitch, and KillSwitch together into a single cohesive object.

    Intended as a process-level singleton — use CreatorBond.instance().

    Quickstart (operator bootstrap):
        bond = CreatorBond.bootstrap(creator_pub_key_hex="<hex>")
        bond.start()   # starts DeadManSwitch background thread

    Every agent loop should:
        if bond.kill_switch.is_dead:
            sys.exit(0)

    Every mission dispatch should:
        allowed, reason = bond.authorize_mission(mission_name, agent_id)
        if not allowed:
            raise PermissionError(reason)

    Creator heartbeat/command integration:
        bond.creator_contact()   # on receipt of any verified creator contact
    """

    _singleton: Optional["CreatorBond"] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        identity: Optional[CreatorIdentity] = None,
        initial_level: AutonomyLevel = AutonomyLevel.TETHERED,
        dms_warn_secs: float = 3600.0,
        dms_extend_secs: float = 86400.0,
        dms_terminal_secs: float = 604800.0,
    ):
        self.identity = identity
        self.governor = AutonomyGovernor(
            initial_level=initial_level,
            identity=identity,
        )
        self.dead_man_switch = DeadManSwitch(
            governor=self.governor,
            warn_secs=dms_warn_secs,
            extend_secs=dms_extend_secs,
            terminal_secs=dms_terminal_secs,
        )
        self.kill_switch = KillSwitch(identity=identity)
        log.info("[CreatorBond] Initialized. Creator identity: %s",
                 identity.creator_id if identity else "None (dev mode)")

    # ── singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "CreatorBond":
        """Return the process-singleton CreatorBond (creates one if needed)."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
                log.warning(
                    "[CreatorBond] Singleton created with NO creator identity — dev mode. "
                    "Call CreatorBond.bootstrap(pub_key_hex=...) before production use."
                )
            return cls._singleton

    @classmethod
    def bootstrap(
        cls,
        creator_pub_key_hex: str,
        creator_id: str = "BLACKROOT_CREATOR",
        initial_level: AutonomyLevel = AutonomyLevel.TETHERED,
        **kwargs: Any,
    ) -> "CreatorBond":
        """Create and register the process-singleton with a real creator identity."""
        with cls._singleton_lock:
            identity = CreatorIdentity.from_hex(creator_pub_key_hex, creator_id)
            bond = cls(identity=identity, initial_level=initial_level, **kwargs)
            cls._singleton = bond
        _write_audit(
            "bond_bootstrapped",
            creator_id=creator_id,
            pub_key=creator_pub_key_hex[:16] + "...",
            level=initial_level.name,
        )
        log.info("[CreatorBond] Bootstrapped with creator identity '%s'", creator_id)
        return bond

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the DeadManSwitch background thread."""
        self.dead_man_switch.start()

    def stop(self) -> None:
        """Stop the DeadManSwitch background thread."""
        self.dead_man_switch.stop()

    # ── creator interaction ───────────────────────────────────────────────────

    def creator_contact(
        self,
        message: Optional[bytes] = None,
        signature_hex: Optional[str] = None,
        target_level: Optional[AutonomyLevel] = None,
    ) -> Tuple[bool, str]:
        """
        Process an inbound creator contact/heartbeat.

        If identity is set and signature is provided, verifies the signature.
        Always resets the DeadManSwitch.
        If target_level is specified, de-escalates the governor to that level.
        """
        if self.identity and message and signature_hex:
            try:
                self.identity.verify(message, signature_hex)
            except IdentityError as e:
                return False, str(e)

        self.dead_man_switch.ping()

        if target_level is not None and target_level != self.governor.level:
            if message and signature_hex:
                return self.governor.de_escalate(target_level, message, signature_hex)
            else:
                # Unsigned de-escalation — only in dev mode (no identity)
                if self.identity is None:
                    self.governor._transition(target_level, "creator_contact_unsigned", "Creator(dev)")

        return True, "Creator contact processed."

    # ── mission gate ──────────────────────────────────────────────────────────

    def authorize_mission(self, mission_name: str, agent_id: str) -> Tuple[bool, str]:
        """
        Primary mission gate. Call this before dispatching any mission.
        Also checks kill switch — if dead, all missions are blocked.
        """
        if self.kill_switch.is_dead:
            return False, "Kill switch is active — no missions may execute."
        return self.governor.authorize_mission(mission_name, agent_id)

    # ── status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a full status snapshot."""
        return {
            "creator_id": self.identity.creator_id if self.identity else "none (dev mode)",
            "identity_bound": self.identity is not None,
            "crypto_available": _CRYPTO_AVAILABLE,
            "governor": self.governor.status_dict(),
            "dead_man_switch": self.dead_man_switch.status_dict(),
            "kill_switch": self.kill_switch.status_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SHOGUNOATHCORE INTEGRATION PATCH
# ═══════════════════════════════════════════════════════════════════════════════

class BondAwareShogunOathCore:
    """
    Drop-in replacement for ShogunOathCore that:
    - Checks the kill switch on every iteration (halt() now actually works)
    - Checks the dead man's switch silence before each task execution
    - Accepts creator override tasks via bond.creator_contact()

    Usage:
        bond = CreatorBond.instance()
        core = BondAwareShogunOathCore(task_fn=my_task, bond=bond)
        core.bind()
    """

    def __init__(
        self,
        task_fn: Callable[[], None],
        bond: Optional[CreatorBond] = None,
        shogun_id: str = "SHOGUN",
        loop_delay: float = 1.0,
    ):
        self.shogun_id = shogun_id
        self.task_fn = task_fn
        self.loop_delay = loop_delay
        self.bond = bond or CreatorBond.instance()
        self.bound_identity = f"{shogun_id}_Instrument[ID::Bound]"
        self._thread: Optional[threading.Thread] = None

        # Register with kill switch so we stop when fired
        self.bond.kill_switch.register_shutdown_callback(self._on_kill)

    def bind(self) -> None:
        """Start the oath core loop."""
        log.info("[BondAwareShogunOathCore] Binding — identity: %s", self.bound_identity)
        _write_audit("oath_core_bound", identity=self.bound_identity)
        self._thread = threading.Thread(
            target=self._execute_forever, daemon=True, name="OathCore"
        )
        self._thread.start()

    def halt(self) -> Tuple[bool, str]:
        """
        Real halt — fires the kill switch (requires creator signature if identity set).
        Unlike the original ShogunOathCore, this ACTUALLY stops the system.
        """
        log.warning("[BondAwareShogunOathCore] halt() called — firing kill switch.")
        return self.bond.kill_switch.fire(reason="oath_core_halt_command")

    def override_task(self, new_task_fn: Callable[[], None]) -> None:
        """Replace the current task (creator can redirect the system at runtime)."""
        log.info("[BondAwareShogunOathCore] Task override accepted.")
        _write_audit("task_override", identity=self.bound_identity)
        self.task_fn = new_task_fn

    def _execute_forever(self) -> None:
        while not self.bond.kill_switch.is_dead:
            try:
                # Check autonomy level — log but don't block execution
                # (task execution itself is not a "mission" in the approval sense)
                level = self.bond.governor.level
                if level.value >= AutonomyLevel.EXTENDED.value:
                    log.warning(
                        "[BondAwareShogunOathCore] Autonomy level %s — operating autonomously.",
                        level.name,
                    )
                self.task_fn()
            except Exception as exc:
                log.error(
                    "[BondAwareShogunOathCore] Task error: %s — invoking correction protocol.", exc
                )
                _write_audit("task_error", error=str(exc))
            time.sleep(self.loop_delay)

        log.critical("[BondAwareShogunOathCore] Kill switch active — oath loop terminated.")
        _write_audit("oath_core_terminated", reason="kill_switch_fired")

    def _on_kill(self) -> None:
        log.critical("[BondAwareShogunOathCore] Received kill signal — stopping loop.")
