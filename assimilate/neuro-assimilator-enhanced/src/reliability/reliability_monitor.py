import os
import json
import time
import logging
import shutil
import tempfile
import subprocess
import threading
from typing import Any, Dict, Optional
from pathlib import Path
import sys

logger = logging.getLogger("ReliabilityMonitor")

class ReliabilityMonitor:
    """
    Production-grade reliability layer:
    - Crash isolation via subprocess sandbox
    - Atomic codex backup/rollback
    - Telemetry & rate-limiting
    """

    def __init__(
        self,
        log_dir: 'Optional[str]' = None,
        max_crash_rate: float = 5.0,
        backup_dir: 'Optional[str]' = None,
    ):
        self.log_dir = Path(log_dir or tempfile.gettempdir()) / "reliability_logs"
        self.backup_dir = Path(backup_dir or tempfile.gettempdir()) / "codex_backup"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.crash_logs_file = self.log_dir / "crashes.jsonl"
        self.telemetry_file = self.log_dir / "telemetry.jsonl"

        self._crash_times: list[float] = []  # sliding-window for rate-limit
        self._max_crash_rate = max_crash_rate  # crashes/second
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 1.  Crash Isolation – run payload in a disposable subprocess
    # ------------------------------------------------------------------ #
    def crash_isolation(
        self,
        payload_name: str,
        code: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """
        Executes `code` (python_script or compiled ELF) inside a *new* Python
        subprocess.  Stdout is returned; any non-zero exit becomes a crash log.
        """
        with self._lock:
            if self._rate_limit_exceeded():
                raise RuntimeError("Crash rate exceeded; payload blocked")

        # Write payload to temp file
        suffix = ".py" if code.get("type") == "python_script" else ".bin"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as f:
            f.write(code.get("source", b""))
            payload_path = f.name

        try:
            cmd = [sys.executable, "-u", payload_path]
            result = subprocess.run(
                cmd,
                input=json.dumps(context).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
            return json.loads(result.stdout or "{}")
        except Exception as e:
            self.log_crash(payload_name, e)
            raise
        finally:
            os.unlink(payload_path)

    def _rate_limit_exceeded(self) -> bool:
        now = time.time()
        self._crash_times = [t for t in self._crash_times if now - t < 60]
        return len(self._crash_times) > self._max_crash_rate

    # ------------------------------------------------------------------ #
    # 2.  Codex Backup & Rollback (atomic JSON write)
    # ------------------------------------------------------------------ #
    def backup_codex(self, codex: Dict[str, Any]) -> None:
        backup_file = self.backup_dir / f"codex_{int(time.time())}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(codex, f, separators=(",", ":"))
        # keep only last N backups
        self._prune_backups()

    def rollback_codex(self) -> Dict[str, Any]:
        backups = sorted(self.backup_dir.glob("codex_*.json"))
        if not backups:
            return {}
        latest = backups[-1]
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prune_backups(self, keep: int = 5):
        for old in sorted(self.backup_dir.glob("codex_*.json"))[:-keep]:
            old.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 3.  Telemetry & Logging (append-only JSONL)
    # ------------------------------------------------------------------ #
    def record_telemetry(self, data: Dict[str, Any]) -> None:
        with open(self.telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **data}) + "\n")

    def log_crash(self, payload_name: str, error: Exception) -> None:
        with open(self.crash_logs_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "payload": payload_name,
                        "error": str(error),
                    }
                )
                + "\n"
            )
        logger.error(f"[CRASH] {payload_name}: {error}")
        self._crash_times.append(time.time())

    # ------------------------------------------------------------------ #
    # 4.  Read-only helpers
    # ------------------------------------------------------------------ #
    def get_crash_logs(self) -> list[Dict[str, Any]]:
        if not self.crash_logs_file.exists():
            return []
        with open(self.crash_logs_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_telemetry_data(self) -> list[Dict[str, Any]]:
        if not self.telemetry_file.exists():
            return []
        with open(self.telemetry_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]