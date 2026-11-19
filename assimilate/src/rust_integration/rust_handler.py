"""
rust_handler.py — standalone Rust script handler for the assimilation pipeline.

Uses single-file `rustc` compilation (no Cargo project required), with graceful
fallback when the Rust toolchain is not installed on the host.
"""

import subprocess
import sys
import shutil
import tempfile
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RustScriptHandler:
    """
    Compile and execute a single-file Rust source blob via rustc.

    Workflow
    --------
    1. Check that `rustc` exists on PATH; return a status dict if absent.
    2. Write the source to a tmpfile `main.rs`.
    3. Compile with `rustc main.rs -o main[.exe]`.
    4. Execute the resulting binary and return stdout / returncode.

    This handler is wired into ``SOFTWARE_HANDLERS['rust']`` in
    ``assimilate/src/neuro_assimilator.py``.
    """

    def compile(self, rust_code: str) -> Dict[str, Any]:
        """
        Compile a Rust source string.

        Returns a dict with keys:
            status   - 'ok' | 'compile_error' | 'toolchain_unavailable' | 'timeout'
            bin_path - absolute path to compiled binary (only on 'ok')
            stderr   - compiler output on error
            tmpdir   - TemporaryDirectory context (caller must manage lifetime)
        """
        rustc = shutil.which("rustc")
        if not rustc:
            return {
                "status": "toolchain_unavailable",
                "reason": "rustc not found in PATH — install from https://rustup.rs",
            }

        tmpdir = tempfile.mkdtemp(prefix="blackroot_rust_")
        src_path = os.path.join(tmpdir, "main.rs")
        bin_name = "main.exe" if sys.platform.startswith("win") else "main"
        bin_path = os.path.join(tmpdir, bin_name)

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(rust_code)

        try:
            result = subprocess.run(
                [rustc, src_path, "-o", bin_path],
                capture_output=True, text=True, timeout=30, cwd=tmpdir,
            )
            if result.returncode != 0:
                logger.error(f"[RustScriptHandler] Compile failed:\n{result.stderr}")
                return {"status": "compile_error", "stderr": result.stderr, "tmpdir": tmpdir}
            return {"status": "ok", "bin_path": bin_path, "tmpdir": tmpdir}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "phase": "compile", "tmpdir": tmpdir}
        except FileNotFoundError as e:
            return {"status": "error", "reason": str(e), "tmpdir": tmpdir}

    def execute(self, bin_path: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute a compiled Rust binary and return its stdout/returncode."""
        try:
            result = subprocess.run(
                [bin_path], capture_output=True, text=True, timeout=timeout
            )
            logger.info(
                f"[RustScriptHandler] exited {result.returncode}: "
                f"{result.stdout.strip()[:200]}"
            )
            return {
                "status": "ok" if result.returncode == 0 else "runtime_error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "phase": "run"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def handle_rust_code(self, rust_code: str) -> Optional[str]:
        """
        High-level convenience: compile + execute a Rust source string.
        Returns stdout on success, None on any failure.
        """
        import shutil as _shutil

        compile_result = self.compile(rust_code)
        tmpdir = compile_result.get("tmpdir")
        try:
            if compile_result.get("status") != "ok":
                logger.warning(f"[RustScriptHandler] Compile failed: {compile_result}")
                return None
            run_result = self.execute(compile_result["bin_path"])
            if run_result.get("status") == "ok":
                return run_result.get("stdout")
            logger.warning(f"[RustScriptHandler] Run failed: {run_result}")
            return None
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                _shutil.rmtree(tmpdir, ignore_errors=True)