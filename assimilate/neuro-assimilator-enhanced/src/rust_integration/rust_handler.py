import subprocess
import sys
from typing import Any
import os

class RustScriptHandler:
    def __init__(self):
        self.rust_project_path = os.path.abspath(os.path.dirname(__file__))

    def compile(self, rust_code: str) -> bool:
        main_rs_path = os.path.join(self.rust_project_path, "main.rs")
        with open(main_rs_path, "w") as f:
            f.write(rust_code)
        try:
            result = subprocess.run(["cargo", "build", "--release"], cwd=self.rust_project_path, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[RUST] Compilation failed: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"[RUST] Compilation error: {e}")
            return False

    def execute(self) -> Any:
        # Detect platform for correct binary extension
        bin_name = "rust_integration.exe" if sys.platform.startswith("win") else "rust_integration"
        binary_path = os.path.join(self.rust_project_path, "target", "release", bin_name)
        try:
            result = subprocess.run([binary_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[RUST] Execution failed: {result.stderr}")
                return None
            return result.stdout
        except Exception as e:
            print(f"[RUST] Execution error: {e}")
            return None

    def handle_rust_code(self, rust_code: str) -> Any:
        if self.compile(rust_code):
            return self.execute()
        return None