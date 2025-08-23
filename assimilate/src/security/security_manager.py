import os
import sys
import json
import logging
import hashlib
import time
import base64
import random
import ast
import re
import psutil
import signal
import tempfile
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import subprocess

class SecurityManager:
    def __init__(self, allowed_modules: Optional[List[str]] = None, public_key_path: Optional[str] = None):
        self.allowed_modules = set(allowed_modules or ['math', 'time', 'datetime', 'random'])
        self.resource_limits = {
            'cpu': 80,  # Maximum CPU usage percentage
            'memory': 512 * 1024 * 1024,  # Maximum memory usage in bytes (512 MB)
            'execution_time': 30  # Maximum execution time in seconds
        }
        self.public_key_path = public_key_path
        self.dangerous_functions = {
            'exec', 'eval', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'vars', 'locals', 'globals',
            'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
        }
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib',
            'requests', 'pickle', 'marshal', 'shelve', 'dbm'
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def static_analysis(self, code: str) -> bool:
        """Perform comprehensive static analysis on the code."""
        self.logger.info("[SECURITY] Performing static analysis...")
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for dangerous imports and function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_modules:
                            self.logger.warning(f"Dangerous module import detected: {alias.name}")
                            return False
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module in self.dangerous_modules:
                        self.logger.warning(f"Dangerous module import detected: {node.module}")
                        return False
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_functions:
                        self.logger.warning(f"Dangerous function call detected: {node.func.id}")
                        return False
            
            # Check for suspicious patterns using regex
            suspicious_patterns = [
                r'__.*__',  # Dunder methods
                r'\.system\(',  # System calls
                r'\.popen\(',  # Process opening
                r'\.exec\w*\(',  # Various exec methods
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    self.logger.warning(f"Suspicious pattern detected: {pattern}")
                    return False
            
            self.logger.info("[SECURITY] Static analysis passed")
            return True
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in code: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error during static analysis: {e}")
            return False

    def verify_signature(self, code: bytes, signature: str) -> bool:
        """Verify the cryptographic signature of the code."""
        if not self.public_key_path or not os.path.exists(self.public_key_path):
            self.logger.warning("[SECURITY] No public key provided, skipping signature verification")
            return True
        
        try:
            self.logger.info("[SECURITY] Verifying signature...")
            
            # Load public key
            with open(self.public_key_path, 'rb') as key_file:
                public_key = load_pem_public_key(key_file.read())
            
            # Decode the signature
            signature_bytes = base64.b64decode(signature)
            
            # Verify the signature (RSA only)
            from cryptography.hazmat.primitives.asymmetric import rsa
            if isinstance(public_key, rsa.RSAPublicKey):
                public_key.verify(
                    signature_bytes,
                    code,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            else:
                self.logger.error(f"Unsupported public key type for signature verification: {type(public_key)}")
                return False
            
            self.logger.info("[SECURITY] Signature verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False

    def enforce_resource_limits(self) -> None:
        """Enforce resource limits on the current process (POSIX only)."""
        self.logger.info("[SECURITY] Enforcing resource limits...")
        try:
            if os.name == "posix":
                # Set memory limit
                resource.setrlimit(
                    resource.RLIMIT_AS, 
                    (self.resource_limits['memory'], self.resource_limits['memory'])
                )
                # Set CPU time limit
                resource.setrlimit(
                    resource.RLIMIT_CPU, 
                    (self.resource_limits['execution_time'], self.resource_limits['execution_time'])
                )
                self.logger.info("[SECURITY] Resource limits enforced (POSIX)")
            else:
                self.logger.warning("[SECURITY] Resource limits not enforced: unsupported OS")
        except Exception as e:
            self.logger.error(f"Failed to enforce resource limits: {e}")
            # Do not raise on Windows

    def monitor_resource_usage(self) -> Dict[str, float]:
        """Monitor current resource usage."""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }

    def hash_code(self, code: str) -> str:
        """Generate a SHA-256 hash of the code for integrity checks."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()

    def create_sandbox_environment(self) -> Dict[str, Any]:
        """Create a restricted environment for code execution."""
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'hex', 'int', 'len', 'list', 'map', 'max',
            'min', 'oct', 'ord', 'pow', 'range', 'reversed', 'round', 'set',
            'sorted', 'str', 'sum', 'tuple', 'zip'
        }
        
        # Create restricted globals
        restricted_globals = {
            '__builtins__': {name: getattr(__builtins__, name) for name in safe_builtins}
        }
        
        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                self.logger.warning(f"Could not import allowed module: {module_name}")
        
        return restricted_globals

    def execute_safely(self, code: str, timeout: int = 30) -> Any:
        """Execute code safely in a sandboxed environment."""
        self.logger.info("[SECURITY] Executing code in sandbox...")

        # Create sandbox environment
        sandbox_globals = self.create_sandbox_environment()
        sandbox_locals = {}

        # Set up timeout handler (POSIX only)
        if os.name == "posix":
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        else:
            old_handler = None

        try:
            # Monitor resource usage before execution
            initial_usage = self.monitor_resource_usage()
            self.logger.info(f"Initial resource usage: {initial_usage}")

            # Execute the code
            result = exec(code, sandbox_globals, sandbox_locals)

            # Monitor resource usage after execution
            final_usage = self.monitor_resource_usage()
            self.logger.info(f"Final resource usage: {final_usage}")

            return sandbox_locals.get('result', result)

        except TimeoutError:
            self.logger.error("Code execution timed out")
            raise
        except Exception as e:
            self.logger.error(f"Error during code execution: {e}")
            raise
        finally:
            if os.name == "posix":
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler

    def analyze_and_execute(self, code: str, signature: str = "") -> Any:
        """Main method to analyze and execute code safely."""
        code_hash = self.hash_code(code)
        self.logger.info(f"[SECURITY] Processing code with hash: {code_hash}")
        
        # Perform static analysis
        if not self.static_analysis(code):
            raise ValueError("Static analysis failed. Code is not safe to execute.")
        
        # Verify signature if provided
        if signature and not self.verify_signature(code.encode('utf-8'), signature):
            raise ValueError("Signature verification failed. Code may be tampered with.")
        
        # Enforce resource limits
        self.enforce_resource_limits()
        
        # Execute the code safely
        try:
            result = self.execute_safely(code, self.resource_limits['execution_time'])
            self.logger.info("[SECURITY] Code executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            raise

    def generate_execution_report(self, code: str, execution_time: float, 
                                resource_usage: Dict[str, float]) -> Dict[str, Any]:
        """Generate a detailed execution report."""
        return {
            'code_hash': self.hash_code(code),
            'execution_time': execution_time,
            'resource_usage': resource_usage,
            'timestamp': time.time(),
            'status': 'completed'
        }


# Example usage
if __name__ == "__main__":
    # Initialize security manager
    security_manager = SecurityManager(
        allowed_modules=['math', 'random', 'datetime'],
        # public_key_path='public_key.pem'  # Optional: path to public key for signature verification
    )
    
    # Example safe code
    safe_code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
"""
    
    try:
        result = security_manager.analyze_and_execute(safe_code)
        print(f"Execution result: {result}")
    except Exception as e:
        print(f"Execution failed: {e}")