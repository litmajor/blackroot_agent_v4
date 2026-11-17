import json
import os
import time
from datetime import datetime

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    HAS_FERNET = False
    import base64
    class Fernet:  # type: ignore
        """Lightweight obfuscation cipher fallback when cryptography package is unavailable."""
        def __init__(self, key):
            self.key = key if isinstance(key, bytes) else str(key).encode()
        @staticmethod
        def generate_key():
            return base64.urlsafe_b64encode(os.urandom(32))
        def encrypt(self, data: bytes) -> bytes:
            k = self.key
            xored = bytes(b ^ k[i % len(k)] for i, b in enumerate(data))
            return base64.b64encode(xored)
        def decrypt(self, token: bytes) -> bytes:
            k = self.key
            decoded = base64.b64decode(token)
            return bytes(b ^ k[i % len(k)] for i, b in enumerate(decoded))

class MirrorStorage:
    def __init__(self, base_dir="mirrorcore/logs", key_path="mirrorcore/.key"):
        self.base_dir = base_dir
        self.key_path = key_path
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                print(f"[MIRROR-STORAGE] Loaded encryption key from {self.key_path}")
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            print(f"[MIRROR-STORAGE] Generated and saved new encryption key to {self.key_path}")
            return key

    def _get_log_path(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.base_dir, f"reflections_{today}.jsonl")

    def persist(self, data):
        try:
            path = self._get_log_path()
            encrypted = self.cipher.encrypt(json.dumps(data).encode())
            with open(path, 'ab') as f:
                f.write(encrypted + b'\n')
            print(f"[MIRROR-STORAGE] Encrypted and saved to {path}")
        except Exception as e:
            print(f"[MIRROR-STORAGE][ERR] Failed to persist reflection: {e}")

    def purge_old_logs(self, days=7):
        print(f"[MIRROR-STORAGE] Purging logs older than {days} days...")
        cutoff = time.time() - days * 86400
        removed = 0
        for fname in os.listdir(self.base_dir):
            if not fname.startswith("reflections_") or not fname.endswith(".jsonl"):
                continue
            path = os.path.join(self.base_dir, fname)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        print(f"[MIRROR-STORAGE] Removed {removed} old log files.")

    def search_by_date(self, date_str):
        path = os.path.join(self.base_dir, f"reflections_{date_str}.jsonl")
        if not os.path.exists(path):
            print(f"[MIRROR-STORAGE] No logs for {date_str}")
            return []
        with open(path, 'rb') as f:
            return [json.loads(self.cipher.decrypt(line.strip()).decode()) for line in f.readlines()]

    def list_log_dates(self):
        return sorted([f.split('_')[-1].replace('.jsonl', '') for f in os.listdir(self.base_dir) if f.startswith("reflections_") and f.endswith(".jsonl")])

    def query_last(self, n: int = 100):
        """
        Query the last n reflection / log entries in reverse chronological order.
        Handles decryption with graceful fallback to plaintext JSON lines.
        """
        results = []
        if not os.path.exists(self.base_dir):
            return results

        files = sorted(
            [f for f in os.listdir(self.base_dir) if f.startswith("reflections_") and f.endswith(".jsonl")],
            reverse=True
        )

        for fname in files:
            path = os.path.join(self.base_dir, fname)
            try:
                with open(path, "rb") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        decrypted = self.cipher.decrypt(line).decode("utf-8")
                        entry = json.loads(decrypted)
                        results.append(entry)
                    except Exception:
                        try:
                            entry = json.loads(line.decode("utf-8"))
                            results.append(entry)
                        except Exception:
                            continue
                    if len(results) >= n:
                        return results
            except Exception as e:
                print(f"[MIRROR-STORAGE][WARN] Failed reading log file {fname}: {e}")
                continue

        return results
