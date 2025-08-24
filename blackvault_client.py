import requests
import os

class BlackVaultClient:
    def __init__(self):
        self.api_url = os.getenv("BLACKVAULT_API_URL")
        self.api_token = os.getenv("BLACKVAULT_API_TOKEN")

    def is_authorized(self, agent_id):
        # Example: check agent authorization
        resp = requests.get(f"{self.api_url}/auth/{agent_id}", headers={"Authorization": f"Bearer {self.api_token}"})
        return resp.status_code == 200

    def log_access(self, agent_id, vault_key_id, action, result="success"):
        # Log access event
        payload = {
            "agent_id": agent_id,
            "vault_key_id": vault_key_id,
            "action": action,
            "result": result
        }
        requests.post(f"{self.api_url}/audit", json=payload, headers={"Authorization": f"Bearer {self.api_token}"})

    def retrieve_key(self, vault_key_id, agent_id):
        # Retrieve key securely
        if not self.is_authorized(agent_id):
            self.log_access(agent_id, vault_key_id, "retrieve", "unauthorized")
            raise PermissionError("Unauthorized access")
        resp = requests.get(f"{self.api_url}/keys/{vault_key_id}", headers={"Authorization": f"Bearer {self.api_token}"})
        self.log_access(agent_id, vault_key_id, "retrieve", "success" if resp.status_code == 200 else "fail")
        if resp.status_code == 200:
            return resp.json()["private_key"]
        raise Exception("Key retrieval failed")