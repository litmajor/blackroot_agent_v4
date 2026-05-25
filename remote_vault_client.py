#!/usr/bin/env python3
"""
Remote Vault Client: Exfiltrate encrypted credentials and artifacts from remote BlackVault
Provides secure client-side interface for retrieving, exfiltrating, and analyzing vault data
"""

import requests
import json
import os
import logging
import time
from typing import Optional, Dict, List, Any
from pathlib import Path
import base64
from urllib.parse import urljoin
import sys


class RemoteVaultClient:
    """Client for secure remote BlackVault access and exfiltration"""
    
    def __init__(self, server_url: str, api_token: str, timeout: int = 30, verify_ssl: bool = False):
        """
        Initialize remote vault client
        
        Args:
            server_url: Base URL of remote vault server (e.g., http://192.168.1.100:5000)
            api_token: Bearer token for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (set False for self-signed certs)
        """
        self.server_url = server_url.rstrip('/')
        self.api_token = api_token
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Setup logging
        self.logger = logging.getLogger('RemoteVaultClient')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Setup HTTP session with auth
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'RemoteVaultClient/1.0'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request to vault server
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            Response JSON or None if request failed
        """
        url = urljoin(self.server_url, endpoint)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout, verify=self.verify_ssl)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout, verify=self.verify_ssl)
            elif method == 'DELETE':
                response = self.session.delete(url, timeout=self.timeout, verify=self.verify_ssl)
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection failed to {self.server_url}")
            return None
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout to {self.server_url}")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None
    
    def health_check(self) -> bool:
        """
        Check if remote vault server is reachable and healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        result = self._make_request('GET', '/health')
        if result and result.get('status') == 'healthy':
            self.logger.info("✓ Connection to remote vault established")
            return True
        else:
            self.logger.error("✗ Remote vault server is not responding")
            return False
    
    def list_artifacts(self) -> List[str]:
        """
        List all artifacts stored in remote vault
        
        Returns:
            List of artifact names
        """
        result = self._make_request('GET', '/list')
        if result and 'artifacts' in result:
            artifacts = result['artifacts']
            self.logger.info(f"Found {len(artifacts)} artifacts in vault")
            return artifacts
        return []
    
    def retrieve_artifact(self, artifact_name: str, decode: bool = True) -> Optional[str]:
        """
        Retrieve a single artifact from remote vault
        
        Args:
            artifact_name: Name of the artifact
            decode: Whether to decode from base64
            
        Returns:
            Artifact data as string or None if not found
        """
        result = self._make_request('GET', f'/retrieve/{artifact_name}')
        
        if result and 'data' in result:
            data = result['data']
            
            if decode and result.get('encoding') == 'base64':
                try:
                    data = base64.b64decode(data).decode('utf-8')
                except Exception as e:
                    self.logger.warning(f"Failed to decode artifact: {e}")
            
            self.logger.info(f"✓ Retrieved artifact: {artifact_name}")
            return data
        else:
            self.logger.error(f"Artifact not found: {artifact_name}")
            return None
    
    def retrieve_artifact_json(self, artifact_name: str) -> Optional[Dict]:
        """
        Retrieve a JSON artifact from remote vault
        
        Args:
            artifact_name: Name of the artifact
            
        Returns:
            Parsed JSON object or None if not found
        """
        result = self._make_request('GET', f'/retrieve-json/{artifact_name}')
        
        if result and 'data' in result:
            self.logger.info(f"✓ Retrieved JSON artifact: {artifact_name}")
            return result.get('data')
        else:
            self.logger.error(f"JSON artifact not found: {artifact_name}")
            return None
    
    def list_credentials(self) -> Dict[str, Any]:
        """
        Get summary statistics of all credentials in vault
        
        Returns:
            Dictionary with credential statistics
        """
        result = self._make_request('GET', '/stats')
        
        if result and 'credentials' in result:
            stats = result['credentials']
            total = stats.get('total', 0)
            self.logger.info(f"Vault contains {total} total credentials across {len(stats)} categories")
            return stats
        return {}
    
    def export_all(self, output_dir: str = "./vault_export") -> Dict[str, Any]:
        """
        Export all artifacts and credentials from vault
        
        Args:
            output_dir: Directory to save exported data
            
        Returns:
            Dictionary with export statistics
        """
        result = self._make_request('GET', '/export-all')
        
        if not result or 'artifacts' not in result:
            self.logger.error("Export failed - invalid response")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        artifacts = result.get('artifacts', {})
        exported_count = 0
        
        for name, data in artifacts.items():
            try:
                output_file = os.path.join(output_dir, f"{name}.json")
                
                if isinstance(data, dict):
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    with open(output_file, 'w') as f:
                        f.write(str(data))
                
                exported_count += 1
                self.logger.debug(f"Exported: {output_file}")
            
            except Exception as e:
                self.logger.warning(f"Failed to export {name}: {e}")
        
        stats = {
            'total_artifacts': len(artifacts),
            'exported_count': exported_count,
            'export_dir': os.path.abspath(output_dir),
            'total_credentials': result.get('total_credentials', 0)
        }
        
        self.logger.info(f"✓ Exported {exported_count}/{len(artifacts)} artifacts to {output_dir}")
        return stats
    
    def exfiltrate_to_file(self, artifact_name: str, output_file: str) -> bool:
        """
        Exfiltrate single artifact to file
        
        Args:
            artifact_name: Name of the artifact
            output_file: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        data = self.retrieve_artifact(artifact_name)
        
        if data is None:
            return False
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(data)
            self.logger.info(f"✓ Exfiltrated to: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Exfiltration failed: {e}")
            return False
    
    def exfiltrate_all_to_directory(self, output_dir: str = "./exfiltrated_vault") -> int:
        """
        Exfiltrate all artifacts to directory with organized structure
        
        Args:
            output_dir: Base directory for exfiltration
            
        Returns:
            Number of successfully exfiltrated artifacts
        """
        artifacts = self.list_artifacts()
        
        if not artifacts:
            self.logger.error("No artifacts to exfiltrate")
            return 0
        
        exfiltrated = 0
        
        for artifact_name in artifacts:
            category = artifact_name.split('_')[0] if '_' in artifact_name else 'misc'
            category_dir = os.path.join(output_dir, category)
            
            output_file = os.path.join(category_dir, f"{artifact_name}.txt")
            
            if self.exfiltrate_to_file(artifact_name, output_file):
                exfiltrated += 1
        
        self.logger.info(f"✓ Exfiltrated {exfiltrated}/{len(artifacts)} artifacts to {output_dir}")
        return exfiltrated
    
    def exfiltrate_credentials_by_type(self, credential_type: str, output_dir: str) -> int:
        """
        Exfiltrate credentials of specific type
        
        Args:
            credential_type: Type of credentials (Environment, System, Infrastructure, Bypass)
            output_dir: Output directory
            
        Returns:
            Number of credentials exfiltrated
        """
        stats = self.list_credentials()
        
        if credential_type not in stats:
            self.logger.error(f"Credential type not found: {credential_type}")
            return 0
        
        artifact_name = f"{credential_type}_indicators"
        
        # Try JSON first
        json_data = self.retrieve_artifact_json(artifact_name)
        if json_data:
            output_file = os.path.join(output_dir, f"{artifact_name}.json")
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                self.logger.info(f"✓ Exfiltrated {credential_type} credentials: {output_file}")
                return len(json_data.get('credentials', []))
            except Exception as e:
                self.logger.error(f"Exfiltration failed: {e}")
                return 0
        
        # Fallback to text
        if self.exfiltrate_to_file(artifact_name, os.path.join(output_dir, f"{artifact_name}.txt")):
            return 1
        
        return 0
    
    def interactive_browse(self):
        """Interactive artifact browsing and exfiltration"""
        print("\n" + "="*60)
        print("REMOTE VAULT CLIENT - Interactive Mode")
        print("="*60)
        
        # Health check
        if not self.health_check():
            print("❌ Cannot connect to remote vault")
            return
        
        # Main menu
        while True:
            print("\n[1] List all artifacts")
            print("[2] View specific artifact")
            print("[3] Export all artifacts")
            print("[4] Exfiltrate to directory")
            print("[5] View credential statistics")
            print("[6] Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                artifacts = self.list_artifacts()
                if artifacts:
                    for i, name in enumerate(artifacts, 1):
                        print(f"  {i}. {name}")
                else:
                    print("No artifacts found")
            
            elif choice == '2':
                artifact_name = input("Enter artifact name: ").strip()
                data = self.retrieve_artifact(artifact_name)
                if data:
                    print(f"\n------- {artifact_name} -------")
                    print(data[:500] + ("..." if len(data) > 500 else ""))
                    print("-------")
            
            elif choice == '3':
                output_dir = input("Enter output directory (default: ./vault_export): ").strip() or "./vault_export"
                stats = self.export_all(output_dir)
                print(f"\n✓ Exported {stats.get('exported_count', 0)} artifacts")
            
            elif choice == '4':
                output_dir = input("Enter output directory (default: ./exfiltrated): ").strip() or "./exfiltrated"
                count = self.exfiltrate_all_to_directory(output_dir)
                print(f"\n✓ Exfiltrated {count} artifacts")
            
            elif choice == '5':
                stats = self.list_credentials()
                print("\n------- Credential Statistics -------")
                for cred_type, count in stats.items():
                    if isinstance(count, int):
                        print(f"  {cred_type}: {count}")
                print("-------")
            
            elif choice == '6':
                print("Exiting...")
                break
            
            else:
                print("Invalid option")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remote BlackVault Client')
    parser.add_argument('--server', required=True, help='Vault server URL (e.g., http://192.168.1.100:5000)')
    parser.add_argument('--token', required=True, help='API authentication token')
    parser.add_argument('--list', action='store_true', help='List all artifacts')
    parser.add_argument('--export', help='Export all to directory')
    parser.add_argument('--retrieve', help='Retrieve specific artifact')
    parser.add_argument('--stats', action='store_true', help='Show credential statistics')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    parser.add_argument('--no-verify-ssl', action='store_true', help='Skip SSL verification')
    
    args = parser.parse_args()
    
    # Initialize client
    client = RemoteVaultClient(
        server_url=args.server,
        api_token=args.token,
        timeout=args.timeout,
        verify_ssl=not args.no_verify_ssl
    )
    
    # Execute operations
    if args.interactive:
        client.interactive_browse()
    
    elif args.list:
        artifacts = client.list_artifacts()
        for name in artifacts:
            print(name)
    
    elif args.export:
        stats = client.export_all(args.export)
        print(f"Exported {stats.get('exported_count', 0)} artifacts")
    
    elif args.retrieve:
        data = client.retrieve_artifact(args.retrieve)
        if data:
            print(data)
    
    elif args.stats:
        stats = client.list_credentials()
        for category, count in stats.items():
            if isinstance(count, int):
                print(f"{category}: {count}")
    
    else:
        # Default: health check
        if client.health_check():
            print("✓ Vault is accessible")
            artifacts = client.list_artifacts()
            print(f"✓ Found {len(artifacts)} artifacts")
        else:
            print("✗ Cannot connect to vault")


if __name__ == '__main__':
    main()
