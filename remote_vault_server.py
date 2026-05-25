#!/usr/bin/env python3
"""
Remote BlackVault Server - RESTful API for remote vault access
Exposes vault operations over HTTP with authentication and encryption
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from black_vault import BlackVault

# Try to import Flask
try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("WARNING: Flask not installed. Install with: pip install flask")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('RemoteVaultServer')

class RemoteVaultServer:
    """REST API server for remote BlackVault access"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5000, api_key: str = None):
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask required. Install: pip install flask")
        
        self.host = host
        self.port = port
        self.api_key = api_key or os.environ.get('VAULT_API_KEY', 'default-insecure-key')
        self.vault = BlackVault()
        self.app = Flask(__name__)
        self._setup_routes()
        
        logger.info(f"RemoteVaultServer initialized on {host}:{port}")
    
    def _check_auth(self) -> bool:
        """Verify API key from request header"""
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return False
        token = auth_header[7:]
        return token == self.api_key
    
    def _auth_required(self, f):
        """Decorator for auth-protected endpoints"""
        def decorated(*args, **kwargs):
            if not self._check_auth():
                return jsonify({'error': 'Unauthorized'}), 401
            return f(*args, **kwargs)
        decorated.__name__ = f.__name__
        return decorated
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint (no auth required)"""
            return jsonify({
                'status': 'online',
                'timestamp': datetime.utcnow().isoformat(),
                'vault_path': self.vault.vault_dir
            })
        
        @self.app.route('/store/<name>', methods=['POST'])
        @self._auth_required
        def store_artifact(name):
            """Store artifact in vault"""
            try:
                data = request.get_data()
                expire_minutes = request.args.get('expire_minutes', type=int)
                
                self.vault.store(name, data, expire_minutes=expire_minutes)
                
                logger.info(f"Stored artifact: {name} ({len(data)} bytes)")
                return jsonify({
                    'status': 'stored',
                    'name': name,
                    'size': len(data),
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Failed to store {name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/retrieve/<name>', methods=['GET'])
        @self._auth_required
        def retrieve_artifact(name):
            """Retrieve artifact from vault"""
            try:
                data = self.vault.retrieve(name)
                logger.info(f"Retrieved artifact: {name} ({len(data)} bytes)")
                return Response(data, mimetype='application/octet-stream')
            except FileNotFoundError:
                return jsonify({'error': 'Artifact not found'}), 404
            except ValueError as e:
                logger.error(f"Integrity check failed for {name}: {e}")
                return jsonify({'error': f'Integrity check failed: {e}'}), 400
            except Exception as e:
                logger.error(f"Failed to retrieve {name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/retrieve-json/<name>', methods=['GET'])
        @self._auth_required
        def retrieve_json(name):
            """Retrieve and parse artifact as JSON"""
            try:
                data = self.vault.retrieve(name)
                parsed = json.loads(data.decode())
                logger.info(f"Retrieved JSON artifact: {name}")
                return jsonify(parsed), 200
            except FileNotFoundError:
                return jsonify({'error': 'Artifact not found'}), 404
            except json.JSONDecodeError:
                return jsonify({'error': 'Artifact is not valid JSON'}), 400
            except Exception as e:
                logger.error(f"Failed to retrieve JSON {name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/list', methods=['GET'])
        @self._auth_required
        def list_artifacts():
            """List all artifacts in vault"""
            try:
                artifacts = self.vault.list_artifacts()
                logger.info(f"Listed {len(artifacts)} artifacts")
                return jsonify({
                    'artifacts': artifacts,
                    'count': len(artifacts),
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Failed to list artifacts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/delete/<name>', methods=['DELETE'])
        @self._auth_required
        def delete_artifact(name):
            """Delete artifact from vault"""
            try:
                self.vault.delete(name)
                logger.info(f"Deleted artifact: {name}")
                return jsonify({
                    'status': 'deleted',
                    'name': name,
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Failed to delete {name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/export-all', methods=['GET'])
        @self._auth_required
        def export_all():
            """Export all artifacts as JSON array"""
            try:
                artifacts = self.vault.list_artifacts()
                exported = []
                
                for artifact_name in artifacts:
                    try:
                        data = self.vault.retrieve(artifact_name)
                        try:
                            parsed = json.loads(data.decode())
                            exported.append({
                                'name': artifact_name,
                                'content': parsed,
                                'format': 'json'
                            })
                        except:
                            # Binary or non-JSON data
                            import base64
                            exported.append({
                                'name': artifact_name,
                                'content': base64.b64encode(data).decode(),
                                'format': 'base64'
                            })
                    except Exception as e:
                        logger.warning(f"Skipped artifact {artifact_name}: {e}")
                        continue
                
                logger.info(f"Exported {len(exported)} artifacts")
                return jsonify({
                    'artifacts': exported,
                    'count': len(exported),
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Failed to export all: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        @self._auth_required
        def vault_stats():
            """Get vault statistics"""
            try:
                artifacts = self.vault.list_artifacts()
                total_size = 0
                
                for artifact in artifacts:
                    filepath = os.path.join(self.vault.vault_dir, artifact)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
                
                logger.info(f"Vault stats requested: {len(artifacts)} artifacts, {total_size} bytes")
                return jsonify({
                    'vault_dir': self.vault.vault_dir,
                    'artifact_count': len(artifacts),
                    'total_size': total_size,
                    'total_size_mb': round(total_size / (1024*1024), 2),
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self, debug: bool = False):
        """Start the Flask server"""
        logger.info(f"Starting RemoteVaultServer on {self.host}:{self.port}")
        logger.info(f"API Key required in Authorization: Bearer header")
        self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Remote BlackVault Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--api-key', help='API key (default: env var VAULT_API_KEY or "default-insecure-key")')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    
    args = parser.parse_args()
    
    server = RemoteVaultServer(
        host=args.host,
        port=args.port,
        api_key=args.api_key
    )
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    REMOTE BLACKVAULT SERVER                           ║
╚════════════════════════════════════════════════════════════════════════╝

Configuration:
  • Host:        {args.host}
  • Port:        {args.port}
  • API Key:     {'Set' if args.api_key or os.environ.get('VAULT_API_KEY') else 'INSECURE (default)'}
  • Debug:       {args.debug}

Endpoints:
  • GET  /health                    - Health check
  • POST /store/<name>              - Store artifact
  • GET  /retrieve/<name>           - Get binary artifact
  • GET  /retrieve-json/<name>      - Get JSON artifact
  • GET  /list                      - List all artifacts
  • DELETE /delete/<name>           - Delete artifact
  • GET  /export-all                - Export all as JSON
  • GET  /stats                     - Vault statistics

Authentication:
  Include header: Authorization: Bearer <api_key>

Starting server...
""")
    
    server.run(debug=args.debug)


if __name__ == '__main__':
    if not FLASK_AVAILABLE:
        print("ERROR: Flask is required!")
        print("Install: pip install flask")
        sys.exit(1)
    
    main()
