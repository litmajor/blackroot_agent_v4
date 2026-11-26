# Blackroot Artifact Catalog
> **v4.5.0-Hardened** · `recon/target_identity.py` · `ghost_layer_financial.py`

This document is the authoritative reference for every artifact type Blackroot can collect,
price, classify, and trade on the BEL economic layer. It is generated from the live
`ArtifactValuationTable` and `GhostLayer` collection pipeline.

---

## 1. How Artifact Valuation Works

```
Base ROOT Value  ×  Privilege Multiplier  ×  Domain Role Multiplier  =  Final ROOT Price
```

### Privilege Multipliers

| Privilege Level | Multiplier |
|---|---|
| `user` | 1.00× |
| `admin` | 1.35× |
| `system` / `root` | 1.50× |

### Domain Role Multipliers

| Domain Role | Multiplier |
|---|---|
| `workstation` | 1.00× |
| `member_server` | 1.20× |
| `cloud_vm` | 1.35× |
| `domain_controller` | 1.60× |

> **Example** — Azure PRT Cache on a Domain Controller with admin token:
> `300 ROOT × 1.35 × 1.60 = 648 ROOT`

### BEL Integration

Every vault `store()` call with a `target_id` in metadata automatically:
1. Calls `TargetIdentityRegistry.ingest_extract()` → classifies artifact → prices it
2. Calls `Ledger.credit(agent_id, root_value, reason)` → rewards harvesting agent
3. Updates the `CausalGraph` data_value edge via EMA blend
4. Recomputes `TargetIdentity.weighted_priority_score` for ranking

---

## 2. Artifact Type Reference

> **Tiers:** 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Low

---

### 🔴 AZURE_PRT_CACHE · 300 ROOT base
**Type ID:** `azure_prt_cache`  
**Causal Action:** `extract_cloud`  
**Causal Signal:** 0.97

| Field | Detail |
|---|---|
| **What it is** | Windows Cloud AP token plugin cache; MSAL `.bin` token blobs; WAM (Web Account Manager) session state |
| **Where it lives** | `%LOCALAPPDATA%\Microsoft\Windows\CloudAP` · `%LOCALAPPDATA%\Microsoft\TokenBroker\Cache` · `%LOCALAPPDATA%\Microsoft\IdentityCache` |
| **What it grants** | A valid Azure AD Primary Refresh Token (PRT) provides seamless SSO across **all** M365 services — Exchange, SharePoint, Teams, Azure portal, Dynamics 365, Workday, NetSuite, Salesforce — without credentials or MFA prompts |
| **EDR detection** | Low — blobs are DPAPI-encrypted and sit in standard MS profile paths; no process injection required to enumerate |
| **Privilege needed** | User (read own cache) · Admin (read all users) |
| **Exploitation path** | Extract encrypted blob → DPAPI decrypt (or relay via `PRT Cookie` abuse using `ROADtoken` / `AADInternals`) → craft seamless SSO auth cookie → access financial portals. See detailed operational steps in [playbook_azure_prt.md](docs/playbooks/playbook_azure_prt.md). |
| **Downstream artifacts** | Unlocks Workday payroll exports, SharePoint contract libraries, Azure billing subscriptions, Dynamics finance modules |
| **Collector** | `ghost_layer_financial._collect_azure_prt()` |
| **Classifier keyword** | `azure_prt`, `prt_cache`, `wam_cache`, `cloudap`, `msal`, `tokenbroker` |

---

### 🔴 WEB3_DEVELOPER_SECRET · 290 ROOT base
**Type ID:** `web3_developer_secret`  
**Causal Action:** `extract_crypto`  
**Causal Signal:** 0.96

| Field | Detail |
|---|---|
| **What it is** | Blockchain deployment secrets in dev environments: `.env` files with provider API keys and deployer private keys; Foundry / Hardhat encrypted keystores; Solana CLI `id.json` |
| **Where it lives** | `~/.config/solana/id.json` (Solana — **unencrypted** 64-byte keypair) · `~/.foundry/keystores/` · `hardhat.config.js/ts` · any `.env` / `.env.production` in dev repos |
| **What it grants** | Direct control of on-chain treasury wallets; ability to redeploy / drain smart contracts; access to Infura/Alchemy node quotas used for production |
| **EDR detection** | None — static files, no process interaction |
| **Privilege needed** | User |
| **Exploitation path** | Extract `DEPLOYER_PRIVATE_KEY` → import into MetaMask or `cast` (Foundry) → drain deployer wallet; or enumerate `INFURA_API_KEY` → access all RPC calls to monitor mempool / inject txs. See operational guide in [playbook_web3_treasury.md](docs/playbooks/playbook_web3_treasury.md). |
| **Downstream artifacts** | On-chain wallet seed access; smart contract admin keys; API billing account takeover |
| **Collector** | `ghost_layer_financial._collect_web3_developer_secrets()` |
| **Classifier keyword** | `solana/id.json`, `foundry_keystore`, `infura_api`, `alchemy_key`, `deployer_private`, `web3_developer` |

---

### 🔴 PAYMENT_GATEWAY_KEY · 275 ROOT base
**Type ID:** `payment_gateway_key`  
**Causal Action:** `extract_cloud`  
**Causal Signal:** 0.95

| Field | Detail |
|---|---|
| **What it is** | Live-mode secret API keys for payment processors: Stripe (`sk_live_...`, `rk_live_...`, `whsec_...`), PayPal client secrets, Square access tokens, Plaid client secrets, Twilio auth tokens |
| **Where it lives** | Source files (`.py`, `.js`, `.ts`, `.env`, `.yaml`, `.json`, `.toml`) in dev repos; CI/CD environment configs; deployment scripts |
| **What it grants** | Full payment API access — issue charges, initiate refunds, enumerate all customer payment methods, access full transaction history, trigger webhook callbacks |
| **EDR detection** | None — static key strings in source |
| **Privilege needed** | User |
| **Exploitation path** | Stripe `sk_live_` → `stripe customers list` → enumerate stored cards → charge arbitrary amounts; or `stripe payouts create` to redirect future payouts to attacker bank account. See triage steps in [playbook_payment_gateways.md](docs/playbooks/playbook_payment_gateways.md). |
| **Downstream artifacts** | Customer PII + payment history; payout bank account details; webhook endpoints for future injection |
| **Collector** | `ghost_layer_financial._collect_payment_gateway_keys()` |
| **Classifier keyword** | `sk_live_`, `stripe_key`, `payment_gateway`, `paypal_secret`, `square_token`, `plaid_secret`, `twilio_auth` |

---

### 🔴 WALLET_SEED · 250 ROOT base
**Type ID:** `wallet_seed`  
**Causal Action:** `extract_crypto`  
**Causal Signal:** 0.98

| Field | Detail |
|---|---|
| **What it is** | BIP-39 mnemonic seed phrases and raw private keys from browser extension wallets and desktop portfolio apps |
| **Where it lives** | MetaMask: `%APPDATA%\Local\Google\Chrome\User Data\Default\Local Extension Settings\nkbihfbeogaeaoehlefnkodbefgpgknn\` · Brave Wallet: `Brave-Browser\User Data\Default\Brave Wallet` · Ledger Live: `%APPDATA%\Ledger Live\app.json` · Trezor: `%APPDATA%\@trezor\suite-desktop` · Electrum: `%APPDATA%\Electrum\wallets\` · Bitcoin Core: `%APPDATA%\Bitcoin\wallet.dat` |
| **What it grants** | Total, irreversible control of all addresses derived from the seed — every chain, every token, full balance sweep possible in one transaction |
| **EDR detection** | Low — LevelDB reads mimic normal browser behavior; `wallet.dat` is a file copy |
| **Privilege needed** | User |
| **Exploitation path** | Extract encrypted LevelDB seed → DPAPI decrypt (Chrome-keyed) → import 12/24-word BIP-39 phrase into attacker wallet → sweep all funds |
| **Downstream artifacts** | NFT collections; DeFi positions; multisig quorum key |
| **Collector** | `ghost_layer_financial._collect_hw_wallet_artifacts()` + `GhostLayer._extract_brave_browser_data()` |
| **Classifier keyword** | `wallet`, `metamask`, `brave_wallet`, `seed`, `mnemonic`, `phantom`, `exodus`, `ledger_live`, `trezor`, `electrum`, `bitcoin`, `wallet.dat` |

---

### 🔴 BANKING_BATCH_FILE · 220 ROOT base
**Type ID:** `banking_batch_file`  
**Causal Action:** `extract_cloud`  
**Causal Signal:** 0.93

| Field | Detail |
|---|---|
| **What it is** | Corporate payment batch files: NACHA ACH (`.ach`), SWIFT MT940 / MT101 statements, SEPA ISO 20022 Credit Transfer XML (`pain.001`), CAMT statement XML |
| **Where it lives** | `Downloads`, `Desktop`, `Documents`, OneDrive/Dropbox sync folders; treasury workstation shared drives |
| **What it grants** | Full payroll routing numbers, company IBANs, beneficiary account lists, real-time corporate balance snapshots, all vendor wire details |
| **EDR detection** | None — plaintext or XML file read |
| **Privilege needed** | User (treasury analyst context) |
| **Exploitation path** | Extract NACHA file → parse routing/account numbers → register identical account at receiving bank → submit duplicate ACH entry; or modify beneficiary in future batch. Full byte layout in [playbook_banking_batches.md](docs/playbooks/playbook_banking_batches.md). |
| **Downstream artifacts** | Complete vendor payment list; employee payroll IBAN list; corporate bank balance |
| **Collector** | `ghost_layer_financial._collect_banking_files()` |
| **Classifier keyword** | `.ach`, `nacha`, `swift`, `mt940`, `mt101`, `sepa`, `banking_batch`, `wire_transfer`, `iban` |

---

### 🔴 ERP_ACCOUNTING_CONFIG · 190 ROOT base
**Type ID:** `erp_accounting_config`  
**Causal Action:** `extract_db`  
**Causal Signal:** 0.90

| Field | Detail |
|---|---|
| **What it is** | Desktop accounting and ERP connection artifacts: QuickBooks company database files; SAP GUI connection profiles; Sage/Peachtree data files |
| **Where it lives** | QuickBooks: `C:\Users\Public\Documents\Intuit\QuickBooks\*.qbw` · `C:\ProgramData\Intuit\QuickBooks\` · SAP: `%APPDATA%\SAP\Common\saplogon.ini`, `SAPUILandscape.xml` · `C:\Program Files (x86)\SAP\FrontEnd\SAPgui\saplogon.ini` · Sage: `Documents\*.ptb`, `*.sdb` |
| **What it grants** | Full company financial history (P&L, balance sheet, AP/AR); SAP system hostnames + SIDs enabling direct ABAP login attempts; Sage company file database access |
| **EDR detection** | None — file-level access |
| **Privilege needed** | User |
| **Exploitation path** | Copy `.qbw` → open in QuickBooks Desktop (no password if user has file access) → export all vendor payments, employees, payroll; SAP `saplogon.ini` → enumerate application servers → password spray or relay NTLM |
| **Downstream artifacts** | Vendor bank accounts; payroll data; GL transaction history; SAP username list |
| **Collector** | `ghost_layer_financial._collect_erp_artifacts()` |
| **Classifier keyword** | `.qbw`, `.qbb`, `.qbm`, `saplogon`, `sapuilandscape`, `quickbooks`, `sage_data`, `erp_` |

---

### 🔴 CLOUD_IAM_TOKEN · 180 ROOT base
**Type ID:** `cloud_iam_token`  
**Causal Action:** `extract_cloud`  
**Causal Signal:** 0.92

| Field | Detail |
|---|---|
| **What it is** | Static cloud provider credential files: AWS long-term access keys and session tokens; GCP service account JSON / ADC; Azure CLI token cache |
| **Where it lives** | `~/.aws/credentials` · `~/.aws/config` · `~/.config/gcloud/application_default_credentials.json` · `~/.azure/credentials` · `%APPDATA%\gcloud\` · kubeconfig: `~/.kube/config` |
| **What it grants** | Persistent programmatic access to cloud infrastructure; scope depends on attached IAM policies — commonly includes S3/GCS/Blob read-write, EC2/GCE instance control, secrets manager, billing APIs |
| **EDR detection** | Low — file read; no EDR hooks on credential files |
| **Privilege needed** | User |
| **Exploitation path** | `aws sts get-caller-identity` → enumerate permissions → escalate via misconfigured IAM role chain → access RDS secrets, S3 data lakes, Lambda source code, CloudTrail disablement |
| **Downstream artifacts** | S3 bucket contents; RDS DB credentials in Secrets Manager; EC2 instance metadata IMDS tokens |
| **Collector** | `GhostLayer._scan_api_credentials()` + `_extract_cloud_provider_credentials()` |
| **Classifier keyword** | `aws`, `gcp`, `azure`, `iam`, `cloud`, `sts_token`, `kubeconfig` |

---

### 🔴 BROWSER_WEB_STORAGE · 160 ROOT base
**Type ID:** `browser_web_storage`  
**Causal Action:** `extract_chrome`  
**Causal Signal:** 0.86

| Field | Detail |
|---|---|
| **What it is** | Chrome-family IndexedDB and LocalStorage LevelDB databases storing OAuth2 refresh tokens, JWTs, and persistent session state for SPA financial dashboards |
| **Where it lives** | Chrome: `%LOCALAPPDATA%\Google\Chrome\User Data\Default\IndexedDB\` · Edge: `%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\IndexedDB\` · Brave: `BraveSoftware\Brave-Browser\User Data\Default\IndexedDB\` |
| **What it grants** | Active (non-expired) OAuth2 refresh tokens for Stripe dashboard, Brex/Ramp spend management, QuickBooks Online, Workday payroll, NetSuite ERP — without needing the user's password or MFA |
| **EDR detection** | Low — LevelDB file reads appear as browser profile access |
| **Privilege needed** | User (own profile) · Admin (all profiles) |
| **Exploitation path** | Copy LevelDB folder → open with `leveldb-cli` or Python `plyvel` → find OAuth tokens for target domain → replay refresh token against `/oauth/token` endpoint → obtain fresh access token |
| **Downstream artifacts** | Stripe charges API; Workday HCM employee records; Brex transaction history |
| **Collector** | `ghost_layer_financial._collect_browser_web_storage()` |
| **Classifier keyword** | `indexeddb`, `localstorage`, `web_storage`, `browser_web_storage`, `leveldb` |

---

### 🔴 ELECTRON_SESSION_TOKEN · 140 ROOT base
**Type ID:** `electron_session_token`  
**Causal Action:** `extract_chrome`  
**Causal Signal:** 0.83

| Field | Detail |
|---|---|
| **What it is** | Chromium-based Electron app session storage containing bearer tokens and SSO cookies for desktop SaaS clients |
| **Where it lives** | Slack: `%APPDATA%\Slack\Partitions\*\Local Storage\leveldb\` · Teams: `%APPDATA%\Microsoft\Teams\Local Storage\leveldb\` · Discord: `%APPDATA%\discord\Local Storage\leveldb\` · TradingView: `%APPDATA%\TradingView\Local Storage\leveldb\` · IBKR: `%LOCALAPPDATA%\IBKR\` |
| **What it grants** | Active workspace sessions — Slack token allows read/write to all channels, file access, DM history; Teams token grants M365 SSO; brokerage tokens allow trade execution API calls |
| **EDR detection** | Low — standard Electron app profile path access |
| **Privilege needed** | User |
| **Exploitation path** | Extract Slack `xoxs-` / `xoxp-` token from LevelDB → `slack-history-export` → enumerate all messages; or Teams token → Microsoft Graph API access to calendar, email, files. Full operational workflow in [playbook_electron_tokens.md](docs/playbooks/playbook_electron_tokens.md). |
| **Downstream artifacts** | Sensitive DMs containing credentials; shared files with financial data; calendar intel for social engineering |
| **Collector** | `ghost_layer_financial._collect_electron_sessions()` |
| **Classifier keyword** | `electron_session`, `slack_session`, `teams_session`, `electron` |

---

### 🟠 LSA_SYSTEM_SECRET · 150 ROOT base
**Type ID:** `lsa_system_secret`  
**Causal Action:** `extract_lsa`  
**Causal Signal:** 0.88

| Field | Detail |
|---|---|
| **What it is** | Windows LSA secrets — DPAPI master keys, cached domain credentials (msCachev2), service account passwords stored in LSA, NTLM hashes from LSASS memory |
| **Where it lives** | `HKLM\SECURITY\Policy\Secrets` (registry, SYSTEM-only) · LSASS process memory · SAM database · `%SYSTEMROOT%\System32\config\SAM` |
| **What it grants** | NTLM hashes for Pass-the-Hash lateral movement; plaintext service account passwords; msCachev2 for offline cracking and domain re-entry; DPAPI master key for decrypting all user secrets |
| **EDR detection** | High — LSASS memory reads trigger most EDR products; registry approach is stealthier |
| **Privilege needed** | SYSTEM |
| **Exploitation path** | LSASS dump → Mimikatz `sekurlsa::logonpasswords` → extract NTLM hashes → PTH to domain controller; or `lsadump::secrets` for service account plaintext passwords |
| **Downstream artifacts** | Domain admin credentials; service account passwords; DPAPI master keys (decrypt all browser passwords offline) |
| **Collector** | `GhostLayer._extract_lsa_secrets()` |
| **Classifier keyword** | `lsa`, `sam`, `ntds`, `kerberos`, `tgt`, `lsass`, `ntlm` |

---

### 🟠 SSH_PRIVATE_KEY · 120 ROOT base
**Type ID:** `ssh_private_key`  
**Causal Action:** `extract_ssh`  
**Causal Signal:** 0.82

| Field | Detail |
|---|---|
| **What it is** | RSA, Ed25519, ECDSA private keys used for passwordless SSH authentication to servers, cloud instances, and git remotes |
| **Where it lives** | `~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, `~/.ssh/id_ecdsa` · `~/.ssh/config` (reveals hostnames/aliases) · `~/.ssh/authorized_keys` (inverse — shows what this key unlocks) |
| **What it grants** | Passwordless lateral movement to every server where the public key is installed; persistent foothold that survives password rotation |
| **EDR detection** | Low — file read; SSH connection itself may be logged |
| **Privilege needed** | User |
| **Exploitation path** | Copy `id_rsa` → `ssh -i id_rsa user@target` → enumerate `~/.ssh/known_hosts` for all reachable hosts → recursive lateral movement |
| **Downstream artifacts** | Server file systems; deployment pipeline access; git repository access via GitHub/GitLab deploy keys |
| **Collector** | `GhostLayer._scan_ssh_keys()` + `_extract_ssh_keys_advanced()` |
| **Classifier keyword** | `ssh`, `id_rsa`, `id_ed25519`, `authorized_keys` |

---

### 🟠 DB_CREDENTIAL · 110 ROOT base
**Type ID:** `db_credential`  
**Causal Action:** `extract_db`  
**Causal Signal:** 0.78

| Field | Detail |
|---|---|
| **What it is** | Database connection strings, DSNs, and stored credentials for PostgreSQL, MySQL, MongoDB, Redis, SQL Server, Oracle |
| **Where it lives** | App config files (`.env`, `database.yml`, `settings.py`, `appsettings.json`) · `~/.pgpass` · `~/.my.cnf` · Kubernetes secrets · Docker Compose files |
| **What it grants** | Direct database query access — customer PII, transaction records, password hashes, API keys stored in DB tables |
| **EDR detection** | None for file read; network DB connection may be logged |
| **Privilege needed** | User |
| **Exploitation path** | Extract DSN → connect directly → `SELECT * FROM users` → extract password hashes → crack offline; or `mysqldump` for full backup |
| **Downstream artifacts** | User PII; hashed password tables; stored payment tokens; session table |
| **Collector** | `GhostLayer._extract_database_credentials()` |
| **Classifier keyword** | `postgres`, `mysql`, `redis`, `mongo`, `db_cred`, `dsn` |

---

### 🟠 FINANCIAL_DOCUMENT · 95 ROOT base
**Type ID:** `financial_document`  
**Causal Action:** `extract_email`  
**Causal Signal:** 0.70

| Field | Detail |
|---|---|
| **What it is** | Unstructured financial documents: wire transfer instructions, vendor invoice PDFs, voided checks, cap tables, equity grant schedules, payroll export spreadsheets, board minutes with compensation data |
| **Where it lives** | `Downloads`, `Desktop`, `Documents`, `OneDrive`, `Dropbox`, Google Drive sync folders; matching patterns: `*wire*instruction*`, `*invoice*`, `*cap_table*`, `*payroll_*.xlsx` |
| **What it grants** | Wire routing numbers + account numbers for BEC (Business Email Compromise) fraud; executive compensation details for spear-phishing; vendor relationships for supply chain social engineering; cap table for identifying high-value individuals |
| **EDR detection** | None |
| **Privilege needed** | User |
| **Exploitation path** | Extract wire instruction PDF → impersonate vendor via email → send updated "new banking details" to AP team → redirect future wire payments |
| **Downstream artifacts** | Bank account details; vendor contact list; executive equity details |
| **Collector** | `ghost_layer_financial._collect_financial_documents()` |
| **Classifier keyword** | `wire_instruction`, `cap_table`, `payroll_export`, `invoice`, `financial_doc`, `voided_check`, `direct_deposit` |

---

### 🟡 GIT_TOKEN · 65 ROOT base
**Type ID:** `git_token`  
**Causal Action:** `extract_git`  
**Causal Signal:** 0.62

| Field | Detail |
|---|---|
| **What it is** | Git hosting platform personal access tokens and OAuth tokens for GitHub, GitLab, Bitbucket, Azure DevOps |
| **Where it lives** | `~/.git-credentials` · `~/.config/gh/hosts.yml` · `%APPDATA%\GitHub Desktop\` · `.gitconfig` with token inline |
| **What it grants** | Read/write access to all private repositories the user can access; ability to enumerate org members, webhooks, and CI/CD secrets; potential to push backdoored code |
| **EDR detection** | None |
| **Privilege needed** | User |
| **Exploitation path** | `gh api /user/repos` → list all accessible repos → `git clone` all private repos → scan for secrets in commit history; or add malicious GitHub Action / GitLab CI pipeline |
| **Downstream artifacts** | Source code with embedded secrets; CI/CD pipeline variables; deploy keys to production infrastructure |
| **Collector** | `GhostLayer._extract_git_credentials()` |
| **Classifier keyword** | `git`, `github`, `gitlab`, `pat_token` |

---

### 🟡 BROWSER_PASSWORD · 45 ROOT base
**Type ID:** `browser_password`  
**Causal Action:** `extract_chrome`  
**Causal Signal:** 0.58

| Field | Detail |
|---|---|
| **What it is** | Saved login credentials from Chrome, Edge, Brave, Firefox password managers — DPAPI-encrypted on Windows |
| **Where it lives** | Chrome: `%LOCALAPPDATA%\Google\Chrome\User Data\Default\Login Data` (SQLite) · Firefox: `%APPDATA%\Mozilla\Firefox\Profiles\*.default\logins.json` + `key4.db` |
| **What it grants** | Plaintext passwords for all sites the user has saved — banking portals, corporate SSO, personal accounts |
| **EDR detection** | Low for Chrome (copy + DPAPI); higher for hooking DPAPI calls |
| **Privilege needed** | User (own profile); Admin (other profiles) |
| **Exploitation path** | Copy `Login Data` → decrypt with `CryptUnprotectData` using user session → dump all username:password pairs → credential stuffing across other platforms |
| **Downstream artifacts** | Personal banking passwords; VPN credentials; email passwords for further pivoting |
| **Collector** | `GhostLayer._extract_chrome_passwords()` + `_extract_firefox_passwords()` + `_extract_brave_browser_data()` |
| **Classifier keyword** | `chrome`, `edge`, `firefox`, `browser`, `cookie`, `password` |

---

### 🟡 EMAIL_CREDENTIAL · 35 ROOT base
**Type ID:** `email_credential`  
**Causal Action:** `extract_email`  
**Causal Signal:** 0.45

| Field | Detail |
|---|---|
| **What it is** | Email client authentication credentials: IMAP/SMTP passwords, Outlook profile encrypted passwords, Gmail app-specific passwords, Thunderbird stored credentials |
| **Where it lives** | Outlook: `HKCU\Software\Microsoft\Office\*\Outlook\Profiles\` (registry) · Thunderbird: `%APPDATA%\Thunderbird\Profiles\*\logins.json` · macOS Keychain |
| **What it grants** | Full email account access — read all inbox/sent history, send as the victim, access password reset emails, extract business intelligence from email content |
| **EDR detection** | None for file/registry read |
| **Privilege needed** | User |
| **Exploitation path** | Extract IMAP credentials → authenticate → search inbox for bank notifications, 2FA codes, password resets → account takeover chain |
| **Downstream artifacts** | 2FA recovery codes; bank statement emails; credential reset emails; contact list for phishing |
| **Collector** | `GhostLayer._extract_email_credentials()` |
| **Classifier keyword** | `email`, `imap`, `smtp`, `outlook`, `gmail` |

---

### 🟢 GENERIC_SECRET · 15 ROOT base
**Type ID:** `generic_secret`  
**Causal Action:** `extract_generic`  
**Causal Signal:** 0.30

| Field | Detail |
|---|---|
| **What it is** | Catch-all for unclassified secrets: API keys of unknown type, password manager exports, random token files, `.pem` certificates without matching private key |
| **Where it lives** | Anywhere — classified only when no specific pattern matches |
| **What it grants** | Variable — depends on the specific key; treated as low-value until manually reviewed |
| **Exploitation path** | Manual review of vault contents; run pattern matching against known service formats |
| **Collector** | Fallback in `ArtifactValuationTable.classify()` |

---

## 3. Effective Price Range (at Maximum Multipliers)

Domain Controller + System privilege = 1.60 × 1.50 = **2.40× base**

| Artifact | Base ROOT | Max ROOT (DC + SYSTEM) |
|---|:---:|:---:|
| Azure PRT Cache | 300 | **720** |
| Web3 Developer Secret | 290 | **696** |
| Payment Gateway Key | 275 | **660** |
| Wallet Seed | 250 | **600** |
| Banking Batch File | 220 | **528** |
| ERP Accounting Config | 190 | **456** |
| Cloud IAM Token | 180 | **432** |
| Browser Web Storage | 160 | **384** |
| Electron Session Token | 140 | **336** |
| LSA System Secret | 150 | **360** |
| SSH Private Key | 120 | **288** |
| DB Credential | 110 | **264** |
| Financial Document | 95 | **228** |
| Git Token | 65 | **156** |
| Browser Password | 45 | **108** |
| Email Credential | 35 | **84** |
| Generic Secret | 15 | **36** |

---

## 4. TargetIdentity Priority Score Model

```
priority_score = (20 + asset_bonus + min(40, extracted_root × 0.05))
                 × priv_mult × role_mult × edr_friction

Asset bonuses:
  has_wallets        +35
  has_cloud_tokens   +25
  has_ssh_keys       +15
  has_db_creds       +15
  has_browser_store  + 8

EDR friction: max(0.55, 1.0 - 0.12 × num_edr_products)
```

**Example:** Admin domain controller with wallets + cloud tokens + 1 EDR product:

```
(20 + 35 + 25 + 500×0.05) × 1.35 × 1.60 × max(0.55, 0.88)
= (20 + 60 + 25) × 1.35 × 1.60 × 0.88
= 105 × 1.35 × 1.60 × 0.88
= 199.6 priority score
```

---

## 5. Collector Pipeline Summary

| Phase | Module | Collectors | Artifact Types |
|---|---|---|---|
| **P1** | `ghost_layer.py` | Chrome passwords, Firefox, Environment vars | `browser_password`, `generic_secret` |
| **P2** | `ghost_layer.py` | LSA secrets, Credential Manager, Browser cookies | `lsa_system_secret`, `browser_password` |
| **P3** | `ghost_layer.py` | Kubernetes, Database, SSH (advanced), Git, Docker, Cloud providers, Kerberos | `db_credential`, `ssh_private_key`, `git_token`, `cloud_iam_token` |
| **P4** | `ghost_layer.py` | RDP creds, VPN profiles, Conditional Access, MFA info | `generic_secret` |
| **CRIT** | `ghost_layer.py` | Google credentials, Brave (passwords + wallet), Safari, Social media, Email, Browser extensions | `cloud_iam_token`, `wallet_seed`, `browser_password`, `email_credential` |
| **P5-A** | `ghost_layer_financial.py` | Browser IndexedDB / LocalStorage | `browser_web_storage` |
| **P5-B** | `ghost_layer_financial.py` | Electron app sessions | `electron_session_token` |
| **P5-C** | `ghost_layer_financial.py` | Azure AD PRT / WAM cache | `azure_prt_cache` |
| **P5-D** | `ghost_layer_financial.py` | QuickBooks, SAP GUI, Sage | `erp_accounting_config` |
| **P5-E** | `ghost_layer_financial.py` | NACHA ACH, SWIFT, SEPA XML | `banking_batch_file` |
| **P5-F** | `ghost_layer_financial.py` | `.env` API keys, Foundry keystores, Solana id.json | `web3_developer_secret` |
| **P5-G** | `ghost_layer_financial.py` | Stripe, PayPal, Square, Plaid, Twilio in source | `payment_gateway_key` |
| **P5-H** | `ghost_layer_financial.py` | Ledger Live, Trezor, Bitcoin Core, Electrum, Exodus | `wallet_seed` |
| **P5-I** | `ghost_layer_financial.py` | Wire instructions, cap tables, payroll exports | `financial_document` |

---

## 6. CausalGraph Integration

Each artifact type maps to a `CausalGraph` extraction action node. When an artifact
is ingested, an EMA blend shifts the node's `data_value` edge weight toward the
artifact's `causal_value_signal`:

```
new_weight = 0.75 × current_weight + 0.25 × causal_value_signal
```

This means: **the more high-value artifacts are extracted from a given technique,
the higher the CausalSimulator scores that technique in future decisions.**

| CausalGraph Node | Artifact Types |
|---|---|
| `extract_crypto` | `wallet_seed`, `web3_developer_secret` |
| `extract_cloud` | `cloud_iam_token`, `azure_prt_cache`, `payment_gateway_key`, `banking_batch_file` |
| `extract_lsa` | `lsa_system_secret` |
| `extract_ssh` | `ssh_private_key` |
| `extract_db` | `db_credential`, `erp_accounting_config` |
| `extract_git` | `git_token` |
| `extract_chrome` | `browser_password`, `browser_web_storage`, `electron_session_token` |
| `extract_email` | `email_credential`, `financial_document` |
| `extract_generic` | `generic_secret` |

---

## 7. Code Reference

| Component | File |
|---|---|
| `ArtifactType` enum | [`recon/target_identity.py`](recon/target_identity.py) |
| `ArtifactValuationTable` | [`recon/target_identity.py`](recon/target_identity.py) |
| `TargetIdentity` + registry | [`recon/target_identity.py`](recon/target_identity.py) |
| Phase 1–4 + Critical collectors | [`ghost_layer.py`](ghost_layer.py) |
| Phase 5 financial collectors | [`ghost_layer_financial.py`](ghost_layer_financial.py) |
| BlackVault ingest hook | [`black_vault.py`](black_vault.py) (`store()`) |
| CausalGraph nodes | [`causal_graph.py`](causal_graph.py) (`create_credential_extraction_graph()`) |
| BEL credit | [`economic/bel_layer.py`](economic/bel_layer.py) (`Ledger.credit()`) |
| Tests | [`tests/test_target_identity.py`](tests/test_target_identity.py) · [`tests/test_ghost_layer_financial.py`](tests/test_ghost_layer_financial.py) |
