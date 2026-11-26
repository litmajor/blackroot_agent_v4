# Playbook: Web3 Developer Secrets & Treasury Key Triage

## 1. Overview
* **Artifact Type**: `web3_developer_secret`
* **Base Value**: 290 ROOT
* **Severity**: Critical
* **Prerequisites**: User-level read access to developer repositories, workspace directories, or home configurations.

Web3 developer environments frequently retain raw, unencrypted cryptographic keys used to deploy smart contracts, fund gas wallets, or operate automated backend bots.

---

## 2. Artifact Locations
* **Local Repo Environment Configs**:
  - `repos/**/.env`, `.env.local`, `.env.production`
  - `hardhat.config.js`, `hardhat.config.ts`, `foundry.toml`
* **CLI Keystores & Keypairs**:
  - `~/.config/solana/id.json` (64-byte raw Solana keypair JSON)
  - `~/.foundry/keystores/` (Encrypted JSON keystores)
  - `~/.ethereum/keystore/`

---

## 3. Extraction & Identification

Blackroot's parser extracts key-value pairs matching:
- `DEPLOYER_PRIVATE_KEY` / `PRIVATE_KEY`: 32-byte hex string (64 characters).
- `INFURA_PROJECT_ID` / `ALCHEMY_API_KEY`: RPC endpoint tokens.
- `ETHERSCAN_API_KEY`: Smart contract source verification token.

---

## 4. Abuse & Liquidation Chain

### Step 1: Address Derivation & Balance Check
For any recovered private key:
1. Derive corresponding public address (`0x...`).
2. Query RPC node (using discovered Infura/Alchemy keys or public endpoints) across multiple EVM networks (Ethereum Mainnet, Arbitrum, Optimism, Polygon, Base):
   ```bash
   # Check native coin balance (ETH)
   cast balance 0x<DerivedAddress> --rpc-url https://eth.llamarpc.com
   ```
3. Query ERC-20 / ERC-721 token balances and contract ownership.

### Step 2: Contract Ownership & Privilege Inspection
1. Check if the address owns deployed contracts:
   - Proxy Admin contracts (allows upgrading bytecode to exfiltrate funds).
   - Treasury / Multi-sig contracts (Safe / Gnosis Safe signer).
   - Minting / Burn / Pausable roles.

### Step 3: Solana Keypair Inspection
If `~/.config/solana/id.json` is recovered:
```bash
solana-keygen pubkey ~/.config/solana/id.json
solana balance --keypair ~/.config/solana/id.json --url mainnet-beta
```
Directly sign transfers to secure collector vaults or sweep treasury SPL tokens.
