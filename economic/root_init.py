# init_root.py

import asyncio, json, uuid, time, os
import redis.asyncio as redis
from bel_layer import Ledger, LedgerEntry  # adjust import to your path

# Always resolve manifest path relative to this script
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "root_tokenomics.yaml")  # or .json

def load_manifest(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def main():
    m = load_manifest(MANIFEST_PATH)
    r = redis.from_url("redis://localhost:6379")
    ledger = Ledger(r)

    treasury = m["accounting"]["wallet_roles"]["treasury"]
    fees     = m["accounting"]["wallet_roles"]["fees"]
    burn     = m["accounting"]["wallet_roles"]["burn"]

    # Create or load ledger map
    data = await ledger._load()

    def ensure(agent_id, balance=0.0):
        e = data.get(agent_id) or LedgerEntry(agent_id=agent_id)
        e.balance += balance
        e.last_seen = time.time()
        data[agent_id] = e

    # Seed wallets
    ensure(treasury, m["distribution"]["initial_mint_to_treasury"])
    ensure(fees, 0.0)
    ensure(burn, 0.0)

    # Airdrop/staking/liquidity pools live in treasury and move out as needed
    pools_total = (
        m["distribution"]["initial_airdrop_pool"]
        + m["distribution"]["initial_staking_rewards_pool"]
        + m["distribution"]["initial_liquidity_pool"]
    )
    ensure(treasury, pools_total)

    # Save ledger and set system state baselines
    await ledger._save(data)
    state = await ledger._load_system_state()
    state.total_supply = sum(e.balance + e.stake for e in data.values())
    await ledger._save_system_state(state)

    print("Root initialized.")
    print("Treasury balance:", data[treasury].balance)

if __name__ == "__main__":
    asyncio.run(main())
