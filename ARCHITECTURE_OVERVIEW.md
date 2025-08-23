# Blackroot Agent v4: Comprehensive Architecture & Integration Guide

## Table of Contents
- [Blackroot Agent v4: Comprehensive Architecture \& Integration Guide](#blackroot-agent-v4-comprehensive-architecture--integration-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. High-Level Architecture Diagram](#2-high-level-architecture-diagram)
  - [3. Core Components](#3-core-components)
    - [3.1 BlackrootHost \& SwarmMesh](#31-blackroothost--swarmmesh)
    - [3.2 PeerLinkerAgent](#32-peerlinkeragent)
  - [4. Artifact System](#4-artifact-system)
  - [5. Agent System](#5-agent-system)
  - [6. Enrichment \& Recon](#6-enrichment--recon)
  - [7. Payloads \& Polyglot Builder](#7-payloads--polyglot-builder)
  - [8. Memory, Identity, and Replication](#8-memory-identity-and-replication)
  - [9. API \& Integration Points](#9-api--integration-points)
  - [10. Security, Anti-Bot, and Persistence](#10-security-anti-bot-and-persistence)
  - [11. Developer Northstar: Integration \& Extension](#11-developer-northstar-integration--extension)
  - [12. Appendix: Detailed Diagrams](#12-appendix-detailed-diagrams)
    - [12.1 Peer-to-Peer Mesh \& Artifact Flow](#121-peer-to-peer-mesh--artifact-flow)
    - [12.2 Module Registration \& Kernel Integration](#122-module-registration--kernel-integration)

---

## 1. Overview
Blackroot Agent v4 is a distributed, modular C2 mesh platform designed for robust, secure, and extensible operation. It features advanced artifact mobility, agent orchestration, enrichment, anti-bot/persistence payloads, and a resilient peer-to-peer mesh with strong cryptography and dynamic discovery.

---

## 2. High-Level Architecture Diagram

```
+-------------------+      +-------------------+      +-------------------+
|   BlackrootHost   |<---->|   SwarmMesh       |<---->|   PeerLinkerAgent |
+-------------------+      +-------------------+      +-------------------+
        |                        |                          |
        v                        v                          v
+-------------------+      +-------------------+      +-------------------+
|  Artifact System  |<---->|  Agents/Loader    |<---->|  Enrichment/Recon |
+-------------------+      +-------------------+      +-------------------+
        |                        |                          |
        v                        v                          v
+-------------------+      +-------------------+      +-------------------+
|  Memory/Identity  |<---->|  Replication      |<---->|  API/Endpoints    |
+-------------------+      +-------------------+      +-------------------+
```

---

## 3. Core Components

### 3.1 BlackrootHost & SwarmMesh
- **BlackrootHost**: Entrypoint, orchestrates agent/module registration, artifact TTL, and mesh lifecycle.
- **SwarmMesh**: Distributed mesh, manages peer discovery, artifact relay, broadcast, and analytics.

### 3.2 PeerLinkerAgent
- Handles peer-to-peer mesh, secure handshake, k-bucket DHT, message queue with TTL, socket cleanup, STUN/UPnP, and robust logging.
- Integrates with SwarmMesh for peer discovery and artifact mobility.

---

## 4. Artifact System
- **Artifact Mobility**: Secure transfer, relay, broadcast, remote fetch, deletion, versioning, ACLs, analytics, and TTL.
- **Persistence**: Redis-backed storage, TTL enforcement, versioned updates, and access control.
- **Integration**: Artifacts are registered and managed via SwarmMesh and BlackrootHost.

---

## 5. Agent System
- **Loader**: Dynamically loads all agent classes (defiler, infiltrator, sensor, etc.), registers with kernel.
- **Agent Classes**: Each agent (e.g., PeerLinkerAgent, Sensor, Replicator) encapsulates a specific mesh or enrichment function.
- **Module Registration**: Unified in main.py and loader.py for robust, error-free integration.

---

## 6. Enrichment & Recon
- **Enrichment**: Passive, week-2/3/7, JWT/GraphQL, anti-bot, and persistence payloads, all as class-based static methods.
- **Recon**: Advanced recon/scanner.py, canary, circuit breaker, checksum cache, PassivePlus static methods.
- **Integration**: Enrichment modules are invoked by agents and via API endpoints.

---

## 7. Payloads & Polyglot Builder
- **Payloads**: desktop.js, mobile.js, universal.js (week-7 anti-bot, WAF, persistence, rate-limiting, IndexedDB).
- **Polyglot Builder**: polyglot_builder.py uses esbuild and UUID stamping for unique, tamper-resistant payloads.

---

## 8. Memory, Identity, and Replication
- **Memory/Identity**: Redis-backed, with local cache, pub/sub, and robust error handling.
- **Replication**: Handles artifact and agent state replication across the mesh.
- **Integration**: Used by all agents for state, artifact, and peer management.

---

## 9. API & Integration Points
- **blackroot_c2_api.py**: REST/WS/Redis endpoints for artifact, peer, and enrichment management.
- **Endpoints**: execute_blob, artifact fetch, peer status, enrichment, and more.
- **Integration**: API is the main external interface for orchestration and monitoring.

---

## 10. Security, Anti-Bot, and Persistence
- **Security**: NaCl/cryptography for all peer and artifact comms, key rotation, and session management.
- **Anti-Bot**: Payloads include anti-bot, WAF, and rate-limiting logic.
- **Persistence**: IndexedDB, web-worker, and Redis for robust, tamper-resistant operation.

---

## 11. Developer Northstar: Integration & Extension
- **Adding Agents**: Implement new agent class in `agents/`, register in loader.py, and ensure kernel integration.
- **Extending Artifacts**: Add new artifact types in artifact system, update SwarmMesh and API endpoints.
- **Enrichment**: Add new enrichment static methods in enrichment modules, update scanner.py and API.
- **Payloads**: Add/modify payloads in payloads/, update polyglot_builder.py for new build logic.
- **Testing**: Use tests/ for all new modules; run with pytest or unittest.
- **Type Safety**: All modules are type-checked; mDNS/zeroconf static errors are silenced with # type: ignore where needed.

---

## 12. Appendix: Detailed Diagrams

### 12.1 Peer-to-Peer Mesh & Artifact Flow
```
+-------------------+      +-------------------+
|   PeerLinkerAgent |<---->|   PeerLinkerAgent |
+-------------------+      +-------------------+
        |                          |
        v                          v
+-------------------+      +-------------------+
|   Artifact Relay  |<---->|   Artifact Relay  |
+-------------------+      +-------------------+
        |                          |
        v                          v
+-------------------+      +-------------------+
|   Enrichment      |<---->|   Enrichment      |
+-------------------+      +-------------------+
```

### 12.2 Module Registration & Kernel Integration
```
main.py
   |
   v
loader.py (registers all agents)
   |
   v
kernel/core.py (instantiates agents, manages lifecycle)
```

---

**This document is the northstar for all Blackroot Agent v4 development. All new features, agents, and integrations should follow the patterns and interfaces described above.**
