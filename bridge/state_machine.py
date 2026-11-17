"""
Bridge Transactional State Machine & Crash Recovery Engine
Phase 7: ERC-20 Bridge & On-Chain DeFi System

Guarantees cross-system transactional idempotency and crash resilience:
1. Inbound pipeline: DETECTED -> CONFIRMED -> CREDITED -> FINALIZED
2. Outbound pipeline: REQUESTED -> RESERVED -> SIGNED -> BROADCAST -> MINED (or FAILED -> REFUNDED)
3. SQLite WAL mode with ACID transactions and unique idempotency keys.
"""

import os
import sqlite3
import hashlib
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("BridgeStateMachine")

class InboundState(str, Enum):
    DETECTED = "DETECTED"
    CONFIRMED = "CONFIRMED"
    CREDITED = "CREDITED"
    FINALIZED = "FINALIZED"
    REJECTED = "REJECTED"

class OutboundState(str, Enum):
    REQUESTED = "REQUESTED"
    RESERVED = "RESERVED"
    SIGNED = "SIGNED"
    BROADCAST = "BROADCAST"
    MINED = "MINED"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"

class BridgeStateMachine:
    """ACID-compliant persistent state store for cross-chain transactions."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "bridge_state.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS inbound_transactions (
                    idempotency_key TEXT PRIMARY KEY,
                    tx_hash TEXT NOT NULL,
                    log_index INTEGER NOT NULL,
                    deposit_nonce INTEGER NOT NULL,
                    sender TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    gross_amount REAL NOT NULL,
                    fee_amount REAL NOT NULL,
                    net_amount REAL NOT NULL,
                    block_number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS outbound_transactions (
                    withdrawal_id TEXT PRIMARY KEY,
                    nonce INTEGER UNIQUE NOT NULL,
                    agent_id TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    amount REAL NOT NULL,
                    fee_amount REAL NOT NULL,
                    net_amount REAL NOT NULL,
                    deadline INTEGER NOT NULL,
                    signatures_json TEXT,
                    tx_hash TEXT,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_inbound_state ON inbound_transactions(state);
                CREATE INDEX IF NOT EXISTS idx_outbound_state ON outbound_transactions(state);
            """)

    # ── Inbound Operations ────────────────────────────────────────────────────

    def compute_inbound_idempotency_key(self, tx_hash: str, log_index: int) -> str:
        raw = f"{tx_hash.lower()}:{log_index}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def record_inbound_detected(
        self,
        tx_hash: str,
        log_index: int,
        deposit_nonce: int,
        sender: str,
        agent_id: str,
        gross_amount: float,
        fee_amount: float,
        net_amount: float,
        block_number: int,
    ) -> Tuple[bool, str]:
        """Record a detected deposit log. Returns (is_new, idempotency_key)."""
        key = self.compute_inbound_idempotency_key(tx_hash, log_index)
        now = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO inbound_transactions 
                    (idempotency_key, tx_hash, log_index, deposit_nonce, sender, agent_id, 
                     gross_amount, fee_amount, net_amount, block_number, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, tx_hash.lower(), log_index, deposit_nonce, sender.lower(),
                        agent_id, gross_amount, fee_amount, net_amount, block_number,
                        InboundState.DETECTED.value, now, now
                    ),
                )
            return True, key
        except sqlite3.IntegrityError:
            # Already exists (idempotent duplicate detection)
            return False, key

    def transition_inbound(self, idempotency_key: str, new_state: InboundState, error: Optional[str] = None) -> bool:
        """Atomically transition inbound transaction state."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE inbound_transactions
                SET state = ?, updated_at = ?, error_message = ?
                WHERE idempotency_key = ?
                """,
                (new_state.value, time.time(), error, idempotency_key),
            )
            return cur.rowcount > 0

    def get_inbound_by_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM inbound_transactions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def get_inbound_by_state(self, state: InboundState) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM inbound_transactions WHERE state = ? ORDER BY block_number ASC",
                (state.value,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Outbound Operations ───────────────────────────────────────────────────

    def create_outbound_request(
        self,
        withdrawal_id: str,
        nonce: int,
        agent_id: str,
        recipient: str,
        amount: float,
        fee_amount: float,
        deadline: int,
    ) -> bool:
        """Record a new outbound withdrawal request."""
        now = time.time()
        net_amount = amount - fee_amount
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO outbound_transactions
                    (withdrawal_id, nonce, agent_id, recipient, amount, fee_amount, net_amount,
                     deadline, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        withdrawal_id, nonce, agent_id, recipient.lower(), amount,
                        fee_amount, net_amount, deadline, OutboundState.REQUESTED.value,
                        now, now
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def transition_outbound(
        self,
        withdrawal_id: str,
        new_state: OutboundState,
        tx_hash: Optional[str] = None,
        signatures_json: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Atomically advance outbound state with optional tx_hash / signatures."""
        with self._get_connection() as conn:
            updates = ["state = ?", "updated_at = ?"]
            params: List[Any] = [new_state.value, time.time()]

            if tx_hash is not None:
                updates.append("tx_hash = ?")
                params.append(tx_hash)
            if signatures_json is not None:
                updates.append("signatures_json = ?")
                params.append(signatures_json)
            if error is not None:
                updates.append("error_message = ?")
                params.append(error)

            params.append(withdrawal_id)
            sql = f"UPDATE outbound_transactions SET {', '.join(updates)} WHERE withdrawal_id = ?"
            cur = conn.execute(sql, params)
            return cur.rowcount > 0

    def get_outbound_by_id(self, withdrawal_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM outbound_transactions WHERE withdrawal_id = ?",
                (withdrawal_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_outbound_by_state(self, state: OutboundState) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM outbound_transactions WHERE state = ? ORDER BY created_at ASC",
                (state.value,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_next_outbound_nonce(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(nonce) as max_nonce FROM outbound_transactions").fetchone()
            if row and row["max_nonce"] is not None:
                return row["max_nonce"] + 1
            return 1
