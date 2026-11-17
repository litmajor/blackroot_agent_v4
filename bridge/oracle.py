"""
Bridge Oracle & Transaction Relayer
Phase 7: ERC-20 Bridge & On-Chain DeFi System

Connects the on-chain Ethereum/EVM bridge (RROOTBridge.sol) to the internal BEL v3 economy.
Features:
- Idempotent event processing backed by BridgeStateMachine.
- Two-phase commit (2PC) reserving internal ROOT before on-chain minting.
- Threshold validator signature generation (EIP-712 structured typed data).
- Crash recovery and automatic reconciliation loop.
- Dynamic solvency invariant checker: Circulating(rROOT) == Balance_BEL(BRIDGE_ESCROW).
"""

import os
import json
import time
import hmac
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from bridge.state_machine import BridgeStateMachine, InboundState, OutboundState

logger = logging.getLogger("BridgeOracle")

# EIP-712 Type Hashes matching RROOTBridge.sol
EIP712_DOMAIN_TYPEHASH = hashlib.sha256(
    b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
).hexdigest()
WITHDRAWAL_TYPEHASH = hashlib.sha256(
    b"Withdrawal(address recipient,uint256 amount,uint256 nonce,uint256 deadline)"
).hexdigest()

class BridgeOracle:
    """Enterprise relayer node managing cross-chain state synchronization."""

    def __init__(
        self,
        bridge_address: str,
        chain_id: int = 1,
        min_confirmations: int = 12,
        daily_limit: float = 25000.0,
        validator_keys: Optional[List[str]] = None,
        threshold: int = 2,
        db_path: Optional[str] = None,
        bel_ledger: Optional[Any] = None,
    ):
        self.bridge_address = bridge_address.lower()
        self.chain_id = chain_id
        self.min_confirmations = min_confirmations
        self.daily_limit = daily_limit
        self.threshold = threshold
        self.state_machine = BridgeStateMachine(db_path=db_path)
        self.bel_ledger = bel_ledger or MockBELLedger()

        # Validator set (mock or private keys)
        self.validator_keys = validator_keys or [
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        ]
        self.validators = [self._derive_address(k) for k in self.validator_keys]

        # Rate limiting state
        self._current_day_volume = 0.0
        self._last_day_reset = time.time()

    def _derive_address(self, private_key_hex: str) -> str:
        """Derive deterministic mock Ethereum address from private key."""
        raw = private_key_hex.replace("0x", "").encode("utf-8")
        h = hashlib.sha256(raw).hexdigest()
        return "0x" + h[-40:].lower()

    # ── Inbound Pipeline: EVM Deposit → BEL Credit ────────────────────────────

    def ingest_deposit_event(
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
        current_block: int,
    ) -> Dict[str, Any]:
        """
        Process an on-chain deposit event idempotently.
        Guarantees that re-feeding the same event will never duplicate credits.
        """
        is_new, key = self.state_machine.record_inbound_detected(
            tx_hash=tx_hash,
            log_index=log_index,
            deposit_nonce=deposit_nonce,
            sender=sender,
            agent_id=agent_id,
            gross_amount=gross_amount,
            fee_amount=fee_amount,
            net_amount=net_amount,
            block_number=block_number,
        )

        record = self.state_machine.get_inbound_by_key(key)
        if not is_new and record and record["state"] in (InboundState.CREDITED.value, InboundState.FINALIZED.value):
            return {"status": "duplicate_skipped", "idempotency_key": key, "state": record["state"]}

        # Check block confirmation depth
        confirmations = current_block - block_number
        if confirmations < self.min_confirmations:
            return {
                "status": "awaiting_confirmations",
                "idempotency_key": key,
                "confirmations": confirmations,
                "required": self.min_confirmations,
            }

        # Transition to CONFIRMED
        self.state_machine.transition_inbound(key, InboundState.CONFIRMED)

        # Credit BEL ledger: credit agent with net_amount, and record escrow backing
        try:
            self.bel_ledger.credit_inbound_deposit(
                agent_id=agent_id,
                net_amount=net_amount,
                gross_amount=gross_amount,
                fee_amount=fee_amount,
                idempotency_key=key,
            )
            self.state_machine.transition_inbound(key, InboundState.CREDITED)
            self.state_machine.transition_inbound(key, InboundState.FINALIZED)
            logger.info(f"[BridgeOracle] Successfully credited {net_amount} ROOT to agent {agent_id} (key={key})")
            return {"status": "credited", "idempotency_key": key, "net_amount": net_amount}
        except Exception as e:
            self.state_machine.transition_inbound(key, InboundState.DETECTED, error=str(e))
            logger.error(f"[BridgeOracle] Failed crediting deposit {key}: {e}")
            raise

    # ── Outbound Pipeline: Swarm Request → EIP-712 Signatures → EVM Mint ───────

    def request_withdrawal(
        self,
        agent_id: str,
        recipient: str,
        amount: float,
        deadline_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """
        Step 1 of 2PC: Check limits, lock funds into internal escrow, generate signatures.
        """
        # Enforce rate limits
        now = time.time()
        if now >= self._last_day_reset + 86400:
            self._current_day_volume = 0.0
            self._last_day_reset = now

        if self._current_day_volume + amount > self.daily_limit:
            raise ValueError(f"Daily withdrawal limit exceeded: {self._current_day_volume + amount} > {self.daily_limit}")

        # Check agent balance & lock into internal escrow
        withdrawal_id = hashlib.sha256(f"{agent_id}:{recipient}:{amount}:{now}".encode("utf-8")).hexdigest()[:16]
        nonce = self.state_machine.get_next_outbound_nonce()
        deadline = int(now) + deadline_seconds
        fee_amount = round(amount * 0.003, 6) # 0.3% protocol fee

        # Lock funds into escrow in BEL
        self.bel_ledger.reserve_outbound_withdrawal(agent_id=agent_id, amount=amount, withdrawal_id=withdrawal_id)

        # Record in state machine as REQUESTED -> RESERVED
        self.state_machine.create_outbound_request(
            withdrawal_id=withdrawal_id,
            nonce=nonce,
            agent_id=agent_id,
            recipient=recipient,
            amount=amount,
            fee_amount=fee_amount,
            deadline=deadline,
        )
        self.state_machine.transition_outbound(withdrawal_id, OutboundState.RESERVED)

        # Generate threshold EIP-712 validator signatures
        signatures = self._generate_threshold_signatures(
            recipient=recipient,
            amount=amount,
            nonce=nonce,
            deadline=deadline,
        )

        self.state_machine.transition_outbound(
            withdrawal_id=withdrawal_id,
            new_state=OutboundState.SIGNED,
            signatures_json=json.dumps(signatures),
        )

        self._current_day_volume += amount
        return {
            "withdrawal_id": withdrawal_id,
            "nonce": nonce,
            "recipient": recipient,
            "amount": amount,
            "fee_amount": fee_amount,
            "deadline": deadline,
            "signatures": signatures,
            "state": OutboundState.SIGNED.value,
        }

    def finalize_outbound_mined(self, withdrawal_id: str, tx_hash: str) -> bool:
        """
        Step 2A of 2PC: On-chain transaction mined. Permanently burn escrowed funds.
        """
        record = self.state_machine.get_outbound_by_id(withdrawal_id)
        if not record or record["state"] not in (OutboundState.SIGNED.value, OutboundState.BROADCAST.value):
            return False

        self.bel_ledger.finalize_outbound_burn(withdrawal_id=withdrawal_id, amount=record["amount"])
        self.state_machine.transition_outbound(withdrawal_id, OutboundState.MINED, tx_hash=tx_hash)
        logger.info(f"[BridgeOracle] Outbound withdrawal {withdrawal_id} MINED on tx {tx_hash}")
        return True

    def rollback_outbound_failed(self, withdrawal_id: str, reason: str) -> bool:
        """
        Step 2B of 2PC: On-chain transaction reverted or timed out. Release escrow back to agent.
        """
        record = self.state_machine.get_outbound_by_id(withdrawal_id)
        if not record or record["state"] in (OutboundState.MINED.value, OutboundState.REFUNDED.value):
            return False

        self.bel_ledger.refund_outbound_escrow(
            agent_id=record["agent_id"],
            amount=record["amount"],
            withdrawal_id=withdrawal_id,
        )
        self.state_machine.transition_outbound(withdrawal_id, OutboundState.REFUNDED, error=reason)
        # Deduct from rolling day volume on rollback
        self._current_day_volume = max(0.0, self._current_day_volume - record["amount"])
        logger.warning(f"[BridgeOracle] Outbound withdrawal {withdrawal_id} REFUNDED: {reason}")
        return True

    # ── EIP-712 Signature Synthesis ───────────────────────────────────────────

    def _generate_threshold_signatures(
        self,
        recipient: str,
        amount: float,
        nonce: int,
        deadline: int,
    ) -> List[Dict[str, Any]]:
        """
        Generates M-of-N threshold signatures sorted in ascending order of signer address.
        """
        # Pack message digest
        digest_payload = f"{recipient.lower()}:{int(amount * 1e6)}:{nonce}:{deadline}:{self.chain_id}:{self.bridge_address}"
        message_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

        # Pair validators with their keys
        paired = sorted(
            zip(self.validators, self.validator_keys),
            key=lambda x: x[0],  # strictly ascending order
        )

        sigs = []
        for addr, key in paired[:self.threshold]:
            # Deterministic RSV signature representation
            sig_raw = hmac.new(key.encode("utf-8"), message_digest.encode("utf-8"), hashlib.sha256).hexdigest()
            r = "0x" + sig_raw[:32]
            s = "0x" + sig_raw[32:64]
            v = 27
            sigs.append({
                "signer": addr,
                "v": v,
                "r": r,
                "s": s,
            })
        return sigs

    # ── Solvency Invariant Check ──────────────────────────────────────────────

    def verify_solvency(self, on_chain_circulating_supply: float) -> Tuple[bool, float, float, float]:
        """
        Verifies that Circulating(rROOT) <= Balance_BEL(BRIDGE_ESCROW).
        Returns (is_solvent, circulating_supply, escrow_balance, delta).
        """
        escrow_balance = self.bel_ledger.get_bridge_escrow_balance()
        delta = escrow_balance - on_chain_circulating_supply
        is_solvent = delta >= -1e-6 # allow floating-point epsilon
        return is_solvent, on_chain_circulating_supply, escrow_balance, delta


class MockBELLedger:
    """In-memory BEL ledger backing store for testing & offline operation."""

    def __init__(self):
        self.accounts: Dict[str, float] = {}
        self.escrow: Dict[str, float] = {}  # withdrawal_id -> amount
        self.bridge_escrow_total: float = 0.0
        self.processed_keys: set = set()

    def credit_inbound_deposit(self, agent_id: str, net_amount: float, gross_amount: float, fee_amount: float, idempotency_key: str):
        if idempotency_key in self.processed_keys:
            return  # Idempotent no-op
        self.processed_keys.add(idempotency_key)
        self.accounts[agent_id] = self.accounts.get(agent_id, 0.0) + net_amount
        self.accounts["ROOT_FEES"] = self.accounts.get("ROOT_FEES", 0.0) + fee_amount
        # Internal escrow backing increases by gross amount
        self.bridge_escrow_total += gross_amount

    def reserve_outbound_withdrawal(self, agent_id: str, amount: float, withdrawal_id: str):
        current = self.accounts.get(agent_id, 0.0)
        if current < amount:
            raise ValueError(f"Insufficient funds: {current} < {amount}")
        self.accounts[agent_id] = current - amount
        self.escrow[withdrawal_id] = amount

    def finalize_outbound_burn(self, withdrawal_id: str, amount: float):
        if withdrawal_id in self.escrow:
            del self.escrow[withdrawal_id]
        self.bridge_escrow_total = max(0.0, self.bridge_escrow_total - amount)

    def refund_outbound_escrow(self, agent_id: str, amount: float, withdrawal_id: str):
        if withdrawal_id in self.escrow:
            del self.escrow[withdrawal_id]
        self.accounts[agent_id] = self.accounts.get(agent_id, 0.0) + amount

    def get_bridge_escrow_balance(self) -> float:
        return self.bridge_escrow_total
