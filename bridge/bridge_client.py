"""
Bridge Client SDK
Phase 7: ERC-20 Bridge & On-Chain DeFi System

High-level interface for Blackroot agents, AccountantAgent, and operators
to interact with the cross-chain bridge and query real-time solvency.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from bridge.oracle import BridgeOracle
from bridge.state_machine import InboundState, OutboundState

logger = logging.getLogger("BridgeClient")

class BridgeClient:
    """Client interface for managing cross-chain deposits, withdrawals, and telemetry."""

    def __init__(self, oracle: BridgeOracle):
        self.oracle = oracle

    def deposit(
        self,
        tx_hash: str,
        log_index: int,
        deposit_nonce: int,
        sender: str,
        agent_id: str,
        amount: float,
        block_number: int,
        current_block: int,
        fee_rate: float = 0.003,
    ) -> Dict[str, Any]:
        """Submit an on-chain deposit event to be processed by the bridge oracle."""
        fee_amount = round(amount * fee_rate, 6)
        net_amount = round(amount - fee_amount, 6)

        return self.oracle.ingest_deposit_event(
            tx_hash=tx_hash,
            log_index=log_index,
            deposit_nonce=deposit_nonce,
            sender=sender,
            agent_id=agent_id,
            gross_amount=amount,
            fee_amount=fee_amount,
            net_amount=net_amount,
            block_number=block_number,
            current_block=current_block,
        )

    def withdraw(
        self,
        agent_id: str,
        recipient: str,
        amount: float,
        deadline_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """Request an outbound withdrawal from the swarm to an EVM address."""
        return self.oracle.request_withdrawal(
            agent_id=agent_id,
            recipient=recipient,
            amount=amount,
            deadline_seconds=deadline_seconds,
        )

    def confirm_withdrawal_mined(self, withdrawal_id: str, tx_hash: str) -> bool:
        """Confirm that the withdrawal transaction was successfully mined on-chain."""
        return self.oracle.finalize_outbound_mined(withdrawal_id=withdrawal_id, tx_hash=tx_hash)

    def cancel_or_fail_withdrawal(self, withdrawal_id: str, reason: str) -> bool:
        """Rollback and refund an outbound withdrawal if reverted or timed out."""
        return self.oracle.rollback_outbound_failed(withdrawal_id=withdrawal_id, reason=reason)

    def get_transaction_status(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Look up status by inbound idempotency key or outbound withdrawal_id."""
        inbound = self.oracle.state_machine.get_inbound_by_key(identifier)
        if inbound:
            return {"type": "inbound", **inbound}
        outbound = self.oracle.state_machine.get_outbound_by_id(identifier)
        if outbound:
            return {"type": "outbound", **outbound}
        return None

    def get_solvency_report(self, on_chain_circulating_supply: float) -> Dict[str, Any]:
        """Generate a complete solvency & backing audit report."""
        is_solvent, circulating, escrow, delta = self.oracle.verify_solvency(on_chain_circulating_supply)
        return {
            "is_solvent": is_solvent,
            "circulating_rroot": circulating,
            "escrow_root": escrow,
            "backing_delta": delta,
            "backing_ratio": (escrow / circulating) if circulating > 0 else 1.0,
            "daily_volume_used": self.oracle._current_day_volume,
            "daily_volume_limit": self.oracle.daily_limit,
        }
