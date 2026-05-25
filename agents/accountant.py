#!/usr/bin/env python3
"""
ACC-LEDGER: The Accountant Agent
Financial Custodian & Economic Regulator

Core Purpose: Manages all financial flows, budgets, and economic health
of the multi-agent organism while providing transparent interface to Creator.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN
import threading
from enum import Enum
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'economic'))
from economic.wallet import WalletManager, ChainWallet, TransactionRequest
from economic.bel_layer import EconomicEngine, Market, RiskManager

class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal" 
    TRANSFER = "transfer"
    BUDGET_ALLOCATION = "budget_allocation"
    AGENT_SPEND = "agent_spend"
    PENALTY = "penalty"
    REWARD = "reward"
    FEE = "fee"

class TransactionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Transaction:
    tx_id: str
    timestamp: float
    tx_type: TransactionType
    from_agent: str
    to_agent: str
    amount: Decimal
    currency: str
    description: str
    status: TransactionStatus
    approval_required: bool = False
    creator_approved: bool = False
    hash_signature: str = ""
    
    def __post_init__(self):
        if not self.hash_signature:
            self.hash_signature = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate immutable hash for audit trail"""
        data = f"{self.tx_id}{self.timestamp}{self.from_agent}{self.to_agent}{self.amount}{self.description}"
        return hashlib.sha256(data.encode()).hexdigest()

@dataclass
class AgentWallet:
    agent_id: str
    balance: Decimal
    allocated_budget: Decimal
    spent_budget: Decimal
    credit_limit: Decimal
    last_activity: float
    status: str = "pending"  # pending until allocation executed, then active, suspended, frozen
    parent_agent: Optional[str] = None
    child_agents: Optional[List[str]] = None
    overspend_attempts: int = 0

    def __post_init__(self):
        if self.child_agents is None:
            self.child_agents = []
    
    @property
    def remaining_budget(self) -> Decimal:
        return self.allocated_budget - self.spent_budget
    
    @property
    def utilization_rate(self) -> float:
        if self.allocated_budget == 0:
            return 0.0
        return float(self.spent_budget / self.allocated_budget)

@dataclass
class BudgetPeriod:
    period_id: str
    start_time: float
    end_time: float
    total_allocated: Decimal
    total_spent: Decimal
    agent_allocations: Dict[str, Decimal]
    
    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now <= self.end_time

class ACCLedger:
    """
    The Accountant Agent - Financial Custodian & Economic Regulator
    SINGLETON: Only one instance per kernel lifetime to prevent ledger duplication
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, creator_id: str, initial_treasury: Decimal = Decimal('5000')):
        """Enforce singleton pattern - all creates return same instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ACCLedger, cls).__new__(cls)
                    # Mark as uninitialized so __init__ runs once
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, creator_id: str, initial_treasury: Decimal = Decimal('5000')):
        # Skip re-initialization on singleton returns
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.agent_id = "ACC-LEDGER"
        self.creator_id = creator_id
        self.version = "1.0.0"
        # Core Financial State
        self.treasury_balance = initial_treasury
        self.burned_total = Decimal('0')  # Track total burned Root
        self.agent_wallets: Dict[str, AgentWallet] = {}
        self.transaction_ledger: List[Transaction] = []
        self.budget_periods: List[BudgetPeriod] = []
        # Security & Safeguards
        self.failsafe_mode = False
        self.creator_approval_threshold = Decimal('100')  # Transactions above this need approval
        self.max_daily_spend_per_agent = Decimal('50')
        self.emergency_reserve_ratio = Decimal('0.2')  # 20% of treasury reserved
        # Monitoring & Alerts
        self.alerts: List[Dict] = []
        self.audit_log: List[Dict] = []
        self.economic_health_metrics: Dict = {}
        # Threading for continuous monitoring
        self.monitoring_active = True
        self.lock = threading.Lock()
        # Initialize system
        self._initialize_system()
        self._start_monitoring()
        self._initialized = True
    
    def _initialize_system(self):
        """Initialize the accounting system"""
        self._log_audit("SYSTEM_INIT", {
            "treasury_balance": str(self.treasury_balance),
            "creator_id": self.creator_id,
            "timestamp": time.time()
        })
        
        # Create system agent wallet
        self._create_agent_wallet("SYSTEM", Decimal('0'), Decimal('0'))
        print(f"🧩 ACC-LEDGER singleton initialized with treasury: {self.treasury_balance}")
    
    def _create_agent_wallet(self, agent_id: str, initial_balance: Decimal, budget: Decimal, parent_agent: Optional[str] = None) -> AgentWallet:
        """Create a new agent wallet"""
        wallet = AgentWallet(
                agent_id=agent_id,
                balance=initial_balance,
                allocated_budget=budget,
                spent_budget=Decimal('0'),
                credit_limit=budget * Decimal('1.1'),
                last_activity=time.time(),
                status="pending",
                parent_agent=parent_agent,
                child_agents=[],
                overspend_attempts=0
            )
        self.agent_wallets[agent_id] = wallet
        
        self._log_audit("WALLET_CREATED", {
            "agent_id": agent_id,
            "initial_balance": str(initial_balance),
            "budget": str(budget)
        })
        
        return wallet
    
    def register_agent(self, agent_id: str, initial_budget: Decimal) -> bool:
        """Register a new agent with initial budget allocation, using only clear names. Merges duplicates if hash/alias detected."""
        import hashlib
        try:
            with self.lock:
                # Always use clear name if possible
                # If agent_id looks like a SHA-1 hash, try to find the clear name
                if len(agent_id) == 40 and all(c in '0123456789abcdef' for c in agent_id):
                    # Try to find a clear name that hashes to this
                    for known in self.agent_wallets.keys():
                        if hashlib.sha1(known.encode()).hexdigest() == agent_id:
                            # Merge: transfer budget, remove hash wallet
                            hash_wallet = self.agent_wallets.get(agent_id)
                            clear_wallet = self.agent_wallets[known]
                            if hash_wallet:
                                clear_wallet.allocated_budget += hash_wallet.allocated_budget
                                clear_wallet.balance += hash_wallet.balance
                                clear_wallet.spent_budget += hash_wallet.spent_budget
                                del self.agent_wallets[agent_id]
                                self._create_alert(AlertLevel.INFO, f"Merged duplicate agent wallet for {agent_id} into {known}", category="deduplication")
                            agent_id = known
                            break
                # Prevent registration of hash/alias if clear name exists
                if agent_id in self.agent_wallets:
                    self._create_alert(AlertLevel.WARNING, f"Agent {agent_id} already registered", category="info")
                    return False
                # Check if treasury has sufficient funds
                required_reserve = self.treasury_balance * self.emergency_reserve_ratio
                if self.treasury_balance - initial_budget < required_reserve:
                    self._create_alert(AlertLevel.CRITICAL, 
                                     f"Insufficient treasury for agent {agent_id} budget. Would breach emergency reserve.", category="emergency")
                    return False
                # Create wallet and allocate budget
                self._create_agent_wallet(agent_id, Decimal('0'), initial_budget)
                self.treasury_balance -= initial_budget
                # Record transaction
                tx = self._create_transaction(
                    tx_type=TransactionType.BUDGET_ALLOCATION,
                    from_agent="TREASURY",
                    to_agent=agent_id,
                    amount=initial_budget,
                    description=f"Initial budget allocation for {agent_id}"
                )
                self._execute_transaction(tx)
                print(f"💰 Agent {agent_id} registered with budget: {initial_budget}")
                return True
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Failed to register agent {agent_id}: {str(e)}", category="registration_error")
            return False
    
    def _create_transaction(self, tx_type: TransactionType, from_agent: str, to_agent: str, 
                              amount: Decimal, description: str, approval_required: Optional[bool] = None) -> Transaction:
        """Create a new transaction"""
        
        # Auto-determine if approval required
        if approval_required is None:
            approval_required = amount >= self.creator_approval_threshold
        
        tx = Transaction(
            tx_id=f"TX_{int(time.time() * 1000)}_{len(self.transaction_ledger)}",
            timestamp=time.time(),
            tx_type=tx_type,
            from_agent=from_agent,
            to_agent=to_agent,
            amount=amount,
            currency="CREDITS",
            description=description,
            status=TransactionStatus.PENDING,
            approval_required=approval_required
        )
        
        self.transaction_ledger.append(tx)
        return tx
    
    def _log_transaction(self, agent_id: str, amount: Decimal, reason: str, balance_after: Decimal, tx_type: str):
        """Record a transaction log entry for audit and reporting."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_id,
            "amount": str(amount),
            "reason": reason,
            "balance_after": str(balance_after),
            "tx_type": tx_type
        }
        self.audit_log.append(entry)
        print(f"[{entry['timestamp']}] Agent {agent_id} {tx_type} {amount} → Remaining budget: {balance_after}")

    def process_agent_spend(self, agent_id: str, amount: Decimal, description: str) -> Tuple[bool, str]:
        """Process spending request from an agent, with scarcity & sinks (80% to treasury, 20% burned)"""
        try:
            with self.lock:
                self._settle_pending_wallets()
                if agent_id not in self.agent_wallets:
                    return False, f"Agent {agent_id} not registered"
                wallet = self.agent_wallets[agent_id]
                # Check if agent is suspended
                if wallet.status != "active":
                    allocation_tx = next((tx for tx in self.transaction_ledger if tx.tx_type == TransactionType.BUDGET_ALLOCATION and tx.to_agent == agent_id), None)
                    if wallet.status == "pending" and allocation_tx and allocation_tx.status != TransactionStatus.EXECUTED:
                        return False, f"Agent {agent_id} wallet is pending; budget allocation not settled."
                    return False, f"Agent {agent_id} wallet is {wallet.status}"
                # Prevent spend if budget allocation transaction is not executed
                allocation_tx = next((tx for tx in self.transaction_ledger if tx.tx_type == TransactionType.BUDGET_ALLOCATION and tx.to_agent == agent_id), None)
                if allocation_tx and allocation_tx.status != TransactionStatus.EXECUTED:
                    self._create_alert(AlertLevel.INFO, f"Agent {agent_id} cannot spend until budget allocation is executed", category="budget_allocation")
                    return False, "Budget allocation not settled"
                # Check budget limits
                if wallet.remaining_budget < amount:
                    self._create_alert(AlertLevel.WARNING, f"Agent {agent_id} attempting to exceed budget: {amount} > {wallet.remaining_budget}", category="budget_violation")
                    return False, "Insufficient budget remaining"
                # Check daily spending limits
                daily_spend = self._calculate_daily_spend(agent_id)
                if daily_spend + amount > self.max_daily_spend_per_agent:
                    self._create_alert(AlertLevel.WARNING, f"Agent {agent_id} daily spending limit exceeded", category="budget_violation")
                    return False, "Daily spending limit exceeded"
                # Scarcity & Sinks logic: split spend
                treasury_share = (amount * Decimal('0.8')).quantize(Decimal('0.0001'))
                burn_share = amount - treasury_share
                # Create and execute transaction (full amount for audit)
                tx = self._create_transaction(
                    tx_type=TransactionType.AGENT_SPEND,
                    from_agent=agent_id,
                    to_agent="EXTERNAL",
                    amount=amount,
                    description=description
                )
                success = self._execute_transaction(tx)
                if success:
                    wallet.spent_budget += amount
                    wallet.last_activity = time.time()
                    self.treasury_balance -= treasury_share  # Only 80% goes to treasury
                    self.burned_total += burn_share          # 20% is burned
                    self._log_transaction(agent_id, amount, f"{description} (80% treasury, 20% burned)", wallet.remaining_budget, "spent")
                    self._log_transaction(agent_id, burn_share, "Root burned (scarcity sink)", wallet.remaining_budget, "burned")
                    return True, f"Transaction {tx.tx_id} executed successfully (80% to treasury, 20% burned)"
                else:
                    return False, "Transaction execution failed"
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Error processing spend for {agent_id}: {str(e)}", category="spend_error")
            return False, str(e)
    
    def request_budget_loan(self, agent_id: str, amount: Decimal, reason: str, approver: Optional[str] = None) -> Tuple[bool, str]:
        """Allow agent to request a loan from treasury, with approval logic."""
        try:
            with self.lock:
                if self.treasury_balance - amount < self.treasury_balance * self.emergency_reserve_ratio:
                    self._create_alert(AlertLevel.CRITICAL, f"Loan denied: insufficient treasury for {agent_id}", category="loan_denied")
                    return False, "Loan denied: insufficient treasury"
                # Approval logic (stub: integrate with Enforcer/Shogun)
                if approver and approver != self.creator_id:
                    self._create_alert(AlertLevel.WARNING, f"Loan for {agent_id} requires approval by {approver}", category="loan_pending")
                    return False, "Loan requires approval"
                wallet = self.agent_wallets.get(agent_id)
                if not wallet:
                    return False, "Agent not registered"
                wallet.allocated_budget += amount
                self.treasury_balance -= amount
                self._log_transaction(agent_id, amount, reason, wallet.remaining_budget, "loaned")
                return True, f"Loan of {amount} granted to {agent_id}"
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Loan error for {agent_id}: {str(e)}", category="loan_error")
            return False, str(e)

    def get_agent_burn_rate(self, agent_id: str) -> float:
        """Calculate agent's burn rate (spend per day)."""
        cutoff = time.time() - 86400 * 7  # 1 week
        total_spent = Decimal('0')
        for tx in self.transaction_ledger:
            if tx.from_agent == agent_id and tx.tx_type == TransactionType.AGENT_SPEND and tx.timestamp >= cutoff:
                total_spent += tx.amount
        return float(total_spent / 7)

    def get_sustainability_index(self, agent_id: str) -> float:
        """Estimate how long agent can operate at current burn rate."""
        wallet = self.agent_wallets.get(agent_id)
        if not wallet or wallet.remaining_budget <= 0:
            return 0.0
        burn_rate = self.get_agent_burn_rate(agent_id)
        if burn_rate == 0:
            return float('inf')
        return float(wallet.remaining_budget / Decimal(burn_rate))

    def check_economic_abuse(self, agent_id: str) -> bool:
        """Auto-alert and freeze if agent drains too fast."""
        index = self.get_sustainability_index(agent_id)
        if index < 2:  # Less than 2 days left
            self._create_alert(AlertLevel.CRITICAL, f"Agent {agent_id} flagged for economic abuse (sustainability index: {index:.2f})", category="abuse_detected")
            self.suspend_agent(agent_id, "Economic abuse detected")
            return True
        return False
    
    def _execute_transaction(self, tx: Transaction) -> bool:
        """Execute a transaction after all checks"""
        try:
            # If approval required and not yet approved, mark as pending
            if tx.approval_required and not tx.creator_approved:
                tx.status = TransactionStatus.PENDING
                self._create_alert(AlertLevel.INFO, f"Transaction {tx.tx_id} pending creator approval", category="approval")
                return False
            
            # Execute the transaction
            tx.status = TransactionStatus.EXECUTED
            
            # Log to audit trail
            self._log_audit("TRANSACTION_EXECUTED", {
                "tx_id": tx.tx_id,
                "type": tx.tx_type.value,
                "from": tx.from_agent,
                "to": tx.to_agent,
                "amount": str(tx.amount),
                "hash": tx.hash_signature
            })
            
            return True
            
        except Exception as e:
            tx.status = TransactionStatus.FAILED
            self._create_alert(AlertLevel.CRITICAL, f"Transaction execution failed: {str(e)}", category="transaction_error")
            return False
    
    def creator_approve_transaction(self, tx_id: str) -> bool:
        """Allow creator to approve pending transactions"""
        try:
            with self.lock:
                for tx in self.transaction_ledger:
                    if tx.tx_id == tx_id and tx.status == TransactionStatus.PENDING:
                        tx.creator_approved = True
                        return self._execute_transaction(tx)
                
                return False
                
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Error approving transaction {tx_id}: {str(e)}", category="approval_error")
            return False
    
    def creator_withdraw(self, amount: Decimal, destination: str) -> Tuple[bool, str]:
        """Allow creator to withdraw funds from treasury"""
        try:
            with self.lock:
                required_reserve = self.treasury_balance * self.emergency_reserve_ratio
                available = self.treasury_balance - required_reserve
                
                if amount > available:
                    self._create_alert(AlertLevel.WARNING, f"Creator withdrawal failed: Insufficient funds. Available: {available}, Requested: {amount}", category="system_withdrawal")
                    return False, f"Insufficient funds. Available: {available}, Requested: {amount}"
                
                # Create withdrawal transaction
                tx = self._create_transaction(
                    tx_type=TransactionType.WITHDRAWAL,
                    from_agent="TREASURY",
                    to_agent=destination,
                    amount=amount,
                    description=f"Creator withdrawal to {destination}",
                    approval_required=False  # Creator doesn't need to approve their own withdrawal
                )
                
                if self._execute_transaction(tx):
                    self.treasury_balance -= amount
                    return True, f"Withdrawal {tx.tx_id} executed successfully"
                else:
                    self._create_alert(AlertLevel.CRITICAL, f"Creator withdrawal execution failed", category="system_withdrawal")
                    return False, "Withdrawal execution failed"
                
        except Exception as e:
            return False, str(e)
    
    def _calculate_daily_spend(self, agent_id: str) -> Decimal:
        """Calculate agent's spending in the last 24 hours"""
        cutoff = time.time() - 86400  # 24 hours ago
        daily_spend = Decimal('0')
        
        for tx in self.transaction_ledger:
            if (tx.from_agent == agent_id and 
                tx.tx_type == TransactionType.AGENT_SPEND and
                tx.timestamp >= cutoff and
                tx.status == TransactionStatus.EXECUTED):
                daily_spend += tx.amount
        
        return daily_spend
    
    def suspend_agent(self, agent_id: str, reason: str) -> bool:
        """Suspend an agent's financial activities"""
        try:
            with self.lock:
                if agent_id in self.agent_wallets:
                    self.agent_wallets[agent_id].status = "suspended"
                    self._create_alert(AlertLevel.WARNING, f"Agent {agent_id} suspended: {reason}")
                    self._log_audit("AGENT_SUSPENDED", {"agent_id": agent_id, "reason": reason})
                    return True
                return False
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Error suspending agent {agent_id}: {str(e)}", category="suspend_error")
            return False
    
    def activate_failsafe_mode(self, reason: str = "Emergency protocol triggered"):
        """Activate failsafe treasury mode - only creator can move funds"""
        with self.lock:
            self.failsafe_mode = True
            self._create_alert(AlertLevel.EMERGENCY, f"FAILSAFE MODE ACTIVATED: {reason}", category="emergency")
            self._log_audit("FAILSAFE_ACTIVATED", {"reason": reason, "timestamp": time.time()})
            
            # Suspend all agents
            for agent_id in self.agent_wallets:
                if agent_id != "SYSTEM":
                    self.agent_wallets[agent_id].status = "frozen"
        
        print(f"🚨 FAILSAFE MODE ACTIVATED: {reason}")
    
    def _create_alert(self, level: AlertLevel, message: str, category: str = "info"):
        """Create system alert with category"""
        alert = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            "category": category,
            "acknowledged": False
        }
        self.alerts.append(alert)
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"🚨 {level.value.upper()} [{category}]: {message}")

    def _log_audit(self, action: str, details: Dict):
        """Log action to immutable audit trail"""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "agent": self.agent_id,
            "details": details,
            "hash": hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
        }
        self.audit_log.append(log_entry)

    def _start_monitoring(self):
        """Start continuous economic health monitoring"""
        def monitor():
            while self.monitoring_active:
                try:
                    self._update_economic_health()
                    self._check_anomalies()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self._create_alert(AlertLevel.CRITICAL, f"Monitoring error: {str(e)}", category="monitoring_error")
                    time.sleep(60)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_economic_health(self):
        """Update economic health metrics"""
        with self.lock:
            total_allocated = sum(wallet.allocated_budget for wallet in self.agent_wallets.values())
            total_spent = sum(wallet.spent_budget for wallet in self.agent_wallets.values())
            
            self.economic_health_metrics = {
                "treasury_balance": float(self.treasury_balance),
                "total_allocated": float(total_allocated),
                "total_spent": float(total_spent),
                "utilization_rate": float(total_spent / total_allocated) if total_allocated > 0 else 0,
                "active_agents": len([w for w in self.agent_wallets.values() if w.status == "active"]),
                "emergency_reserve": float(self.treasury_balance * self.emergency_reserve_ratio),
                "last_updated": time.time()
            }
    
    def _check_anomalies(self):
        """Check for economic anomalies and suspicious activity"""
        try:
            # Check treasury depletion
            if self.treasury_balance < self.treasury_balance * self.emergency_reserve_ratio:
                self._create_alert(AlertLevel.CRITICAL, "Treasury approaching emergency reserve limit", category="emergency")
            
            # Check for unusual spending patterns
            for agent_id, wallet in self.agent_wallets.items():
                if wallet.utilization_rate > 0.9:  # 90% budget used
                    self._create_alert(AlertLevel.WARNING, 
                                     f"Agent {agent_id} has used {wallet.utilization_rate:.1%} of budget", category="budget_violation")
                
                # Check for dormant agents
                if time.time() - wallet.last_activity > 86400 * 7:  # 1 week
                    self._create_alert(AlertLevel.INFO, f"Agent {agent_id} dormant for 7+ days", category="info")
                    
        except Exception as e:
            self._create_alert(AlertLevel.CRITICAL, f"Anomaly detection error: {str(e)}", category="anomaly_error")
    
    def get_financial_report(self) -> Dict:
        """Generate comprehensive financial report for creator"""
        with self.lock:
            recent_transactions = [
                {
                    "tx_id": tx.tx_id,
                    "timestamp": datetime.fromtimestamp(tx.timestamp).isoformat(),
                    "type": tx.tx_type.value,
                    "from": tx.from_agent,
                    "to": tx.to_agent,
                    "amount": str(tx.amount),
                    "status": tx.status.value,
                    "description": tx.description
                }
                for tx in sorted(self.transaction_ledger, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
            
            agent_summaries = {}
            for agent_id, wallet in self.agent_wallets.items():
                agent_summaries[agent_id] = {
                    "balance": str(wallet.balance),
                    "allocated_budget": str(wallet.allocated_budget),
                    "spent_budget": str(wallet.spent_budget),
                    "remaining_budget": str(wallet.remaining_budget),
                    "utilization_rate": f"{wallet.utilization_rate:.1%}",
                    "status": wallet.status,
                    "last_activity": datetime.fromtimestamp(wallet.last_activity).isoformat()
                }
            
            active_alerts = [alert for alert in self.alerts[-5:] if not alert.get("acknowledged", False)]
            
            return {
                "report_timestamp": datetime.now().isoformat(),
                "treasury_balance": str(self.treasury_balance),
                "failsafe_mode": self.failsafe_mode,
                "economic_health": self.economic_health_metrics,
                "agent_wallets": agent_summaries,
                "recent_transactions": recent_transactions,
                "active_alerts": active_alerts,
                "audit_entries": len(self.audit_log),
                "total_transactions": len(self.transaction_ledger)
            }
    
    def shutdown(self):
        """Gracefully shutdown the agent"""
        self.monitoring_active = False
        self._log_audit("SYSTEM_SHUTDOWN", {"timestamp": time.time()})
        print("🧩 ACC-LEDGER shutdown complete")

    def _settle_pending_wallets(self):
        """Settle all pending agent wallets whose budget allocation transaction is executed."""
        with self.lock:
            for agent_id, wallet in self.agent_wallets.items():
                if wallet.status == "pending":
                    allocation_tx = next((tx for tx in self.transaction_ledger if tx.tx_type == TransactionType.BUDGET_ALLOCATION and tx.to_agent == agent_id), None)
                    if allocation_tx and allocation_tx.status == TransactionStatus.EXECUTED:
                        wallet.status = "active"
                        self._log_audit("WALLET_ACTIVATED", {"agent_id": agent_id, "timestamp": time.time()})

    def setup_economic_layer(self, redis_client=None):
        """Instantiate wallet manager, market, and risk engine for agent."""
        self.wallet_manager = WalletManager(redis_client)
        self.economic_engine = EconomicEngine(redis_client, ledger=self)
        self.market = Market(redis_client, ledger=self)
        self.risk_manager = RiskManager(redis_client)

    def create_agent_chain_wallet(self, agent_id: str, networks: List[str]) -> Dict[str, ChainWallet]:
        """Create blockchain wallets for agent using WalletManager."""
        if not hasattr(self, 'wallet_manager'):
            self.setup_economic_layer()
        # This is an async method, so you may need to run it in an event loop in production
        import asyncio
        return asyncio.run(self.wallet_manager.create_wallet(agent_id, networks))

    def process_cross_chain_transaction(self, from_agent: str, to_agent: str, amount: float, from_network: str = "ethereum", to_network: str = "ethereum", token_address: str = "", priority: str = "medium", max_gas: float = 0.0, deadline: float = 0.0, metadata: dict = {}) -> str:
        """Create and process a cross-chain transaction for an agent."""
        if not hasattr(self, 'wallet_manager'):
            self.setup_economic_layer()
        import uuid, time, asyncio
        from economic.wallet import TransactionRequest, CrossChainTransactionProcessor
        tx_request = TransactionRequest(
            request_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            amount=amount,
            token_address=token_address,
            from_network=from_network,
            to_network=to_network,
            priority=priority,
            max_gas=max_gas,
            deadline=deadline,
            metadata=metadata,
            status="pending",
            created_at=time.time()
        )
        processor = CrossChainTransactionProcessor(self.wallet_manager, None)
        return asyncio.run(processor.submit_transaction(tx_request))

    def assess_systemic_risk(self, system_state):
        """Use RiskManager to assess systemic risk."""
        if not hasattr(self, 'risk_manager'):
            self.setup_economic_layer()
        import asyncio
        return asyncio.run(self.risk_manager.assess_systemic_risk(system_state))

    def get_market_price(self):
        """Get current market price using Market."""
        if not hasattr(self, 'market'):
            self.setup_economic_layer()
        import asyncio
        return asyncio.run(self.market.get_market_price())

# Example Usage & Testing
def main():
    """Demonstrate ACC-LEDGER capabilities"""
    print("🧩 Initializing ACC-LEDGER Agent...")
    
    # Initialize the accountant agent
    accountant = ACCLedger(creator_id="CREATOR_001", initial_treasury=Decimal('5000'))
    
    # Register some agents
    print("\n💰 Registering agents...")
    accountant.register_agent("C2-COMMS", Decimal('500'))
    accountant.register_agent("SHOGUN-ENFORCE", Decimal('300'))
    accountant.register_agent("DATA-COLLECTOR", Decimal('200'))
    
    # Simulate some spending
    print("\n💸 Processing agent spending...")
    success, msg = accountant.process_agent_spend("C2-COMMS", Decimal('50'), "API calls for coordination")
    print(f"C2-COMMS spend: {msg}")
    
    success, msg = accountant.process_agent_spend("DATA-COLLECTOR", Decimal('25'), "Data acquisition")
    print(f"DATA-COLLECTOR spend: {msg}")
    
    # Try to exceed budget
    success, msg = accountant.process_agent_spend("SHOGUN-ENFORCE", Decimal('350'), "Enforcement action")
    print(f"SHOGUN-ENFORCE overspend attempt: {msg}")
    
    # Creator withdrawal
    print("\n🏦 Creator withdrawal...")
    success, msg = accountant.creator_withdraw(Decimal('1000'), "CREATOR_WALLET")
    print(f"Creator withdrawal: {msg}")
    
    # Generate report
    print("\n📊 Financial Report:")
    report = accountant.get_financial_report()
    
    print(f"Treasury Balance: {report['treasury_balance']}")
    print(f"Active Agents: {report['economic_health']['active_agents']}")
    print(f"System Utilization: {report['economic_health']['utilization_rate']:.1%}")
    
    print("\nAgent Status:")
    for agent_id, wallet in report['agent_wallets'].items():
        if agent_id != "SYSTEM":
            print(f"  {agent_id}: {wallet['remaining_budget']} remaining ({wallet['utilization_rate']} used)")
    
    print("\nRecent Transactions:")
    for tx in report['recent_transactions'][:3]:
        print(f"  {tx['tx_id']}: {tx['from']} → {tx['to']} | {tx['amount']} | {tx['status']}")
    
    # Demonstrate emergency protocols
    print("\n🚨 Testing emergency protocols...")
    accountant.suspend_agent("DATA-COLLECTOR", "Suspicious activity detected")
    
    # Wait a moment for monitoring
    time.sleep(2)
    
    # Final report
    final_report = accountant.get_financial_report()
    print(f"\nFinal Treasury: {final_report['treasury_balance']}")
    print(f"Active Alerts: {len(final_report['active_alerts'])}")
    
    # Shutdown
    accountant.shutdown()

if __name__ == "__main__":
    main()