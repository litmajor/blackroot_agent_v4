# agent_wallet_integration.py
import asyncio
import json
import os
from dotenv import load_dotenv
from web3 import Web3
from logging import config
import uuid
import time
import logging
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import redis.asyncio as redis
from black_vault import BlackVault
from blackvault_client import BlackVaultClient

# Import the enhanced BEL v3 system
from bel_layer import (
    Ledger, EconomicV3Mixin, LedgerEntry, SystemState, 
    Market, MarketOrder, RiskManager, EconomicEngine,
    CFG, AgentStatus
)

# Wallet-specific configurations
WALLET_CFG = {
    "supported_networks": ["ethereum", "polygon", "arbitrum", "celo"],
    "default_gas_limit": 100000,
    "gas_buffer": 1.2,
    "max_slippage": 0.05,  # 5%
    "bridge_fee": 0.001,   # 0.1%
    "cross_chain_timeout": 300,  # 5 minutes
    "wallet_creation_cost": 10.0,  # ROOT cost to create wallet
    "transaction_fee_percentage": 0.005,  # 0.5% of transaction value
}

class NetworkType(Enum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    CELO = "celo"

@dataclass
class ChainWallet:
    """Represents an agent's wallet on a specific blockchain"""
    agent_id: str
    network: str
    address: str
    vault_key_id: str  # Reference to BlackVault entry
    balance_native: float = 0.0
    balance_tokens: Dict[str, float] = {}
    nonce: int = 0
    created_at: float = 0.0
    last_sync: float = 0.0

    def __post_init__(self):
        if self.balance_tokens is None:
            self.balance_tokens = {}
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class TransactionRequest:
    """Represents a cross-chain or on-chain transaction request"""
    request_id: str
    from_agent: str
    to_agent: str
    amount: float
    token_address: Optional[str] = None
    from_network: str = "ethereum"
    to_network: str = "ethereum" 
    priority: str = "medium"  # low, medium, high
    max_gas: Optional[float] = None
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = {} 
    status: str = "pending"
    tx_hash: Optional[str] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class BridgeRoute:
    """Represents a cross-chain bridge route"""
    from_network: str
    to_network: str
    estimated_time: int  # seconds
    fee_percentage: float
    min_amount: float
    max_amount: float
    available: bool = True

class WalletManager:
    """Manages multi-chain wallets for agents"""
    
    def __init__(self, redis_client):
        load_dotenv()
        self.r = redis_client
        self.wallets_key = "agent_wallets"
        self.pending_txs_key = "pending_transactions"
        eth_rpc_url = os.getenv("ETH_RPC_URL", "http://localhost:8545")
        self.blackvault = BlackVault()  # Initialize BlackVault client
        self.blackvault_client = BlackVaultClient()
        self.web3 = Web3(Web3.HTTPProvider(eth_rpc_url))
        # Define available bridge routes
        self.bridge_routes = [
            BridgeRoute("ethereum", "polygon", 600, 0.001, 0.01, 1000000),
            BridgeRoute("polygon", "ethereum", 1800, 0.002, 0.01, 1000000),
            BridgeRoute("ethereum", "arbitrum", 900, 0.0015, 0.005, 500000),
            BridgeRoute("arbitrum", "ethereum", 1200, 0.002, 0.005, 500000),
            BridgeRoute("celo", "ethereum", 2400, 0.003, 0.1, 100000),
            BridgeRoute("ethereum", "celo", 1800, 0.0025, 0.1, 100000),
        ]
    
    async def create_wallet(self, agent_id: str, networks: List[str]) -> Dict[str, ChainWallet]:
        """Create wallets for an agent across multiple networks using BlackVault for key management"""
        wallets = {}
        for network in networks:
            if network not in [n.value for n in NetworkType]:
                raise ValueError(f"Unsupported network: {network}")

            # BlackVault key generation and storage (replace with real API)
            vault_key_id = await self._generate_and_store_key_blackvault(agent_id, network)
            address = await self._get_address_from_vault_key(vault_key_id, network)

            wallet = ChainWallet(
                agent_id=agent_id,
                network=network,
                address=address,
                vault_key_id=vault_key_id
            )
            wallets[network] = wallet

        await self._save_wallets(agent_id, wallets)
        return wallets

    async def _generate_and_store_key_blackvault(self, agent_id: str, network: str) -> str:
        # Replace with real BlackVault API
        vault_key_id = f"vault_{agent_id}_{network}_{int(time.time())}"
        return vault_key_id

    async def _get_address_from_vault_key(self, vault_key_id: str, network: str) -> str:
        # Replace with real blockchain address derivation
        address = f"0x{hashlib.sha256(vault_key_id.encode()).hexdigest()[:40]}"
        return address

    async def get_wallets(self, agent_id: str) -> Dict[str, ChainWallet]:
        """Get all wallets for an agent"""
        raw = await self.r.hget(self.wallets_key, agent_id)
        if not raw:
            return {}
        
        data = json.loads(raw)
        wallets = {}
        
        for network, wallet_data in data.items():
            wallet = ChainWallet(agent_id=agent_id, network=network, address="", vault_key_id="")
            wallet.__dict__.update(wallet_data)
            wallets[network] = wallet
        
        return wallets
    
    async def _save_wallets(self, agent_id: str, wallets: Dict[str, ChainWallet]):
        """Save agent wallets to Redis"""
        serialized = {network: asdict(wallet) for network, wallet in wallets.items()}
        await self.r.hset(self.wallets_key, agent_id, json.dumps(serialized))

    async def update_balance(self, agent_id: str, network: str, native_balance: Optional[float] = None, token_balances: Optional[Dict[str, float]] = None):
        """Update wallet balance using real blockchain data for Ethereum"""
        wallets = await self.get_wallets(agent_id)
        if network not in wallets:
            raise ValueError(f"Wallet not found for network {network}")
        wallet = wallets[network]
        # Query real balance if not provided
        if native_balance is None and network == "ethereum":
            wallet.balance_native = self.web3.eth.get_balance(Web3.to_checksum_address(wallet.address)) / 1e18
        else:
            wallet.balance_native = native_balance if native_balance is not None else wallet.balance_native
        # Token balances: add ERC20 logic here if needed
        if token_balances:
            wallet.balance_tokens.update(token_balances)
        wallet.last_sync = time.time()
        await self._save_wallets(agent_id, wallets)
    
    def get_bridge_route(self, from_network: str, to_network: str) -> Optional[BridgeRoute]:
        """Find bridge route between networks"""
        for route in self.bridge_routes:
            if route.from_network == from_network and route.to_network == to_network and route.available:
                return route
        return None
    
    async def estimate_bridge_cost(self, from_network: str, to_network: str, amount: float) -> Dict[str, float]:
        """Estimate cost of bridging assets"""
        route = self.get_bridge_route(from_network, to_network)
        if not route:
            raise ValueError(f"No bridge route from {from_network} to {to_network}")
        
        if amount < route.min_amount or amount > route.max_amount:
            raise ValueError(f"Amount {amount} outside bridge limits [{route.min_amount}, {route.max_amount}]")
        
        fee = amount * route.fee_percentage
        return {
            "bridge_fee": fee,
            "net_amount": amount - fee,
            "estimated_time": route.estimated_time,
            "fee_percentage": route.fee_percentage
        }

    async def sign_and_send_transaction(self, agent_id, network, tx_data):
        wallets = await self.get_wallets(agent_id)
        wallet = wallets[network]
        vault_key_id = wallet.vault_key_id
        private_key = self.blackvault_client.retrieve_key(vault_key_id, agent_id)
        if network == "ethereum":
            tx = {
                'to': tx_data['to'],
                'value': int(tx_data['amount'] * 1e18),
                'gas': WALLET_CFG['default_gas_limit'],
                'nonce': self.web3.eth.get_transaction_count(Web3.to_checksum_address(wallet.address)),
            }
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return tx_hash.hex()
        # ...other networks integration here...
        return "tx_hash_simulated"

class CrossChainTransactionProcessor:
    """Handles cross-chain transactions and bridges"""
    
    def __init__(self, wallet_manager: WalletManager, redis_client):
        self.wallet_manager = wallet_manager
        self.r = redis_client
        self.processing_queue_key = "tx_processing_queue"
    
    async def submit_transaction(self, tx_request: TransactionRequest) -> str:
        """Submit a transaction for processing"""
        # Validate request
        if not await self._validate_transaction(tx_request):
            raise ValueError("Invalid transaction request")
        
        # Add to processing queue
        await self.r.lpush(self.processing_queue_key, json.dumps(asdict(tx_request)))
        
        return tx_request.request_id
    
    async def _validate_transaction(self, tx_request: TransactionRequest) -> bool:
        """Validate transaction request"""
        # Check if wallets exist
        from_wallets = await self.wallet_manager.get_wallets(tx_request.from_agent)
        to_wallets = await self.wallet_manager.get_wallets(tx_request.to_agent)
        
        if tx_request.from_network not in from_wallets:
            return False
        
        if tx_request.to_network not in to_wallets:
            return False
        
        # Check balance
        from_wallet = from_wallets[tx_request.from_network]
        if tx_request.token_address:
            available = from_wallet.balance_tokens.get(tx_request.token_address, 0.0)
        else:
            available = from_wallet.balance_native
        
        return available >= tx_request.amount
    
    async def process_transactions(self):
        """Process pending transactions (background task)"""
        while True:
            try:
                # Get next transaction
                raw_tx = await self.r.brpop(self.processing_queue_key, timeout=10)
                if not raw_tx:
                    continue
                
                tx_data = json.loads(raw_tx[1])
                tx_request = TransactionRequest(**tx_data)
                
                await self._execute_transaction(tx_request)
                
            except Exception as e:
                logging.error(f"Transaction processing error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_transaction(self, tx_request: TransactionRequest):
        """Execute a transaction (simulated)"""
        try:
            # Simulate transaction execution
            if tx_request.from_network != tx_request.to_network:
                # Cross-chain transaction
                await self._execute_bridge_transaction(tx_request)
            else:
                # Same-chain transaction
                await self._execute_same_chain_transaction(tx_request)
            
            # Update status
            tx_request.status = "completed"
            tx_request.tx_hash = f"0x{hashlib.sha256(f'{tx_request.request_id}-{time.time()}'.encode()).hexdigest()}"
            
        except Exception as e:
            tx_request.status = "failed"
            tx_request.metadata["error"] = str(e)
            logging.error(f"Transaction {tx_request.request_id} failed: {e}")
    
    async def _execute_bridge_transaction(self, tx_request: TransactionRequest):
        """Execute cross-chain bridge transaction using BlackVault signing"""
        route = self.wallet_manager.get_bridge_route(tx_request.from_network, tx_request.to_network)
        if not route:
            raise ValueError("No bridge route available")

        # Calculate fees
        bridge_fee = tx_request.amount * route.fee_percentage
        net_amount = tx_request.amount - bridge_fee

        # Prepare transaction data (replace with real blockchain tx data)
        tx_data = {
            "from": tx_request.from_agent,
            "to": tx_request.to_agent,
            "amount": tx_request.amount,
            "token_address": tx_request.token_address,
            "from_network": tx_request.from_network,
            "to_network": tx_request.to_network
        }

        # Use WalletManager to sign and send transaction via BlackVault
        tx_hash = await self.wallet_manager.sign_and_send_transaction(tx_request.from_agent, tx_request.from_network, tx_data)
        tx_request.tx_hash = tx_hash

        # Simulate bridge time
        await asyncio.sleep(1)  # In reality, would wait for bridge confirmation

        await self._credit_wallet(tx_request.to_agent, tx_request.to_network, net_amount, tx_request.token_address)

        # Record bridge fee
        tx_request.metadata["bridge_fee"] = bridge_fee
        tx_request.metadata["net_amount"] = net_amount
    
    async def _execute_same_chain_transaction(self, tx_request: TransactionRequest):
        """Execute same-chain transaction"""
        # Calculate network fees (simulated)
        gas_fee = 0.001  # Fixed for simulation
        net_amount = tx_request.amount
        
        # Update balances
        await self._debit_wallet(tx_request.from_agent, tx_request.from_network, tx_request.amount, tx_request.token_address)
        await self._credit_wallet(tx_request.to_agent, tx_request.to_network, net_amount, tx_request.token_address)
        
        # Record gas fee
        tx_request.metadata["gas_fee"] = gas_fee
    
    async def _debit_wallet(self, agent_id: str, network: str, amount: float, token_address: Optional[str]):
        """Debit amount from wallet"""
        wallets = await self.wallet_manager.get_wallets(agent_id)
        wallet = wallets[network]
        
        if token_address:
            if token_address not in wallet.balance_tokens:
                wallet.balance_tokens[token_address] = 0.0
            wallet.balance_tokens[token_address] -= amount
        else:
            wallet.balance_native -= amount
        
        await self.wallet_manager._save_wallets(agent_id, wallets)
    
    async def _credit_wallet(self, agent_id: str, network: str, amount: float, token_address: Optional[str]):
        """Credit amount to wallet"""
        wallets = await self.wallet_manager.get_wallets(agent_id)
        wallet = wallets[network]
        
        if token_address:
            if token_address not in wallet.balance_tokens:
                wallet.balance_tokens[token_address] = 0.0
            wallet.balance_tokens[token_address] += amount
        else:
            wallet.balance_native += amount
        
        await self.wallet_manager._save_wallets(agent_id, wallets)

class EnhancedEconomicMixin(EconomicV3Mixin):
    """Enhanced economic mixin with wallet integration"""
    
    def __init__(self, kernel):
        super().__init__(kernel)
        # Ensure Ledger is initialized with Redis
        if not hasattr(self, 'ledger') or not hasattr(self.ledger, 'r'):
            self.ledger = Ledger(self.r)
        self.wallet_manager = WalletManager(self.r)
        self.tx_processor = CrossChainTransactionProcessor(self.wallet_manager, self.r)
        self.supported_networks = ["ethereum", "polygon", "arbitrum"]
    
    async def initialize_wallets(self, networks: Optional[List[str]] = None) -> Dict[str, ChainWallet]:
        """Initialize wallets for this agent"""
        if networks is None:
            networks = self.supported_networks
        
        # Check if agent can afford wallet creation
        balance = await self.balance()
        wallet_cost = WALLET_CFG["wallet_creation_cost"] * len(networks)
        
        if balance < wallet_cost:
            raise ValueError(f"Insufficient balance for wallet creation. Need {wallet_cost} ROOT, have {balance}")
        
        # Debit wallet creation cost
        await self.ledger.debit(self.agent_id, wallet_cost, f"wallet_creation_{len(networks)}_networks")
        
        # Create wallets
        wallets = await self.wallet_manager.create_wallet(self.agent_id, networks)
        
        # Initialize with some funds (simulated)
        for network, wallet in wallets.items():
            await self.wallet_manager.update_balance(
                self.agent_id, 
                network, 
                0.1,  # 0.1 ETH equivalent
                {"USDC": 100.0, "DAI": 50.0}  # Some stablecoins
            )
        
        return wallets
    
    async def get_wallet_portfolio(self) -> Dict[str, Any]:
        """Get comprehensive wallet portfolio"""
        wallets = await self.wallet_manager.get_wallets(self.agent_id)
        root_balance = await self.balance()
        root_status = await self.my_status()
        
        portfolio = {
            "agent_id": self.agent_id,
            "root_economy": root_status,
            "wallets": {},
            "total_networks": len(wallets),
            "cross_chain_capability": len(wallets) > 1
        }
        
        for network, wallet in wallets.items():
            portfolio["wallets"][network] = {
                "address": wallet.address,
                "native_balance": wallet.balance_native,
                "token_balances": wallet.balance_tokens,
                "last_sync": wallet.last_sync,
                "age_hours": (time.time() - wallet.created_at) / 3600
            }
        
        return portfolio
    
    async def bridge_assets(self, from_network: str, to_network: str, amount: float, token_address: Optional[str] = None) -> str:
        """Bridge assets between networks"""
        # Estimate bridge cost
        cost_info = await self.wallet_manager.estimate_bridge_cost(from_network, to_network, amount)
        
        # Create transaction request
        tx_request = TransactionRequest(
            request_id=str(uuid.uuid4()),
            from_agent=self.agent_id,
            to_agent=self.agent_id,  # Bridging to self
            amount=amount,
            token_address=token_address,
            from_network=from_network,
            to_network=to_network,
            priority="medium",
            metadata={"operation": "bridge", "cost_info": cost_info}
        )
        
        # Submit for processing
        return await self.tx_processor.submit_transaction(tx_request)
    
    async def send_cross_chain(self, to_agent_id: str, amount: float, from_network: str, to_network: str, token_address: Optional[str] = None) -> str:
        """Send assets to another agent across chains"""
        # Check if we have permission and balance
        can_send = await self._validate_cross_chain_send(to_agent_id, amount, from_network, token_address)
        if not can_send:
            raise ValueError("Cannot execute cross-chain send")
        
        # Create transaction request
        tx_request = TransactionRequest(
            request_id=str(uuid.uuid4()),
            from_agent=self.agent_id,
            to_agent=to_agent_id,
            amount=amount,
            token_address=token_address,
            from_network=from_network,
            to_network=to_network,
            priority="high",  # Inter-agent transfers are high priority
            metadata={"operation": "cross_chain_transfer"}
        )
        
        return await self.tx_processor.submit_transaction(tx_request)
    
    async def _validate_cross_chain_send(self, to_agent_id: str, amount: float, from_network: str, token_address: Optional[str] = None) -> bool:
        """Validate cross-chain send capability"""
        # Check our wallets
        our_wallets = await self.wallet_manager.get_wallets(self.agent_id)
        if from_network not in our_wallets:
            return False
        
        # Check recipient wallets exist
        recipient_wallets = await self.wallet_manager.get_wallets(to_agent_id)
        if not recipient_wallets:
            return False
        
        # Check balance
        from_wallet = our_wallets[from_network]
        if token_address:
            available = from_wallet.balance_tokens.get(token_address, 0.0)
        else:
            available = from_wallet.balance_native
        
        return available >= amount
    
    async def optimize_liquidity(self) -> Dict[str, Any]:
        """Optimize liquidity across networks"""
        wallets = await self.wallet_manager.get_wallets(self.agent_id)
        optimization_plan = {
            "rebalancing_needed": False,
            "suggested_moves": [],
            "current_distribution": {},
            "optimal_distribution": {}
        }
        
        # Calculate current distribution
        total_value = 0
        for network, wallet in wallets.items():
            value = wallet.balance_native + sum(wallet.balance_tokens.values())
            optimization_plan["current_distribution"][network] = value
            total_value += value
        
        # Simple optimization: balance liquidity across chains
        if len(wallets) > 1 and total_value > 0:
            optimal_per_chain = total_value / len(wallets)
            
            for network, current_value in optimization_plan["current_distribution"].items():
                optimal_value = optimal_per_chain
                optimization_plan["optimal_distribution"][network] = optimal_value
                
                if abs(current_value - optimal_value) > optimal_value * 0.2:  # 20% threshold
                    optimization_plan["rebalancing_needed"] = True
                    
                    if current_value > optimal_value:
                        # Need to send out
                        excess = current_value - optimal_value
                        for target_network, target_value in optimization_plan["current_distribution"].items():
                            if target_network != network and target_value < optimal_per_chain:
                                needed = optimal_per_chain - target_value
                                move_amount = min(excess, needed)
                                
                                optimization_plan["suggested_moves"].append({
                                    "from_network": network,
                                    "to_network": target_network,
                                    "amount": move_amount,
                                    "reason": "liquidity_rebalancing"
                                })
                                break
        
        return optimization_plan
    
    async def execute_liquidity_optimization(self) -> List[str]:
        """Execute liquidity optimization plan"""
        plan = await self.optimize_liquidity()
        
        if not plan["rebalancing_needed"]:
            return []
        
        executed_txs = []
        
        for move in plan["suggested_moves"]:
            try:
                tx_id = await self.bridge_assets(
                    move["from_network"],
                    move["to_network"], 
                    move["amount"]
                )
                executed_txs.append(tx_id)
            except Exception as e:
                logging.error(f"Liquidity optimization move failed: {e}")
        
        return executed_txs
    
    async def get_cross_chain_opportunities(self) -> Dict[str, Any]:
        """Analyze cross-chain arbitrage and yield opportunities"""
        wallets = await self.wallet_manager.get_wallets(self.agent_id)
        
        opportunities = {
            "arbitrage": [],
            "yield_farming": [],
            "bridge_efficiency": {},
            "network_costs": {}
        }
        
        # Simulate finding opportunities (in reality, would query DeFi protocols)
        for network in wallets.keys():
            # Mock yield opportunities
            opportunities["yield_farming"].append({
                "network": network,
                "protocol": f"MockDeFi-{network}",
                "apy": 8.5 + hash(network) % 10,  # Mock APY
                "risk_level": "medium",
                "min_deposit": 100.0
            })
            
            # Mock network costs
            opportunities["network_costs"][network] = {
                "gas_price": 0.001 + (hash(network) % 100) / 100000,
                "congestion": "low",
                "recommended_times": ["00:00-06:00 UTC", "14:00-18:00 UTC"]
            }
        
        # Mock arbitrage opportunities between networks
        if len(wallets) > 1:
            networks = list(wallets.keys())
            for i, net1 in enumerate(networks):
                for net2 in networks[i+1:]:
                    # Mock price difference
                    price_diff = (hash(f"{net1}{net2}") % 100) / 1000  # 0-10% difference
                    
                    if price_diff > 0.02:  # >2% difference worth arbitraging
                        opportunities["arbitrage"].append({
                            "buy_network": net1,
                            "sell_network": net2,
                            "price_difference": price_diff,
                            "estimated_profit": price_diff * 0.8,  # Account for fees
                            "min_capital": 50.0
                        })
        
        return opportunities

# Background task to process transactions
async def start_transaction_processor(wallet_manager: WalletManager, redis_client):
    """Start the transaction processor as a background task"""
    processor = CrossChainTransactionProcessor(wallet_manager, redis_client)
    await processor.process_transactions()

# Example usage
async def example_agent_wallet_usage():
    """Example of how an agent would use the enhanced wallet system"""
    
    # Import the actual kernel from core.py
    from kernel.core import BlackrootKernel
    kernel = BlackrootKernel(config={})

    try:
        # Create enhanced economic agent
        agent = EnhancedEconomicMixin(kernel)
        
        # Initialize wallets on multiple networks
        print("Initializing wallets...")
        wallets = await agent.initialize_wallets(["ethereum", "polygon", "arbitrum"])
        print(f"Created {len(wallets)} wallets")

        # Get portfolio overview
        portfolio = await agent.get_wallet_portfolio()
        print(f"Portfolio: {json.dumps(portfolio, indent=2)}")

        # Check for cross-chain opportunities
        opportunities = await agent.get_cross_chain_opportunities()
        print(f"Found {len(opportunities['arbitrage'])} arbitrage opportunities")

        # Optimize liquidity distribution
        optimization = await agent.optimize_liquidity()
        print(f"Liquidity optimization needed: {optimization['rebalancing_needed']}")

        if optimization["rebalancing_needed"]:
            print(f"Suggested moves: {len(optimization['suggested_moves'])}")

        # Optionally execute liquidity optimization
        executed_txs = await agent.execute_liquidity_optimization()
        if executed_txs:
            print(f"Executed liquidity moves: {executed_txs}")

        print("Agent wallet system integrated successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")

# Run example
if __name__ == "__main__":
    asyncio.run(example_agent_wallet_usage())