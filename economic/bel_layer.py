# bel_v3_complete_fixed.py
import asyncio, json, uuid, time, logging, math, random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import redis.asyncio as redis
from redis.exceptions import WatchError

# Enhanced configuration for full v3 features
CFG = {
    "redis_url": "redis://localhost:6379",
    "currency": "ROOT",
    "ledger_ttl": 3600,
    "stake_lock": 600,
    "gossip": "bel_v3",
    "max_debt": -50,
    "reputation_decay": 0.995,
    "base_roi": 0.02,                    # 2% base return
    "roi_volatility": 0.5,               # ROI adjustment factor
    "market_maker_spread": 0.05,         # 5% bid-ask spread
    "bankruptcy_threshold": -100,        # Automatic bankruptcy level
    "systemic_risk_threshold": 0.3,      # 30% system leverage limit
    "supply_growth_rate": 0.01,          # 1% ROOT supply growth per cycle
    "scarcity_multiplier": 1.0,          # Dynamic scarcity factor
    "circuit_breaker_loss": 0.2,         # 20% loss triggers circuit breaker
}

class AgentStatus(Enum):
    ACTIVE = "active"
    BANKRUPT = "bankrupt" 
    SUSPENDED = "suspended"

@dataclass
class MarketOrder:
    agent_id: str
    order_id: str
    side: str  # "buy" or "sell"
    amount: float
    price: float
    timestamp: float
    filled: float = 0.0

@dataclass
class LedgerEntry:
    agent_id: str
    balance: float = 100.0
    stake: float = 0.0
    reputation: float = 1.0
    last_seen: float = 0.0
    earned: float = 0.0
    spent: float = 0.0
    debt: float = 0.0
    roi_rate: float = 0.02              # Individual ROI rate
    status: str = AgentStatus.ACTIVE.value
    bankruptcy_count: int = 0
    systemic_contribution: float = 0.0   # Contribution to systemic risk
    
    def __post_init__(self):
        if self.last_seen == 0.0:
            self.last_seen = time.time()

@dataclass
class SystemState:
    total_supply: float = 0.0
    total_staked: float = 0.0
    total_debt: float = 0.0
    active_agents: int = 0
    bankrupt_agents: int = 0
    market_price: float = 1.0            # ROOT price in market
    system_leverage: float = 0.0         # Debt/Supply ratio
    circuit_breaker_active: bool = False
    last_roi_update: float = 0.0
    scarcity_factor: float = 1.0

class Market:
    def __init__(self, redis_client, ledger):
        self.r = redis_client
        self.ledger = ledger
        self.buy_orders_key = "bel_buy_orders"
        self.sell_orders_key = "bel_sell_orders"
        self.trade_history_key = "bel_trades"
        self.escrow_key = "bel_escrow"  # hash: order_id -> locked amount

    async def place_order(self, order: MarketOrder) -> bool:
        """Place buy/sell order with proper escrow and settlement"""
        # Simple escrow: lock max spend for buys; nothing for sells (they're supplying ROOT)
        if order.side == "buy":
            notional = order.amount * order.price
            ok = await self.ledger.debit(order.agent_id, notional, f"escrow:{order.order_id}")
            if not ok:
                return False
            await self.r.hset(self.escrow_key, order.order_id, str(notional))

        order_key = self.buy_orders_key if order.side == "buy" else self.sell_orders_key
        await self.r.zadd(order_key, {json.dumps(asdict(order)): order.price})
        await self._match_orders()
        return True

    async def _match_orders(self):
        """Match buy and sell orders with settlement"""
        while True:
            buy_row = await self.r.zrevrange(self.buy_orders_key, 0, 0, withscores=False)
            sell_row = await self.r.zrange(self.sell_orders_key, 0, 0, withscores=False)
            
            if not buy_row or not sell_row:
                return

            best_buy = MarketOrder(**json.loads(buy_row[0]))
            best_sell = MarketOrder(**json.loads(sell_row[0]))
            
            if best_buy.price < best_sell.price:
                return

            trade_price = (best_buy.price + best_sell.price) / 2
            size = min(best_buy.amount - best_buy.filled, best_sell.amount - best_sell.filled)
            
            if size <= 0:
                # Clean zeroed orders
                await self.r.zrem(self.buy_orders_key, buy_row[0])
                await self.r.zrem(self.sell_orders_key, sell_row[0])
                continue

            # Settle: release buyer escrow, pay seller
            notional = size * trade_price
            
            # Load escrow
            locked_str = await self.r.hget(self.escrow_key, best_buy.order_id)
            locked = float(locked_str) if locked_str else 0.0
            
            if locked + 1e-9 < notional:
                # Buyer can't cover (race/corner case) → cancel buy
                await self.r.zrem(self.buy_orders_key, buy_row[0])
                await self.r.hdel(self.escrow_key, best_buy.order_id)
                if locked > 0:
                    await self.ledger.credit(best_buy.agent_id, locked, f"escrow_refund:{best_buy.order_id}")
                continue

            # Consume escrow -> pay seller
            remain = locked - notional
            await self.ledger.credit(best_sell.agent_id, notional, f"trade:{best_buy.order_id}->{best_sell.order_id}")

            # Update fills
            best_buy.filled += size
            best_sell.filled += size

            # Persist updated orders or remove if done
            if best_buy.filled + 1e-9 >= best_buy.amount:
                await self.r.zrem(self.buy_orders_key, buy_row[0])
                # Refund leftover escrow
                if remain > 0:
                    await self.ledger.credit(best_buy.agent_id, remain, f"escrow_refund:{best_buy.order_id}")
                await self.r.hdel(self.escrow_key, best_buy.order_id)
            else:
                await self.r.zadd(self.buy_orders_key, {json.dumps(asdict(best_buy)): best_buy.price})
                await self.r.zrem(self.buy_orders_key, buy_row[0])
                # Keep escrow with new remaining
                await self.r.hset(self.escrow_key, best_buy.order_id, str(remain))

            if best_sell.filled + 1e-9 >= best_sell.amount:
                await self.r.zrem(self.sell_orders_key, sell_row[0])
            else:
                await self.r.zadd(self.sell_orders_key, {json.dumps(asdict(best_sell)): best_sell.price})
                await self.r.zrem(self.sell_orders_key, sell_row[0])

            # Record trade & headline price
            trade = {
                "buyer": best_buy.agent_id, 
                "seller": best_sell.agent_id,
                "amount": size, 
                "price": trade_price, 
                "timestamp": time.time()
            }
            await self.r.lpush(self.trade_history_key, json.dumps(trade))
            await self.r.set("bel_market_price", str(trade_price))
    
    async def get_market_price(self) -> float:
        """Get current market price of ROOT"""
        price_str = await self.r.get("bel_market_price")
        return float(price_str) if price_str else 1.0

    async def cancel_order(self, agent_id: str, order_id: str) -> bool:
        """Cancel an order and refund escrow if applicable"""
        # Check buy orders first
        buy_orders = await self.r.zrange(self.buy_orders_key, 0, -1, withscores=False)
        for order_json in buy_orders:
            order = MarketOrder(**json.loads(order_json))
            if order.order_id == order_id and order.agent_id == agent_id:
                await self.r.zrem(self.buy_orders_key, order_json)
                # Refund escrow
                locked_str = await self.r.hget(self.escrow_key, order_id)
                if locked_str:
                    locked = float(locked_str)
                    await self.ledger.credit(agent_id, locked, f"order_cancel:{order_id}")
                    await self.r.hdel(self.escrow_key, order_id)
                return True
        
        # Check sell orders
        sell_orders = await self.r.zrange(self.sell_orders_key, 0, -1, withscores=False)
        for order_json in sell_orders:
            order = MarketOrder(**json.loads(order_json))
            if order.order_id == order_id and order.agent_id == agent_id:
                await self.r.zrem(self.sell_orders_key, order_json)
                return True
        
        return False

class RiskManager:
    def __init__(self, redis_client):
        self.r = redis_client
    
    async def assess_systemic_risk(self, system_state: SystemState) -> float:
        """Calculate system-wide risk level"""
        if system_state.total_supply == 0:
            return 0.0
        
        leverage_risk = system_state.total_debt / system_state.total_supply
        concentration_risk = await self._calculate_concentration_risk()
        
        return min(1.0, leverage_risk + concentration_risk)
    
    async def _calculate_concentration_risk(self) -> float:
        """Calculate risk from wealth concentration"""
        # Simplified - in practice would use Gini coefficient
        return 0.1  # Placeholder
    
    async def should_trigger_circuit_breaker(self, system_state: SystemState) -> bool:
        """Determine if circuit breaker should activate"""
        risk_level = await self.assess_systemic_risk(system_state)
        return risk_level > CFG["systemic_risk_threshold"] or system_state.circuit_breaker_active

class EconomicEngine:
    def __init__(self, redis_client, ledger):
        self.r = redis_client
        self.ledger = ledger  # Store ledger reference
        self.market = Market(redis_client, ledger)
        self.risk_manager = RiskManager(redis_client)
        self.system_state_key = "bel_system_state"
    
    async def calculate_adaptive_roi(self, system_state: SystemState) -> float:
        """Calculate system-wide ROI based on economic conditions"""
        base_roi = CFG["base_roi"]
        
        # Adjust for system leverage
        leverage_penalty = system_state.system_leverage * 0.5
        
        # Adjust for scarcity
        scarcity_bonus = (2.0 - system_state.scarcity_factor) * 0.01
        
        # Market volatility adjustment
        market_price = await self.market.get_market_price()
        volatility_adj = (market_price - 1.0) * CFG["roi_volatility"]
        
        return max(0.001, base_roi - leverage_penalty + scarcity_bonus + volatility_adj)
    
    async def update_scarcity(self, system_state: SystemState):
        """Update scarcity multiplier based on supply/demand"""
        if system_state.total_supply > 0:
            demand_pressure = system_state.total_staked / system_state.total_supply
            system_state.scarcity_factor = 1.0 + (demand_pressure * 0.5)
        else:
            system_state.scarcity_factor = 1.0
    
    async def mint_new_supply(self, system_state: SystemState) -> float:
        """Create new ROOT based on economic conditions"""
        if system_state.total_supply == 0:
            return 0.0
        
        growth_rate = CFG["supply_growth_rate"] / system_state.scarcity_factor
        new_supply = system_state.total_supply * growth_rate
        system_state.total_supply += new_supply
        return new_supply

class Ledger:
    def __init__(self, r):
        self.r = r
        self.key = "belv3_ledger"
        self.engine = EconomicEngine(r, self)  # Pass self reference
        self.system_state_key = "bel_system_state"
    
    async def _load(self) -> Dict[str, LedgerEntry]:
        raw = await self.r.get(self.key)
        data = json.loads(raw or "{}")
        ledger = {}
        for k, v in data.items():
            entry = LedgerEntry(agent_id=k)
            for field, value in v.items():
                if hasattr(entry, field):
                    setattr(entry, field, value)
            ledger[k] = entry
        return ledger
    
    async def _save(self, data: Dict[str, LedgerEntry]):
        serialized = {k: asdict(v) for k, v in data.items()}
        await self.r.setex(self.key, CFG["ledger_ttl"], json.dumps(serialized))
    
    async def _load_system_state(self) -> SystemState:
        raw = await self.r.get(self.system_state_key)
        if raw:
            data = json.loads(raw)
            return SystemState(**data)
        return SystemState()
    
    async def _save_system_state(self, state: SystemState):
        await self.r.set(self.system_state_key, json.dumps(asdict(state)))
    
    async def balance(self, agent_id: str) -> float:
        ledger = await self._load()
        entry = ledger.get(agent_id)
        return entry.balance if entry else 0.0
    
    async def credit(self, agent_id: str, amount: float, reason: str):
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        entry = ledger.setdefault(agent_id, LedgerEntry(agent_id=agent_id))
        
        # Apply scarcity multiplier to earnings
        adjusted_amount = amount * system_state.scarcity_factor
        
        entry.balance += adjusted_amount
        entry.earned += adjusted_amount
        entry.reputation = min(1.0, entry.reputation + 0.01)
        entry.last_seen = time.time()
        
        # Update system state
        system_state.total_supply += adjusted_amount
        
        await self._save(ledger)
        await self._save_system_state(system_state)
        
        await self.r.publish(CFG["gossip"], json.dumps({
            "action": "credit", 
            "agent": agent_id, 
            "amount": adjusted_amount, 
            "reason": reason
        }))
    
    async def debit(self, agent_id: str, amount: float, reason: str) -> bool:
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        entry = ledger.get(agent_id)
        if not entry or entry.status != AgentStatus.ACTIVE.value:
            return False
        
        # Check circuit breaker
        if await self.engine.risk_manager.should_trigger_circuit_breaker(system_state):
            return False
        
        available = entry.balance + entry.stake
        if available < amount:
            return False
        
        entry.balance -= amount
        entry.spent += amount
        entry.last_seen = time.time()
        
        # Handle debt and bankruptcy - FIXED CONDITION
        if entry.balance < 0:
            entry.debt += abs(entry.balance)
            
            # Fixed: Compare debt to absolute value of threshold
            if entry.debt > abs(CFG["bankruptcy_threshold"]):
                await self._declare_bankruptcy(agent_id, entry, system_state)
                return False
        
        entry.reputation = max(0.0, entry.reputation - 0.01)
        
        await self._save(ledger)
        await self._save_system_state(system_state)
        
        await self.r.publish(CFG["gossip"], json.dumps({
            "action": "debit", 
            "agent": agent_id, 
            "amount": amount, 
            "reason": reason
        }))
        return True
    
    async def _declare_bankruptcy(self, agent_id: str, entry: LedgerEntry, system_state: SystemState):
        """Handle agent bankruptcy with proper procedures"""
        entry.status = AgentStatus.BANKRUPT.value
        entry.bankruptcy_count += 1
        
        # Liquidate staked assets
        liquidated = entry.stake * 0.8  # 20% haircut
        entry.balance += liquidated
        entry.stake = 0
        
        # Update systemic debt
        system_state.total_debt += entry.debt
        system_state.bankrupt_agents += 1
        
        await self.r.publish(CFG["gossip"], json.dumps({
            "action": "bankruptcy",
            "agent": agent_id,
            "debt": entry.debt,
            "liquidated": liquidated
        }))
    
    async def transfer(self, from_agent: str, to_agent: str, amount: float, reason: str) -> bool:
        """Atomic transfer with market price consideration"""
        async with self.r.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(self.key)
                    ledger = await self._load()
                    
                    src = ledger.get(from_agent)
                    if not src or src.balance < amount or src.status != AgentStatus.ACTIVE.value:
                        await pipe.unwatch()
                        return False
                    
                    dst = ledger.setdefault(to_agent, LedgerEntry(agent_id=to_agent))
                    
                    # Fixed: Use flat 1% fee instead of market-price scaling
                    fee = amount * 0.01
                    
                    src.balance -= amount
                    src.spent += amount
                    src.last_seen = time.time()
                    src.reputation = max(0.0, src.reputation - 0.005)
                    
                    dst.balance += (amount - fee)
                    dst.earned += (amount - fee)
                    dst.last_seen = time.time()
                    dst.reputation = min(1.0, dst.reputation + 0.005)
                    
                    pipe.multi()
                    pipe.set(self.key, json.dumps({k: asdict(v) for k, v in ledger.items()}))
                    await pipe.execute()
                    
                    await self.r.publish(CFG["gossip"], json.dumps({
                        "action": "transfer",
                        "from": from_agent,
                        "to": to_agent,
                        "amount": amount,
                        "fee": fee,
                        "reason": reason
                    }))
                    return True
                    
                except WatchError:  # Fixed import
                    continue
    
    async def stake(self, agent_id: str, amount: float) -> bool:
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        entry = ledger.get(agent_id)
        if not entry or entry.balance < amount or entry.status != AgentStatus.ACTIVE.value:
            return False
        
        entry.balance -= amount
        entry.stake += amount
        entry.last_seen = time.time()
        
        # Staking earns reputation
        entry.reputation = min(1.0, entry.reputation + 0.02)
        
        # Update system state
        system_state.total_staked += amount
        
        await self._save(ledger)
        await self._save_system_state(system_state)
        
        await self.r.publish(CFG["gossip"], json.dumps({
            "action": "stake",
            "agent": agent_id,
            "amount": amount
        }))
        return True
    
    async def unstake(self, agent_id: str, amount: float) -> bool:
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        entry = ledger.get(agent_id)
        if not entry or entry.stake < amount:
            return False
        
        entry.stake -= amount
        entry.balance += amount
        entry.last_seen = time.time()
        
        # Update system state
        system_state.total_staked -= amount
        
        await self._save(ledger)
        await self._save_system_state(system_state)
        
        await self.r.publish(CFG["gossip"], json.dumps({
            "action": "unstake",
            "agent": agent_id,
            "amount": amount
        }))
        return True
    
    async def apply_roi(self):
        """Apply adaptive ROI to all staked amounts"""
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        # Calculate current system ROI
        current_roi = await self.engine.calculate_adaptive_roi(system_state)
        
        total_rewards = 0
        for entry in ledger.values():
            if entry.stake > 0 and entry.status == AgentStatus.ACTIVE.value:
                # Individual ROI can vary based on reputation
                individual_roi = current_roi * (0.5 + entry.reputation * 0.5)
                reward = entry.stake * individual_roi
                
                entry.balance += reward
                entry.earned += reward
                entry.roi_rate = individual_roi
                total_rewards += reward
        
        # Mint new supply if needed
        new_supply = await self.engine.mint_new_supply(system_state)
        if new_supply > total_rewards:
            # Distribute excess to active agents
            active_agents = [e for e in ledger.values() if e.status == AgentStatus.ACTIVE.value]
            if active_agents:
                bonus_per_agent = (new_supply - total_rewards) / len(active_agents)
                for entry in active_agents:
                    entry.balance += bonus_per_agent
        
        system_state.last_roi_update = time.time()
        # Fixed: Update total supply with rewards
        system_state.total_supply += total_rewards
        
        await self._save(ledger)
        await self._save_system_state(system_state)
    
    async def decay_reputation(self):
        """Apply reputation decay and update system metrics"""
        ledger = await self._load()
        system_state = await self._load_system_state()
        
        # Update system metrics
        system_state.active_agents = sum(1 for e in ledger.values() if e.status == AgentStatus.ACTIVE.value)
        system_state.total_supply = sum(e.balance + e.stake for e in ledger.values())
        system_state.total_debt = sum(e.debt for e in ledger.values())
        system_state.system_leverage = system_state.total_debt / max(system_state.total_supply, 1.0)
        
        # Apply reputation decay
        for entry in ledger.values():
            entry.reputation *= CFG["reputation_decay"]
            
            # Rehabilitation for bankrupt agents
            if entry.status == AgentStatus.BANKRUPT.value and entry.reputation > 0.5:
                entry.status = AgentStatus.ACTIVE.value
                entry.debt *= 0.9  # Reduce debt for good behavior
        
        # Update scarcity
        await self.engine.update_scarcity(system_state)
        
        await self._save(ledger)
        await self._save_system_state(system_state)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        system_state = await self._load_system_state()
        risk_level = await self.engine.risk_manager.assess_systemic_risk(system_state)
        market_price = await self.engine.market.get_market_price()
        
        return {
            "total_supply": system_state.total_supply,
            "total_staked": system_state.total_staked,
            "total_debt": system_state.total_debt,
            "active_agents": system_state.active_agents,
            "bankrupt_agents": system_state.bankrupt_agents,
            "system_leverage": system_state.system_leverage,
            "risk_level": risk_level,
            "market_price": market_price,
            "scarcity_factor": system_state.scarcity_factor,
            "circuit_breaker": system_state.circuit_breaker_active
        }

class EconomicV3Mixin:
    def __init__(self, kernel):
        self.r = kernel.swarm.redis
        self.ledger = Ledger(self.r)
        self.agent_id = str(uuid.uuid4())
    
    async def earn(self, amount: float, reason: str):
        await self.ledger.credit(self.agent_id, amount, reason)
    
    async def pay(self, amount: float, to: str, reason: str) -> bool:
        return await self.ledger.transfer(self.agent_id, to, amount, reason)
    
    async def balance(self) -> float:
        return await self.ledger.balance(self.agent_id)
    
    async def stake(self, amount: float) -> bool:
        return await self.ledger.stake(self.agent_id, amount)
    
    async def unstake(self, amount: float) -> bool:
        return await self.ledger.unstake(self.agent_id, amount)
    
    # v2 Market features
    async def place_buy_order(self, amount: float, price: float) -> bool:
        order = MarketOrder(
            agent_id=self.agent_id,
            order_id=str(uuid.uuid4()),
            side="buy",
            amount=amount,
            price=price,
            timestamp=time.time()
        )
        return await self.ledger.engine.market.place_order(order)
    
    async def place_sell_order(self, amount: float, price: float) -> bool:
        order = MarketOrder(
            agent_id=self.agent_id,
            order_id=str(uuid.uuid4()),
            side="sell", 
            amount=amount,
            price=price,
            timestamp=time.time()
        )
        return await self.ledger.engine.market.place_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        return await self.ledger.engine.market.cancel_order(self.agent_id, order_id)
    
    async def get_market_price(self) -> float:
        return await self.ledger.engine.market.get_market_price()
    
    # v3 System features
    async def system_health(self) -> Dict[str, Any]:
        return await self.ledger.get_system_metrics()
    
    async def my_status(self) -> Dict[str, Any]:
        ledger = await self.ledger._load()
        entry = ledger.get(self.agent_id)
        if not entry:
            return {"status": "not_found"}
        
        return {
            "balance": entry.balance,
            "stake": entry.stake,
            "reputation": entry.reputation,
            "status": entry.status,
            "debt": entry.debt,
            "roi_rate": entry.roi_rate,
            "earned_total": entry.earned,
            "spent_total": entry.spent
        }

# System maintenance tasks - FIXED TIMING
async def economic_cycle_task(ledger: Ledger):
    """Background task to maintain economic system"""
    while True:
        try:
            # Apply ROI every 60 seconds
            await ledger.apply_roi()
            await asyncio.sleep(60)
            
            # Reputation decay every 60 more seconds (120 total)
            await ledger.decay_reputation()
            await asyncio.sleep(60)
            
            # Full system health check every 180 more seconds (300 total)
            metrics = await ledger.get_system_metrics()
            logging.info(f"Economic cycle: {metrics}")
            await asyncio.sleep(180)
            
        except Exception as e:
            logging.error(f"Economic cycle error: {e}")
            await asyncio.sleep(60)