# agent_economy.py
# A safe, capability-agnostic internal economy for distributed agents.
# Models credits, pricing, ROI, trust, and a 12-month simulation.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from agent_core_anatomy import AgentCore, MissionStatus
import math
import random

# ----------------------------
# Core types
# ----------------------------
@dataclass
class AgentAccount:
    agent_id: str
    credits: float = 0.0
    trust: float = 0.50          # 0..1, updated by fulfillment quality
    capacity_compute: float = 0  # unit-hours available per month
    capacity_storage: float = 0  # GB-month available per month
    operating_cost: float = 0.0  # credits/month
    revenue: float = 0.0         # tracked per tick
    expense: float = 0.0         # tracked per tick
    status: MissionStatus = MissionStatus.PENDING

    def reset_month(self):
        self.revenue = 0.0
        self.expense = 0.0

@dataclass
class Order:
    buyer: str
    seller: Optional[str]     # None -> go through market matching
    resource: str            # "compute" | "storage" | "data_prep" (example)
    quantity: float          # unit-hours or GB-month
    max_price: float         # buyer’s max willingness per unit
    month: int

@dataclass
class Fill:
    order: Order
    seller: str
    unit_price: float
    quantity_filled: float
    quality: float           # 0..1 quality score affecting trust/penalties

    status: MissionStatus = MissionStatus.PENDING
# ----------------------------
# Pricing and ROI policies
# ----------------------------
@dataclass
class PricingPolicy:
    base_price_compute: float = 1.0     # credits per unit-hour
    base_price_storage: float = 0.10    # credits per GB-month
    price_sensitivity: float = 0.25     # reacts to utilization

    def quote(self, resource: str, seller_utilization: float) -> float:
        if resource == "compute":
            base = self.base_price_compute
        elif resource == "storage":
            base = self.base_price_storage
        else:
            base = 0.5  # generic service
        # simple surge/discount relative to utilization (0..1)
        return max(0.01, base * (1.0 + self.price_sensitivity * (seller_utilization - 0.50)))

@dataclass
class ROIModel:
    # Simple ROI: (revenue - cost) / (cost + epsilon)
    def monthly_roi(self, acct: AgentAccount) -> float:
        denom = max(1e-6, acct.expense + acct.operating_cost)
        return (acct.revenue - (acct.expense + acct.operating_cost)) / denom

# ----------------------------
# Ledger & Market
# ----------------------------
@dataclass
class Ledger:
    balances: Dict[str, float] = field(default_factory=dict)

    def ensure(self, agent_id: str):
        self.balances.setdefault(agent_id, 0.0)

    def credit(self, agent_id: str, amount: float):
        self.ensure(agent_id)
        self.balances[agent_id] += amount

    def debit(self, agent_id: str, amount: float) -> bool:
        self.ensure(agent_id)
        if self.balances[agent_id] >= amount:
            self.balances[agent_id] -= amount
            return True
        return False

@dataclass
class Market:
    accounts: Dict[str, AgentAccount]
    ledger: Ledger
    pricing: PricingPolicy

    def seller_candidates(self, resource: str) -> List[AgentAccount]:
        sellers = []
        for a in self.accounts.values():
            cap = a.capacity_compute if resource == "compute" else a.capacity_storage if resource == "storage" else (a.capacity_compute + a.capacity_storage) * 0.25
            if cap > 0:
                sellers.append(a)
        # higher trust first, then lower utilization
        return sorted(
            sellers,
            key=lambda s: (-s.trust, self._utilization(s, resource))
        )

    def _utilization(self, acct: AgentAccount, resource: str) -> float:
        # Proxy utilization: revenue vs notional max at base price
        if resource == "compute":
            base = self.pricing.base_price_compute
            max_rev = acct.capacity_compute * base
        elif resource == "storage":
            base = self.pricing.base_price_storage
            max_rev = acct.capacity_storage * base
        else:
            base = 0.5
            max_rev = (acct.capacity_compute + acct.capacity_storage) * 0.25 * base
        return 0.0 if max_rev <= 0 else min(1.0, acct.revenue / max_rev)

    def match_and_fill(self, order: Order) -> Optional[Fill]:
        if order.quantity <= 0: 
            return None

        # Direct seller specified?
        if order.seller and order.seller in self.accounts:
            seller_acct = self.accounts[order.seller]
            return self._attempt_fill(order, seller_acct)

        # Otherwise pick best candidate
        for seller in self.seller_candidates(order.resource):
            fill = self._attempt_fill(order, seller)
            if fill:
                return fill
        return None

    def _attempt_fill(self, order: Order, seller: AgentAccount) -> Optional[Fill]:
        util = self._utilization(seller, order.resource)
        ask = self.pricing.quote(order.resource, util)
        if ask > order.max_price:
            return None

        qty_avail = seller.capacity_compute if order.resource == "compute" else seller.capacity_storage if order.resource == "storage" else min(seller.capacity_compute, seller.capacity_storage) * 0.25
        qty = min(order.quantity, qty_avail)
        if qty <= 0:
            return None

        cost = qty * ask
        # Buyer pays if enough credits
        if not self.ledger.debit(order.buyer, cost):
            return None

        self.ledger.credit(seller.agent_id, cost)

        # Update monthly tallies
        buyer = self.accounts[order.buyer]
        buyer.expense += cost

        seller.revenue += cost
        # Burn capacity used
        if order.resource == "compute":
            seller.capacity_compute -= qty
        elif order.resource == "storage":
            seller.capacity_storage -= qty
        else:
            burn = qty  # generic
            seller.capacity_compute = max(0, seller.capacity_compute - burn * 0.5)
            seller.capacity_storage = max(0, seller.capacity_storage - burn * 0.5)

        # Determine fulfillment quality (noise added, trust affects it too)
        quality = max(0.0, min(1.0, random.gauss(0.85 * seller.trust + 0.10, 0.08)))
        # Trust updates
        seller.trust = max(0.0, min(1.0, 0.90 * seller.trust + 0.10 * quality))
        buyer.trust  = max(0.0, min(1.0, 0.98 * buyer.trust  + 0.02 * quality))

        return Fill(order=order, seller=seller.agent_id, unit_price=ask, quantity_filled=qty, quality=quality)

# ----------------------------
# One-year simulation
# ----------------------------
@dataclass
class EconomySim:
    accounts: Dict[str, AgentAccount]
    market: Market
    ledger: Ledger
    roi: ROIModel
    months: int = 12
    history: List[Dict] = field(default_factory=list)

    def run(self) -> List[Dict]:
        for m in range(1, self.months + 1):
            # Monthly reset & operating costs
            for acct in self.accounts.values():
                acct.reset_month()
                # charge operating cost
                if self.ledger.debit(acct.agent_id, acct.operating_cost):
                    acct.expense += acct.operating_cost
                # refresh capacity (replenish for new month)
                # model scale-up/down based on last month ROI and trust
                scale = 1.0 + 0.25 * math.tanh(self.roi.monthly_roi(acct)) + 0.10 * (acct.trust - 0.5)
                acct.capacity_compute = max(0.0, acct.capacity_compute * scale + random.uniform(-0.1, 0.1))
                acct.capacity_storage = max(0.0, acct.capacity_storage * scale + random.uniform(-1.0, 1.0))

            # Demand generation (benign workloads)
            orders = self._generate_orders(m)

            # Match & fill
            fills: List[Fill] = []
            for od in orders:
                filled = self.market.match_and_fill(od)
                if filled:
                    fills.append(filled)

            # Record month summary
            snapshot = {
                "month": m,
                "balances": {k: round(v, 2) for k, v in self.ledger.balances.items()},
                "revenue": {a.agent_id: round(a.revenue, 2) for a in self.accounts.values()},
                "expense": {a.agent_id: round(a.expense, 2) for a in self.accounts.values()},
                "trust":   {a.agent_id: round(a.trust, 3) for a in self.accounts.values()},
                "roi":     {a.agent_id: round(self.roi.monthly_roi(a), 3) for a in self.accounts.values()},
                "fills":   [dict(buyer=f.order.buyer, seller=f.seller, res=f.order.resource, qty=round(f.quantity_filled,2), price=round(f.unit_price,2), q=round(f.quality,2)) for f in fills],
            }
            self.history.append(snapshot)
        return self.history

    def _generate_orders(self, month: int) -> List[Order]:
        agents = list(self.accounts.keys())
        orders: List[Order] = []
        # Seasonality: Q4 bump
        demand_factor = 1.0 + (0.25 if month in (10, 11, 12) else 0.0)

        for buyer_id in agents:
            # Each agent may place a few benign service orders per month
            for _ in range(random.randint(1, 3)):
                res = random.choice(["compute", "storage", "data_prep"])
                qty = (random.uniform(1, 5) if res == "compute" else
                       random.uniform(10, 100) if res == "storage" else
                       random.uniform(0.5, 2.0))
                qty *= demand_factor

                max_price = (self.market.pricing.base_price_compute * random.uniform(0.9, 1.3) if res == "compute" else
                             self.market.pricing.base_price_storage * random.uniform(0.9, 1.5) if res == "storage" else
                             0.5 * random.uniform(0.8, 1.4))

                orders.append(Order(
                    buyer=buyer_id,
                    seller=None,
                    resource=res,
                    quantity=qty,
                    max_price=max_price,
                    month=month
                ))
        random.shuffle(orders)
        return orders

# ----------------------------
# Helper to bootstrap a tiny economy
# ----------------------------

def make_demo_economy() -> Tuple[Dict[str, AgentAccount], Market, Ledger, EconomySim]:
    random.seed(7)
    accounts = {
        "alpha": AgentAccount("alpha", credits=200.0, trust=0.6, capacity_compute=12, capacity_storage=120, operating_cost=8, status=MissionStatus.PENDING),
        "bravo": AgentAccount("bravo", credits=150.0, trust=0.5, capacity_compute=8,  capacity_storage=60,  operating_cost=6, status=MissionStatus.PENDING),
        "charlie": AgentAccount("charlie", credits=120.0, trust=0.55, capacity_compute=10, capacity_storage=80, operating_cost=7, status=MissionStatus.PENDING),
    }
    ledger = Ledger({aid: acct.credits for aid, acct in accounts.items()})
    pricing = PricingPolicy()
    market = Market(accounts, ledger, pricing)
    roi = ROIModel()
    sim = EconomySim(accounts, market, ledger, roi, months=12)
    return accounts, market, ledger, sim

if __name__ == "__main__":
    _, _, _, sim = make_demo_economy()
    history = sim.run()
    # Print final month summary
    print(history[-1])
