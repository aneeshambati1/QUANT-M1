# quant_m1/alpha/controller.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from quant_m1.authority import Intent

@dataclass(frozen=True)
class MarketSnapshot:
    date: str
    prices: Dict[str, float]  # symbol -> close price
    volumes: Dict[str, float] # symbol -> ADV

class BaseStrategy:
    """
    Base class for all QUANT strategies.
    Subclasses must implement generate_intents().
    """
    def generate_intents(
        self, 
        snapshot: MarketSnapshot, 
        current_positions: Dict[str, float],
        cash_settled: float
    ) -> List[Intent]:
        raise NotImplementedError

class MomentumStrategy(BaseStrategy):
    """
    Example strategy that generates Intents for the Gateway.
    """
    def __init__(self, symbol: str, lookback_days: int, target_weight: float, cost_bps: int):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.target_weight = target_weight
        self.cost_bps = cost_bps
        self.history: List[float] = []

    def update_history(self, price: float):
        self.history.append(price)
        # Keep only what we need + 1 for return calculation
        if len(self.history) > self.lookback_days + 1:
            self.history.pop(0)

    def generate_intents(
        self, 
        snapshot: MarketSnapshot, 
        current_positions: Dict[str, float],
        cash_settled: float
    ) -> List[Intent]:
        self.update_history(snapshot.prices.get(self.symbol, 0.0))
        
        if len(self.history) < self.lookback_days + 1:
            return [] # Not enough data yet

        # Calculate momentum
        ret = (self.history[-1] / self.history[-self.lookback_days - 1]) - 1.0
        
        if ret <= 0:
            return [] # No edge, abstain

        # Calculate desired notional based on cash and target weight
        # (Simplified: assumes cash_settled is the investable equity)
        desired_notional = cash_settled * self.target_weight
        current_price = snapshot.prices[self.symbol]
        
        if current_price <= 0:
            return []

        desired_qty = int(desired_notional / current_price)
        current_qty = int(current_positions.get(self.symbol, 0.0))
        delta_qty = desired_qty - current_qty

        if delta_qty <= 0:
            return [] # Already at or above target

        # Generate the Intent for the Kernel to evaluate
        return [
            Intent(
                symbol=self.symbol,
                side="BUY",
                qty=delta_qty,
                price=current_price,
                cost=self.cost_bps
            )
        ]
