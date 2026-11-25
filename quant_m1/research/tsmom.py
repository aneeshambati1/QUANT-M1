"""Time-Series Momentum (TSMOM) Falsification Strategy.

Implements a 12-month trend with a 1-month skip, volatility-targeted sizing,
and monthly rebalancing. This is the specific hypothesis recommended by the
red-team audit to test whether any edge remains in the 10-ETF panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tsmom_signals(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    Computes time-series momentum signals.
    +1 if the asset's price skip days ago is higher than lookback days ago.
    0 otherwise (long-only, cash is a valid position).
    """
    price_skip = prices.shift(skip)
    price_lookback = prices.shift(lookback)
    
    # Signal is +1 if recent trend is positive, else 0
    signals = (price_skip > price_lookback).astype(int)
    return signals


def compute_volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Computes trailing realized volatility (annualized)."""
    return returns.rolling(window).std() * np.sqrt(252)


def generate_tsmom_weights(prices: pd.DataFrame, target_vol: float = 0.10, 
                           lookback: int = 252, skip: int = 21, vol_window: int = 63) -> pd.DataFrame:
    """
    Generates target weights for the TSMOM strategy.
    1. Identify positive trend assets.
    2. Size them inversely to their realized volatility to target a specific portfolio vol.
    3. Cap individual asset weights to prevent concentration.
    """
    returns = prices.pct_change()
    signals = compute_tsmom_signals(prices, lookback, skip)
    realized_vol = compute_volatility(returns, vol_window)
    
    # Avoid division by zero
    realized_vol = realized_vol.replace(0, np.nan)
    
    # Volatility-parity sizing: target_vol / asset_vol
    vol_scalar = target_vol / realized_vol
    
    # Apply signals (0 if trend is negative)
    raw_weights = signals * vol_scalar
    
    # Cap individual asset weights at 30% to prevent concentration
    max_weight = 0.30
    capped_weights = raw_weights.clip(upper=max_weight)
    
    # Normalize so the sum of weights <= 1.0 (allow cash)
    weight_sums = capped_weights.sum(axis=1)
    scale_factor = np.where(weight_sums > 1.0, 1.0 / weight_sums, 1.0)
    
    final_weights = capped_weights.multiply(scale_factor, axis=0)
    
    # Only rebalance monthly (every 21 days) to minimize turnover
    monthly_mask = np.arange(len(prices)) % 21 == 0
    final_weights[~monthly_mask] = np.nan
    final_weights = final_weights.ffill()
    
    return final_weights.fillna(0.0)