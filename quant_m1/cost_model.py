"""
Institutional cost accounting.

CRITICAL:
  Corwin-Schultz and Roll estimate FULL quoted spread S.
  Buy pays S/2 above mid.
  Sell receives S/2 below mid.
  Round-trip spread cost = S, NOT 2S.

Impact:
  I ~= Y * sigma_daily * sqrt(Q / V)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float
    t1_drag_bps: float
    total_bps: float


def corwin_schultz_full_spread(highs, lows) -> float:
    """Return FULL spread as a fraction of price, not half-spread."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    if h.shape != l.shape or h.ndim != 1 or len(h) < 2:
        raise ValueError("need 1-D highs/lows with length >= 2")
    if np.any(h <= 0) or np.any(l <= 0) or np.any(h < l):
        raise ValueError("invalid high/low prices")

    beta = float(np.mean((h - l) ** 2))
    gamma = float(np.mean((np.maximum(h[1:], h[:-1]) - np.minimum(l[1:], l[:-1])) ** 2))
    alpha = (2.0 * beta - gamma) / (3.0 - 2.0 * math.sqrt(2.0))
    alpha = max(alpha, 0.0)
    ea = math.exp(alpha)
    spread = 2.0 * (ea - 1.0) / (1.0 + ea)
    return float(min(max(spread, 0.0), 1.0))


def roll_full_spread(prices) -> float | None:
    """Roll (1984) FULL spread as fraction of price. None if undefined."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 3 or np.any(p <= 0):
        raise ValueError("need >= 3 positive prices")
    dp = np.diff(p)
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    if cov >= 0:
        return None
    return float(2.0 * math.sqrt(-cov) / float(np.mean(p)))


def round_trip_cost_bps(
    full_spread_frac: float,
    notional_buy: float,
    notional_sell: float,
    commission_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
    impact_bps: float = 0.0,
    t1_drag_bps: float = 0.0,
) -> CostBreakdown:
    if full_spread_frac < 0:
        raise ValueError("spread must be >= 0")
    if notional_buy <= 0 or notional_sell <= 0:
        raise ValueError("notionals must be positive")
    if commission_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("per-side costs must be >= 0")
    if impact_bps < 0 or t1_drag_bps < 0:
        raise ValueError("impact/drag must be >= 0")

    half = full_spread_frac / 2.0
    spread_usd = notional_buy * half + notional_sell * half
    avg_notional = (notional_buy + notional_sell) / 2.0
    spread_bps = spread_usd / avg_notional * 1e4

    commission = 2.0 * commission_bps_per_side
    slippage = 2.0 * slippage_bps_per_side
    total = spread_bps + commission + slippage + impact_bps + t1_drag_bps

    return CostBreakdown(
        commission_bps=commission,
        spread_bps=spread_bps,
        slippage_bps=slippage,
        impact_bps=impact_bps,
        t1_drag_bps=t1_drag_bps,
        total_bps=total,
    )


def square_root_impact_bps(
    participation_rate: float,
    daily_vol: float,
    Y: float = 0.5,
    max_participation: float = 0.10,
) -> float:
    if participation_rate < 0:
        raise ValueError("participation must be >= 0")
    if participation_rate > max_participation:
        raise ValueError("participation exceeds cap")
    if daily_vol <= 0 or Y <= 0:
        raise ValueError("daily_vol and Y must be positive")
    impact_frac = Y * daily_vol * math.sqrt(participation_rate)
    return impact_frac * 1e4


def capacity_notional(
    alpha_bps: float,
    daily_vol: float,
    adv_notional: float,
    Y: float = 0.5,
    max_participation: float = 0.10,
) -> float:
    """Max notional before square-root impact consumes alpha."""
    if alpha_bps <= 0:
        return 0.0
    if daily_vol <= 0 or adv_notional <= 0 or Y <= 0:
        raise ValueError("daily_vol, adv_notional, Y must be positive")
    alpha_frac = alpha_bps / 1e4
    q = adv_notional * (alpha_frac / (Y * daily_vol)) ** 2
    return float(min(q, max_participation * adv_notional))
