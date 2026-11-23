"""M1.3 cost model.

VERIFIED FINDING: Corwin-Schultz from DAILY bars is INADMISSIBLE for liquid
US ETFs. Zero-spread control at 1.2%/day vol returns ~44bps while true SPY
spread is ~1bp. CS is therefore gated: value_frac=None unless admissible.

CS and Roll estimate the FULL spread S. Round-trip spread cost = S, NOT 2S.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

K_CS = 3.0 - 2.0 * math.sqrt(2.0)
MIN_SPREAD_RANGE_RATIO = 0.30


class SpreadSource(Enum):
    MEASURED_NBBO = "measured_nbbo"
    ISSUER_PUBLISHED = "issuer_published"
    ASSUMED_CONSERVATIVE = "assumed_conservative"
    ESTIMATED_CS = "estimated_corwin_schultz"
    ESTIMATED_ROLL = "estimated_roll"


class InadmissibleSpread(Exception):
    pass


@dataclass(frozen=True)
class SpreadEstimate:
    value_frac: Optional[float]
    source: SpreadSource
    admissible: bool
    spread_range_ratio: Optional[float]
    negative_clip_frac: Optional[float]
    note: str

    def require(self) -> float:
        if not self.admissible or self.value_frac is None:
            raise InadmissibleSpread(self.note)
        return self.value_frac


def corwin_schultz_raw(highs, lows) -> np.ndarray:
    h = np.asarray(highs, float)
    l = np.asarray(lows, float)
    if h.shape != l.shape or h.ndim != 1 or len(h) < 2:
        raise ValueError("need 1-D highs/lows, length >= 2")
    if np.any(h <= 0) or np.any(l <= 0) or np.any(h < l):
        raise ValueError("invalid high/low prices")
    lr = np.log(h / l)
    beta = lr[:-1] ** 2 + lr[1:] ** 2
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(l[:-1], l[1:])) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
    return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def corwin_schultz_full_spread(highs, lows, *,
                               min_ratio: float = MIN_SPREAD_RANGE_RATIO) -> SpreadEstimate:
    raw = corwin_schultz_raw(highs, lows)
    est = float(np.maximum(raw, 0.0).mean())
    h = np.asarray(highs, float)
    l = np.asarray(lows, float)
    mean_range = float(np.mean(np.log(h / l)))
    ratio = est / mean_range if mean_range > 0 else 0.0
    ok = ratio >= min_ratio
    return SpreadEstimate(
        est if ok else None, SpreadSource.ESTIMATED_CS, ok, ratio,
        float((raw < 0).mean()),
        "admissible" if ok else
        f"INADMISSIBLE: spread/range={ratio:.4f} < {min_ratio}. "
        f"Estimator returns its noise floor. Use measured spread.")


def roll_full_spread(closes) -> SpreadEstimate:
    c = np.asarray(closes, float)
    if c.ndim != 1 or len(c) < 4 or np.any(c <= 0):
        raise ValueError("need >= 4 positive closes")
    dp = np.diff(c)
    cov = float(np.mean(dp[1:] * dp[:-1]) - np.mean(dp[1:]) * np.mean(dp[:-1]))
    if cov >= 0.0:
        return SpreadEstimate(None, SpreadSource.ESTIMATED_ROLL, False, None, None,
                              "UNDEFINED: serial covariance >= 0")
    est = 2.0 * math.sqrt(-cov) / float(c.mean())
    return SpreadEstimate(est, SpreadSource.ESTIMATED_ROLL, False, None, None,
                          "Roll on daily closes NOT admissible for liquid ETFs "
                          "(~24x overstatement, 65-70% undefined). Diagnostic only.")


_ASSUMED_SPREAD_BPS = {
    "SPY": 2.0, "QQQ": 2.0, "IVV": 3.0, "VOO": 3.0,
    "IWM": 5.0, "TLT": 5.0, "IEF": 5.0, "GLD": 5.0,
    "XLK": 8.0, "XLF": 8.0, "XLE": 8.0, "XLV": 8.0,
}


def assumed_spread(symbol: str, default_bps: float = 10.0) -> SpreadEstimate:
    bps = _ASSUMED_SPREAD_BPS.get(symbol.upper(), default_bps)
    return SpreadEstimate(bps / 1e4, SpreadSource.ASSUMED_CONSERVATIVE, True, None, None,
                          f"ASSUMED {bps}bps for {symbol}. NOT MEASURED. "
                          f"Replace with NBBO or issuer-published median spread.")


def measured_spread(bps: float,
                    source: SpreadSource = SpreadSource.MEASURED_NBBO) -> SpreadEstimate:
    if bps < 0:
        raise ValueError("spread bps must be >= 0")
    if source not in {SpreadSource.MEASURED_NBBO, SpreadSource.ISSUER_PUBLISHED}:
        raise ValueError("measured_spread requires a measured source")
    return SpreadEstimate(bps / 1e4, source, True, None, None, "measured")


@dataclass(frozen=True)
class CostBreakdown:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float
    t1_drag_bps: float
    total_bps: float
    spread_source: str


def round_trip_cost_bps(full_spread_frac: float, notional_buy: float, notional_sell: float,
                        commission_bps_per_side: float = 0.0,
                        slippage_bps_per_side: float = 0.0,
                        impact_bps: float = 0.0, t1_drag_bps: float = 0.0,
                        spread_source: str = "unspecified") -> CostBreakdown:
    if full_spread_frac < 0:
        raise ValueError("spread must be >= 0")
    if notional_buy <= 0 or notional_sell <= 0:
        raise ValueError("notionals must be positive")
    if min(commission_bps_per_side, slippage_bps_per_side, impact_bps, t1_drag_bps) < 0:
        raise ValueError("cost components must be >= 0")
    half = full_spread_frac / 2.0
    spread_usd = notional_buy * half + notional_sell * half
    avg = (notional_buy + notional_sell) / 2.0
    spread_bps = spread_usd / avg * 1e4
    comm = 2.0 * commission_bps_per_side
    slip = 2.0 * slippage_bps_per_side
    return CostBreakdown(comm, spread_bps, slip, impact_bps, t1_drag_bps,
                         spread_bps + comm + slip + impact_bps + t1_drag_bps, spread_source)


def square_root_impact_bps(participation_rate: float, daily_vol: float,
                           Y: float = 0.5, max_participation: float = 0.10) -> float:
    if participation_rate < 0:
        raise ValueError("participation >= 0")
    if participation_rate > max_participation:
        raise ValueError("participation exceeds cap")
    if daily_vol <= 0 or Y <= 0:
        raise ValueError("daily_vol and Y must be positive")
    return Y * daily_vol * math.sqrt(participation_rate) * 1e4


def capacity_notional(alpha_bps: float, daily_vol: float, adv_notional: float,
                      Y: float = 0.5, max_participation: float = 0.10) -> float:
    if alpha_bps <= 0:
        return 0.0
    if daily_vol <= 0 or adv_notional <= 0 or Y <= 0:
        raise ValueError("daily_vol, adv_notional, Y must be positive")
    q = adv_notional * ((alpha_bps / 1e4) / (Y * daily_vol)) ** 2
    return float(min(q, max_participation * adv_notional))
