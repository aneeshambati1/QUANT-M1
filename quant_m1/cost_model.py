"""M1.3 cost model.

Corwin-Schultz from DAILY bars is inadmissible for liquid US ETFs when
the estimator is dominated by its noise floor.

CS and Roll estimate the FULL spread S.
Round-trip spread cost is S, not 2S.

All public numeric inputs are checked for finiteness. NaN and infinity
must never silently pass cost, impact, or capacity calculations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


K_CS = 3.0 - 2.0 * math.sqrt(2.0)
MIN_SPREAD_RANGE_RATIO = 0.30


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None

    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")

    return result


def _finite_array(name: str, values) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain finite numeric values") from None

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")

    return result


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

        value = _finite_number("spread estimate", self.value_frac)
        if value < 0:
            raise InadmissibleSpread("spread estimate is negative")

        return value


def corwin_schultz_raw(highs, lows) -> np.ndarray:
    highs = _finite_array("highs", highs)
    lows = _finite_array("lows", lows)

    if highs.shape != lows.shape or highs.ndim != 1 or len(highs) < 2:
        raise ValueError("need 1-D highs/lows, length >= 2")

    if np.any(highs <= 0) or np.any(lows <= 0) or np.any(highs < lows):
        raise ValueError("invalid high/low prices")

    log_range = np.log(highs / lows)

    beta = log_range[:-1] ** 2 + log_range[1:] ** 2

    gamma = np.log(
        np.maximum(highs[:-1], highs[1:])
        / np.minimum(lows[:-1], lows[1:])
    ) ** 2

    alpha = (
        (np.sqrt(2.0 * beta) - np.sqrt(beta)) / K_CS
        - np.sqrt(gamma / K_CS)
    )

    raw = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    if not np.all(np.isfinite(raw)):
        raise ValueError("Corwin-Schultz produced non-finite estimates")

    return raw


def corwin_schultz_full_spread(
    highs,
    lows,
    *,
    min_ratio: float = MIN_SPREAD_RANGE_RATIO,
) -> SpreadEstimate:
    min_ratio = _finite_number("min_ratio", min_ratio)

    if min_ratio < 0:
        raise ValueError("min_ratio must be >= 0")

    raw = corwin_schultz_raw(highs, lows)

    estimate = float(np.maximum(raw, 0.0).mean())

    highs = _finite_array("highs", highs)
    lows = _finite_array("lows", lows)

    mean_range = float(np.mean(np.log(highs / lows)))

    ratio = estimate / mean_range if mean_range > 0 else 0.0

    if not math.isfinite(estimate) or not math.isfinite(ratio):
        raise ValueError("Corwin-Schultz summary is non-finite")

    admissible = ratio >= min_ratio

    return SpreadEstimate(
        estimate if admissible else None,
        SpreadSource.ESTIMATED_CS,
        admissible,
        ratio,
        float((raw < 0).mean()),
        (
            "admissible"
            if admissible
            else (
                f"INADMISSIBLE: spread/range={ratio:.4f} < {min_ratio}. "
                "Estimator returns its noise floor. Use measured spread."
            )
        ),
    )


def roll_full_spread(closes) -> SpreadEstimate:
    closes = _finite_array("closes", closes)

    if closes.ndim != 1 or len(closes) < 4 or np.any(closes <= 0):
        raise ValueError("need >= 4 positive closes")

    differences = np.diff(closes)

    covariance = float(
        np.mean(differences[1:] * differences[:-1])
        - np.mean(differences[1:]) * np.mean(differences[:-1])
    )

    if not math.isfinite(covariance):
        raise ValueError("Roll covariance is non-finite")

    if covariance >= 0.0:
        return SpreadEstimate(
            None,
            SpreadSource.ESTIMATED_ROLL,
            False,
            None,
            None,
            "UNDEFINED: serial covariance >= 0",
        )

    estimate = 2.0 * math.sqrt(-covariance) / float(closes.mean())

    if not math.isfinite(estimate):
        raise ValueError("Roll spread estimate is non-finite")

    return SpreadEstimate(
        estimate,
        SpreadSource.ESTIMATED_ROLL,
        False,
        None,
        None,
        (
            "Roll on daily closes NOT admissible for liquid ETFs "
            "(~24x overstatement, 65-70% undefined). Diagnostic only."
        ),
    )


_ASSUMED_SPREAD_BPS = {
    "SPY": 2.0,
    "QQQ": 2.0,
    "IVV": 3.0,
    "VOO": 3.0,
    "IWM": 5.0,
    "TLT": 5.0,
    "IEF": 5.0,
    "GLD": 5.0,
    "XLK": 8.0,
    "XLF": 8.0,
    "XLE": 8.0,
    "XLV": 8.0,
}


def assumed_spread(
    symbol: str,
    default_bps: float = 10.0,
) -> SpreadEstimate:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    default_bps = _finite_number("default_bps", default_bps)

    if default_bps < 0:
        raise ValueError("default_bps must be >= 0")

    bps = _ASSUMED_SPREAD_BPS.get(symbol.upper(), default_bps)

    return SpreadEstimate(
        bps / 1e4,
        SpreadSource.ASSUMED_CONSERVATIVE,
        True,
        None,
        None,
        (
            f"ASSUMED {bps}bps for {symbol}. NOT MEASURED. "
            "Replace with NBBO or issuer-published median spread."
        ),
    )


def measured_spread(
    bps: float,
    source: SpreadSource = SpreadSource.MEASURED_NBBO,
) -> SpreadEstimate:
    bps = _finite_number("bps", bps)

    if bps < 0:
        raise ValueError("spread bps must be >= 0")

    if source not in {
        SpreadSource.MEASURED_NBBO,
        SpreadSource.ISSUER_PUBLISHED,
    }:
        raise ValueError("measured_spread requires a measured source")

    return SpreadEstimate(
        bps / 1e4,
        source,
        True,
        None,
        None,
        "measured",
    )


@dataclass(frozen=True)
class CostBreakdown:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float
    t1_drag_bps: float
    total_bps: float
    spread_source: str


def round_trip_cost_bps(
    full_spread_frac: float,
    notional_buy: float,
    notional_sell: float,
    commission_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
    impact_bps: float = 0.0,
    t1_drag_bps: float = 0.0,
    spread_source: str = "unspecified",
) -> CostBreakdown:
    full_spread_frac = _finite_number(
        "full_spread_frac", full_spread_frac
    )
    notional_buy = _finite_number("notional_buy", notional_buy)
    notional_sell = _finite_number("notional_sell", notional_sell)
    commission_bps_per_side = _finite_number(
        "commission_bps_per_side", commission_bps_per_side
    )
    slippage_bps_per_side = _finite_number(
        "slippage_bps_per_side", slippage_bps_per_side
    )
    impact_bps = _finite_number("impact_bps", impact_bps)
    t1_drag_bps = _finite_number("t1_drag_bps", t1_drag_bps)

    if full_spread_frac < 0:
        raise ValueError("spread must be >= 0")

    if notional_buy <= 0 or notional_sell <= 0:
        raise ValueError("notionals must be positive")

    components = (
        commission_bps_per_side,
        slippage_bps_per_side,
        impact_bps,
        t1_drag_bps,
    )

    if min(components) < 0:
        raise ValueError("cost components must be >= 0")

    if not isinstance(spread_source, str) or not spread_source:
        raise ValueError("spread_source must be a non-empty string")

    half_spread = full_spread_frac / 2.0

    spread_usd = (
        notional_buy * half_spread
        + notional_sell * half_spread
    )

    average_notional = (notional_buy + notional_sell) / 2.0

    spread_bps = spread_usd / average_notional * 1e4
    commission = 2.0 * commission_bps_per_side
    slippage = 2.0 * slippage_bps_per_side

    total = (
        spread_bps
        + commission
        + slippage
        + impact_bps
        + t1_drag_bps
    )

    values = (spread_bps, commission, slippage, total)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("round-trip cost calculation produced non-finite value")

    return CostBreakdown(
        commission,
        spread_bps,
        slippage,
        impact_bps,
        t1_drag_bps,
        total,
        spread_source,
    )


def square_root_impact_bps(
    participation_rate: float,
    daily_vol: float,
    Y: float = 0.5,
    max_participation: float = 0.10,
) -> float:
    participation_rate = _finite_number(
        "participation_rate", participation_rate
    )
    daily_vol = _finite_number("daily_vol", daily_vol)
    Y = _finite_number("Y", Y)
    max_participation = _finite_number(
        "max_participation", max_participation
    )

    if not 0.0 <= max_participation <= 1.0:
        raise ValueError("max_participation must be in [0, 1]")

    if participation_rate < 0:
        raise ValueError("participation >= 0")

    if participation_rate > max_participation:
        raise ValueError("participation exceeds cap")

    if daily_vol <= 0 or Y <= 0:
        raise ValueError("daily_vol and Y must be positive")

    result = Y * daily_vol * math.sqrt(participation_rate) * 1e4

    if not math.isfinite(result):
        raise ValueError("impact calculation produced non-finite value")

    return result


def capacity_notional(
    alpha_bps: float,
    daily_vol: float,
    adv_notional: float,
    Y: float = 0.5,
    max_participation: float = 0.10,
) -> float:
    alpha_bps = _finite_number("alpha_bps", alpha_bps)
    daily_vol = _finite_number("daily_vol", daily_vol)
    adv_notional = _finite_number("adv_notional", adv_notional)
    Y = _finite_number("Y", Y)
    max_participation = _finite_number(
        "max_participation", max_participation
    )

    if not 0.0 <= max_participation <= 1.0:
        raise ValueError("max_participation must be in [0, 1]")

    if alpha_bps <= 0:
        return 0.0

    if daily_vol <= 0 or adv_notional <= 0 or Y <= 0:
        raise ValueError("daily_vol, adv_notional, Y must be positive")

    capacity = adv_notional * (
        (alpha_bps / 1e4) / (Y * daily_vol)
    ) ** 2

    result = float(min(capacity, max_participation * adv_notional))

    if not math.isfinite(result):
        raise ValueError("capacity calculation produced non-finite value")

    return result
