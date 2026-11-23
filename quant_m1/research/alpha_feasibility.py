"""ACTIVE-ALPHA feasibility gate. NOT a total-return profitability gate.

FLAM constrains active alpha / information ratio, not beta risk premia.
A long-only ETF portfolio can earn risk premia with IC ~ 0; this gate must
never be used to reject beta harvesting or drawdown control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy import special

TRADING_DAYS = 252


@dataclass(frozen=True)
class Verdict:
    admissible: bool
    reasons: Tuple[str, ...]
    diagnostics: dict


def effective_breadth(n_assets: int, avg_corr: float) -> float:
    """Conservative effective breadth. Capped at N; min of two bounds."""
    if n_assets < 1:
        raise ValueError("n_assets >= 1")
    lo = -1.0 / max(n_assets - 1, 1)
    if not (lo < avg_corr <= 1.0):
        raise ValueError("avg_corr outside PSD equicorrelation range")
    var_based = min(n_assets / (1.0 + (n_assets - 1) * avg_corr), float(n_assets))
    lam1 = 1.0 + (n_assets - 1) * avg_corr
    lam_rest = 1.0 - avg_corr
    sum_sq = lam1 ** 2 + (n_assets - 1) * lam_rest ** 2
    participation = (n_assets ** 2) / sum_sq if sum_sq > 0 else float(n_assets)
    return float(min(var_based, participation, n_assets))


def annual_effective_breadth(n_assets: int, avg_corr: float, hold_days: float,
                             signal_persistence_days: float) -> float:
    if hold_days <= 0 or signal_persistence_days <= 0:
        raise ValueError("durations must be positive")
    n_cross = effective_breadth(n_assets, avg_corr)
    rebal = TRADING_DAYS / hold_days
    independence = min(1.0, hold_days / signal_persistence_days)
    return n_cross * rebal * independence


def achievable_ir(ic: float, breadth: float, transfer_coeff: float) -> float:
    if breadth < 0:
        raise ValueError("breadth >= 0")
    if not (0.0 <= transfer_coeff <= 1.0):
        raise ValueError("transfer_coeff in [0,1]")
    return ic * math.sqrt(breadth) * transfer_coeff


def optimal_holding_days(alpha_max: float, decay_days: float,
                         roundtrip_cost_frac: float) -> Optional[float]:
    """alpha(h)=alpha_max*(1-exp(-h/H)).
    FOC: (1+h/H)*exp(-h/H) = 1 - c/alpha_max
    h* = H*(-W_{-1}(-(1-beta)/e) - 1).  None => cost dominates, never trade.
    """
    if alpha_max <= 0 or decay_days <= 0:
        raise ValueError("alpha_max and decay_days must be positive")
    if roundtrip_cost_frac < 0:
        raise ValueError("cost >= 0")
    beta = roundtrip_cost_frac / alpha_max
    if beta >= 1.0:
        return None
    if beta == 0.0:
        return float("inf")
    z = -(1.0 - beta) / math.e
    x = -special.lambertw(z, k=-1).real - 1.0
    return float(decay_days * x)


def net_alpha_rate_annual(alpha_max: float, decay_days: float,
                          roundtrip_cost_frac: float, hold_days: float) -> float:
    if hold_days <= 0:
        raise ValueError("hold_days > 0")
    a = alpha_max * (1.0 - math.exp(-hold_days / decay_days))
    return (a - roundtrip_cost_frac) / hold_days * TRADING_DAYS


def alpha_feasibility_gate(*, n_assets: int, avg_corr: float, ic: float,
                           transfer_coeff: float, signal_decay_days: float,
                           signal_persistence_days: float,
                           alpha_max_per_trade: float, roundtrip_cost_bps: float,
                           target_ir: float = 0.5) -> Verdict:
    c = roundtrip_cost_bps / 1e4
    reasons, diag = [], {}
    h = optimal_holding_days(alpha_max_per_trade, signal_decay_days, c)
    if h is None:
        reasons.append("COST_DOMINATES: alpha_max <= roundtrip cost.")
        diag["optimal_hold_days"] = None
    else:
        h_eff = max(h, 1.0)
        br = annual_effective_breadth(n_assets, avg_corr, h_eff, signal_persistence_days)
        ir = achievable_ir(ic, br, transfer_coeff)
        net = net_alpha_rate_annual(alpha_max_per_trade, signal_decay_days, c, h_eff)
        diag.update(optimal_hold_days=h, hold_used=h_eff, breadth=br,
                    achievable_ir=ir, net_alpha_rate_annual=net)
        if ir < target_ir:
            reasons.append(f"BREADTH_STARVED: active IR {ir:.3f} < target {target_ir}.")
        if net <= 0:
            reasons.append(f"NEGATIVE_NET_ALPHA at feasible hold: {net*100:.2f}%/yr.")
    return Verdict(len(reasons) == 0, tuple(reasons), diag)
