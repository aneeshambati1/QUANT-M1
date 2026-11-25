"""Deterministic cost and turnover attribution. ACCOUNTING, NOT INFERENCE.

TWO CLASSES OF OUTPUT, DELIBERATELY SEPARATED:

  MECHANICAL - turnover and cost paid. Given the realised weight path and a
    cost assumption these are arithmetic identities. No sampling uncertainty.
    Uncertainty lives ONLY in the cost assumption, reported as sensitivity
    across pre-declared scenarios.

  PATH-DEPENDENT - return, volatility, Sharpe, drawdown, rebalancing premium.
    These are measurements of ONE realised history. They DO carry sampling
    uncertainty which is NOT quantified here. No significance is claimed, no
    p-value is computed, and no generalisation to other periods is asserted.

Sharpe uses arithmetic mean annualised over annualised vol, risk-free = 0.
The zero risk-free assumption inflates Sharpe; it is a convention, not a claim.

Policies are pre-declared and ALL are reported. None is selected as a winner.
The chase proxy is a BEHAVIOURAL cost illustration, not a candidate strategy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

POLICIES = ("buy_hold", "monthly_21d", "band_rel25", "daily", "chase_top3_wk")
COST_SCENARIOS_BP = (1.0, 3.0, 10.0)   # ONE-WAY bp on traded notional
BAND_REL = 0.25
CHASE_N = 3
CHASE_EVERY = 5
TRAIL = 21


class AccountingError(Exception):
    pass


def equal_target(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def drift(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights after one period of asset returns, before any rebalance."""
    grown = w * (1.0 + r)
    tot = grown.sum()
    if tot <= 0:
        raise AccountingError("portfolio value non-positive")
    return grown / tot


def traded_fraction(w_before: np.ndarray, w_after: np.ndarray) -> float:
    """Total traded notional / NAV, BOTH sides summed.

    Selling 10% of A and buying 10% of B gives 0.20. Cost is this times the
    one-way bp, because both legs are executed.
    """
    return float(np.abs(w_after - w_before).sum())


def _trailing(returns: pd.DataFrame) -> pd.DataFrame:
    """Trailing TRAIL-period return, shifted so row t uses data through t-1."""
    cum = (1.0 + returns).rolling(TRAIL).apply(np.prod, raw=True) - 1.0
    return cum.shift(1)


def run_policy(returns: pd.DataFrame, policy: str) -> dict:
    if policy not in POLICIES:
        raise AccountingError(f"unknown policy {policy}")
    R = returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(R)):
        raise AccountingError("returns must be finite")
    T, N = R.shape
    if T < 60 or N < 2:
        raise AccountingError("need T>=60 and N>=2")
    target = equal_target(N)
    trail = _trailing(returns).to_numpy(dtype=float) if policy == "chase_top3_wk" else None

    w = target.copy()
    gross = np.zeros(T)
    traded = np.zeros(T)
    weights = np.zeros((T, N))

    for t in range(T):
        if t > 0:
            if policy == "monthly_21d":
                do = (t % 21 == 0)
                new = target if do else w
            elif policy == "daily":
                new = target
            elif policy == "band_rel25":
                rel = np.abs(w / target - 1.0)
                new = target if rel.max() > BAND_REL else w
            elif policy == "chase_top3_wk":
                if t % CHASE_EVERY == 0 and np.all(np.isfinite(trail[t])):
                    idx = np.argsort(-trail[t])[:CHASE_N]
                    new = np.zeros(N)
                    new[idx] = 1.0 / CHASE_N
                else:
                    new = w
            else:                       # buy_hold
                new = w
            traded[t] = traded_fraction(w, new)
            w = new
        weights[t] = w
        gross[t] = float(w @ R[t])
        w = drift(w, R[t])

    return {"policy": policy, "gross": gross, "traded": traded,
            "weights": weights, "n_assets": N, "T": T}


def _max_dd(wealth: np.ndarray) -> float:
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peak))


def summarize(res: dict, cost_bp: float) -> dict:
    if cost_bp < 0:
        raise AccountingError("cost_bp must be >= 0")
    g, tr, T = res["gross"], res["traded"], res["T"]
    years = T / 252.0
    cost = tr * (cost_bp / 10_000.0)
    net = g - cost
    wg = np.cumprod(1.0 + g)
    wn = np.cumprod(1.0 + net)
    vol = float(net.std(ddof=1) * math.sqrt(252))
    ann_arith = float(net.mean() * 252)
    return {
        "policy": res["policy"],
        "cost_bp_oneway": cost_bp,
        # MECHANICAL
        "annual_traded_notional": float(tr.sum() / years),
        "annual_cost_bp": float(tr.sum() / years * cost_bp),
        "total_traded_notional": float(tr.sum()),
        # PATH-DEPENDENT, DESCRIPTIVE ONLY
        "gross_ann_return": float(wg[-1] ** (1.0 / years) - 1.0),
        "net_ann_return": float(wn[-1] ** (1.0 / years) - 1.0),
        "net_ann_vol": vol,
        "net_sharpe_descriptive": (ann_arith / vol) if vol > 0 else 0.0,
        "net_max_dd_descriptive": _max_dd(wn),
        "years": years,
    }


def attribute(returns: pd.DataFrame) -> dict:
    rows = []
    for p in POLICIES:
        res = run_policy(returns, p)
        for c in COST_SCENARIOS_BP:
            rows.append(summarize(res, c))
    return {
        "policies": list(POLICIES),
        "cost_scenarios_bp_oneway": list(COST_SCENARIOS_BP),
        "rows": rows,
        "mechanical_fields": ["annual_traded_notional", "annual_cost_bp",
                              "total_traded_notional"],
        "descriptive_fields": ["gross_ann_return", "net_ann_return", "net_ann_vol",
                               "net_sharpe_descriptive", "net_max_dd_descriptive"],
        "label": ("ACCOUNTING ON ONE REALISED HISTORY - turnover and cost are "
                  "mechanical; return/vol/Sharpe/DD are descriptive with "
                  "UNQUANTIFIED sampling uncertainty - NO significance claim"),
        "risk_free_assumed": 0.0,
    }