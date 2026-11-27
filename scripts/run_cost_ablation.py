#!/usr/bin/env python3
"""Batch A: paired cost-aware vs engine+band allocation ablation (MECHANICAL).

Research-only, allocation-level. Grants no authority, routes no orders, changes
no live default. See evidence/batchA_cost_aware_protocol.md (frozen). Both arms
use identical placeholder forecasts + realized LW covariance; any after-cost
difference is cost/timing mechanics, NOT alpha.

Run:
    python scripts/run_cost_ablation.py --days 252
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.allocate import solve_allocation
from quant_m1.allocate.cost_aware import cost_aware_allocator
from quant_m1.portfolio.covariance import ledoit_wolf_constant_correlation
from quant_m1.core import sha

RA, BAND, COST_BPS, MIN_TRADE_BPS, COV_W, MOM = 2.0, 0.25, 5, 5, 63, 21
COST = COST_BPS / 1e4
DUST = MIN_TRADE_BPS / 1e4


def _exec(w_prev, target):
    delta = np.asarray(target) - w_prev
    delta = np.where(np.abs(delta) < DUST, 0.0, delta)
    w = w_prev + delta
    guard = 0
    if w.sum() > 1.0 + 1e-9:
        w = w / w.sum(); guard = 1
    w = np.clip(w, 0.0, None)
    traded = float(np.abs(w - w_prev).sum())
    return w, traded, guard


def _run_arm(panel, start, alloc_fn):
    n = panel.shape[1]
    w = np.zeros(n); nav = 1.0; peak = 1.0; maxdd = 0.0
    cost_sum = 0.0; turn_sum = 0.0; gross_sum = 0.0
    rebal = 0; fails = 0; guards = 0; days = 0; max_gross = 0.0
    for i in range(start, len(panel) - 1):
        X = panel.iloc[i - COV_W:i].pct_change(fill_method=None).dropna().values
        cov, _ = ledoit_wolf_constant_correlation(X)
        mom = panel.iloc[i - MOM:i].pct_change(fill_method=None).mean()
        ranks = mom.rank(pct=True).values
        er = np.where(ranks > 0.8, 0.001, np.where(ranks < 0.2, -0.001, 0.0))
        res = alloc_fn(er, cov, w)
        if not getattr(res, "solver_ok", False):
            target = w; fails += 1
        else:
            target = np.asarray(res.weights, float)
        w_new, traded, guard = _exec(w, target)
        guards += guard
        if traded > 0: rebal += 1
        c = COST * traded
        r = panel.iloc[i + 1].values / panel.iloc[i].values - 1.0
        gross_ret = float(w_new @ r)
        net = gross_ret - c
        nav *= (1.0 + net); peak = max(peak, nav)
        maxdd = max(maxdd, (peak - nav) / peak)
        cost_sum += c; turn_sum += traded; g=float(w_new.sum()); gross_sum += g; max_gross=max(max_gross,g); days += 1
        denom = 1.0 + gross_ret
        w = w_new * (1.0 + r) / denom if denom > 0 else w_new
    ann = 252.0 / max(days, 1)
    return {"total_cost_bp": cost_sum * 1e4, "turnover_annual_x": turn_sum * ann,
            "mean_gross": gross_sum / max(days, 1), "rebalance_days": rebal,
            "solver_fail_days": fails, "guardrail_events": guards,
            "net_cum_return": nav - 1.0, "max_drawdown": maxdd, "days": days,
            "max_gross": max_gross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="evidence/panel_real_v1.csv")
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--out", default="evidence/cost_ablation_report.json")
    a = ap.parse_args()
    close = pd.read_csv(a.panel, index_col=0, parse_dates=True)
    close = close[sorted(close.columns)]
    if close.index.tzinfo is None:
        close.index = close.index.tz_localize("UTC")
    start = max(len(close) - a.days, COV_W + 1, MOM + 1)
    ctrl = _run_arm(close, start, lambda er, cov, w: solve_allocation(
        er, cov, risk_aversion=RA, current_weights=w, band_frac=BAND))
    treat = _run_arm(close, start, lambda er, cov, w: cost_aware_allocator(
        er, cov, risk_aversion=RA, current_weights=w, band_frac=BAND, cost_bps=COST_BPS))
    primary = treat["total_cost_bp"] <= ctrl["total_cost_bp"] + 1e-9
    secondary = treat["turnover_annual_x"] <= ctrl["turnover_annual_x"] + 1e-9
    # Protocol condition is gross <= 1 every day (the guard enforces it);
    # guardrail_events is a diagnostic count, not a failure.
    guardrails = (ctrl["max_gross"] <= 1.0 + 1e-9 and treat["max_gross"] <= 1.0 + 1e-9)
    accept = bool(primary and secondary and guardrails)
    report = {"protocol": "batchA_cost_aware_frozen",
              "params": {"risk_aversion": RA, "band_frac": BAND, "cost_bps": COST_BPS,
                         "min_trade_bps": MIN_TRADE_BPS, "cov_window": COV_W,
                         "momentum": MOM, "days_requested": a.days,
                         "start_index": start, "n_days": ctrl["days"]},
              "control": ctrl, "treatment": treat,
              "paired": {"net_diff_T_minus_C": treat["net_cum_return"] - ctrl["net_cum_return"],
                         "cost_saving_bp_C_minus_T": ctrl["total_cost_bp"] - treat["total_cost_bp"]},
              "acceptance": {"primary_cost_noninferior": primary,
                             "secondary_turnover_noninferior": secondary,
                             "guardrails": guardrails, "accept_switch_candidate": accept},
              "decision": "OWNER ONLY — switch runner default iff accept AND explicit approval",
              "disclosure": "non-PIT panel; placeholder forecasts; mechanical cost study, NOT alpha"}
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"days={ctrl['days']}")
    print(f"CONTROL   cost={ctrl['total_cost_bp']:.1f}bp turnover={ctrl['turnover_annual_x']:.2f}x "
          f"gross={ctrl['mean_gross']:.2f} net={ctrl['net_cum_return']:.2%} fails={ctrl['solver_fail_days']}")
    print(f"TREATMENT cost={treat['total_cost_bp']:.1f}bp turnover={treat['turnover_annual_x']:.2f}x "
          f"gross={treat['mean_gross']:.2f} net={treat['net_cum_return']:.2%} fails={treat['solver_fail_days']}")
    print(f"cost saving (C-T)={ctrl['total_cost_bp']-treat['total_cost_bp']:.1f}bp | "
          f"accept_switch_candidate={accept}")
    print("DECISION IS OWNER'S — no live default changed.")


if __name__ == "__main__":
    main()
