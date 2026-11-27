#!/usr/bin/env python3
"""Honest exploratory simulation runner (Phase 1, Batch 2).

Plumbing-only repair. NO signal, authority-rule, or Risk-Kernel change:
  - Capital default is $100,000 = 10_000_000 cents (the amount the experiment
    always intended; the previous 10_000_00 default was $10,000).
  - Universe is constructed at build time from the real panel marks, so the
    mandate allowlist, PLAIN_ETF product class, and stress scenarios cover
    every ETF before any decision. holdings=None keeps the shared UNK:ALL
    issuer bucket and the 40% issuer cap (preserved by design).
  - Trackers are keyed by MODEL ID, not ETF symbol, and forecasts are passed
    as model_signals through the intended blend path (not a pre-blended
    expected_returns vector). kappa, probation_obs, half_life_obs are UNCHANGED.
  - Covariance is estimated from the SAME trailing return window that drives
    MarketState (Ledoit-Wolf constant-correlation, already in the repository).
    These are realized sample estimates, NOT validated market forecasts.
  - Actual drawdown is computed from current marks and the mandate's own
    drawdown_trigger_bps is used (no hardcoded 5000 bp / zero drawdown).
  - current_weights and a documented no-trade band (band_frac=0.25, per the
    RESULTS.md mechanical cost finding) are passed to the allocator.
  - Transaction costs are charged at execution and REPORTED (costs, turnover,
    drawdown, cash fraction, net return) against cash / SPY / equal-weight
    baselines. Costs are not yet inside the optimizer objective (Phase 2+).

AUTHORITY / BOOTSTRAP (intentional, do not "fix" by forcing trades):
  run_paper_step is called WITHOUT an attribution callback. The Phase 2
  probe/shadow bootstrap protocol is not yet approved, so trackers receive no
  observations, remain in PROBATION at n=0, and earn no authority. The runner
  therefore reports NO AUTHORITY / NO TRADES. It must NOT manufacture
  observations or eligibility. An all-cash run is the correct fail-closed
  outcome, not a performance result.

Placeholders carried from Batch 1 (documented, not measured):
  - ADV = $10M and unit beta for every ETF. With a $20k order-notional cap,
    5% participation of $10M ($500k) does not bind; participation is effectively
    inactive under these placeholders.
  - liquidity_score=1.0 and data_age_secs=0 are replay assumptions (the panel
    has no liquidity field); model_disagreement=0.0 (single model).
  - The +/-0.001 daily expected-return magnitude and 0.8/0.2 rank thresholds
    are the ORIGINAL uncalibrated research placeholders, kept unchanged.

The panel is back-adjusted and non-Point-In-Time. No alpha or profitability
claim is made or supported by this runner.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_m1.system import build
from quant_m1.universe import etf_instruments
from quant_m1.meta.state import MarketState
from quant_m1.meta.contribution import ContributionTracker
from quant_m1.allocate import solve_allocation
from quant_m1.orchestrator.paper_loop import run_paper_step
from quant_m1.portfolio.covariance import ledoit_wolf_constant_correlation

MODEL_ID = "mom_21d_h1"
AUTHORITY_NOTE = "No model has earned positive authority"
TRADING_DAYS = 252


def load_panel(path: str) -> pd.DataFrame:
    close = pd.read_csv(path, index_col=0, parse_dates=True)
    symbols = sorted(close.columns.tolist())
    close = close[symbols]
    if close.index.tzinfo is None:
        close.index = close.index.tz_localize("UTC")
    return close


def _nav_cents(ledger, prices: dict) -> int:
    st = ledger.state
    return (st.cash_settled + st.receivable_total()
            + sum(q * prices[s] for s, q in st.positions.items() if q))


def prospective_drawdown_bps(ledger, prices: dict) -> int:
    """Drawdown as of the incoming marks, matching ledger.drawdown_bps()
    after advance_session + mark_peak (peak includes the current NAV)."""
    nav = _nav_cents(ledger, prices)
    peak = max(ledger.state.peak_nav, nav)
    if peak <= 0:
        return 0
    return (peak - nav) * 10_000 // peak


def current_weights(ledger, prices: dict, symbols: list) -> np.ndarray:
    nav = _nav_cents(ledger, prices)
    if nav <= 0:
        return np.zeros(len(symbols))
    w = np.array([ledger.state.positions.get(s, 0) * prices[s] / nav for s in symbols])
    w = np.clip(w, 0.0, None)
    if w.sum() > 1.0:
        w = w / w.sum()
    return w


def run_simulation(panel: pd.DataFrame, *, days: int = 60, cash: int = 10_000_000,
                   cov_window: int = 63, cost_bps: int = 5, min_trade_bps: int = 5,
                   band_frac: float = 0.25, risk_aversion: float = 2.0,
                   workdir: str | None = None) -> dict:
    if isinstance(cash, bool) or not isinstance(cash, int) or cash <= 0:
        raise ValueError("cash must be a positive integer (cents)")
    symbols = sorted(panel.columns.tolist())
    n = len(symbols)
    if cov_window < 3:
        raise ValueError("cov_window must be >= 3")
    # Warmup: covariance needs cov_window closes (cov_window-1 returns); momentum needs 21.
    min_history = max(cov_window + 1, 22)
    if len(panel) < min_history + 1:
        raise ValueError(f"panel too short: need >= {min_history + 1} rows, got {len(panel)}")
    start_idx = max(len(panel) - days, min_history)

    first_row = panel.iloc[start_idx]
    if (first_row <= 0).any() or first_row.isna().any():
        raise ValueError("first simulation day has non-positive/missing prices")
    first_prices = {s: int(round(first_row[s] * 100)) for s in symbols}

    # Fresh unique workdir: reusing a directory would let Gateway.restore_flags()
    # reload a stale composite ledger from a prior run.
    workdir = workdir or tempfile.mkdtemp(prefix="quant_sim_")
    os.makedirs(workdir, exist_ok=True)
    sx = build(workdir, cash=cash, positions={}, marks=first_prices,
               instruments=etf_instruments(symbols))
    cfg, ledger = sx["cfg"], sx["ledger"]
    if ledger.state.nav() != cash:
        raise ValueError(f"starting NAV {ledger.state.nav()} != intended cash {cash}")

    # Model-ID tracker; authority parameters UNCHANGED. No synthetic eligibility.
    trackers = {MODEL_ID: ContributionTracker(half_life_obs=60.0, kappa=1.0,
                                              probation_obs=30, horizon_days=1)}

    rows = []
    skipped = 0
    cum_cost = 0
    cum_notional = 0
    authority_blocks = 0
    intents_total = 0

    for i in range(start_idx, len(panel)):
        date = panel.index[i]
        row = panel.iloc[i]
        if (row <= 0).any() or row.isna().any():
            skipped += 1
            continue
        prices = {s: int(round(row[s] * 100)) for s in symbols}

        # One trailing window drives BOTH covariance and MarketState (no lookahead:
        # window ends at i-1). Realized sample estimates, not validated forecasts.
        X = panel.iloc[i - cov_window:i].pct_change().dropna().values
        Sigma, shrink = ledoit_wolf_constant_correlation(X)
        corr = np.corrcoef(X, rowvar=False)
        avg_corr = float(corr[~np.eye(n, dtype=bool)].mean())
        vol_ann = float(X.mean(axis=1).std(ddof=1) * math.sqrt(TRADING_DAYS))

        # Original uncalibrated momentum placeholder, unchanged (plumbing-only batch).
        mom = panel.iloc[i - 21:i].pct_change().mean()
        ranks = mom.rank(pct=True).values
        er = np.where(ranks > 0.8, 0.001, np.where(ranks < 0.2, -0.001, 0.0))

        dd = prospective_drawdown_bps(ledger, prices)
        ms = MarketState(
            as_of=date.isoformat(), avg_corr=avg_corr, vol_ann=vol_ann,
            liquidity_score=1.0, data_age_secs=0, model_disagreement=0.0,
            n_assets=n, drawdown_bps=dd,
            drawdown_trigger_bps=cfg.mandate.drawdown_trigger_bps,
        )
        fc = {
            "model_signals": {MODEL_ID: er},
            "cov": Sigma,
            "risk_aversion": risk_aversion,
            "horizon_days": 1,
            "current_weights": current_weights(ledger, prices, symbols),
            "band_frac": band_frac,
        }
        rep = run_paper_step(
            sx, ms, trackers, fc,
            new_trade_date=ledger.state.trade_date + 1,
            new_mark_wall=int(date.timestamp()) + 1,
            new_marks=prices,
            allocator=solve_allocation,
            min_trade_bps=min_trade_bps, cost_bps=cost_bps,
        )

        nav = ledger.state.nav()
        cash_equiv = ledger.state.cash_settled + ledger.state.receivable_total()
        gross_val = sum(q * prices[s] for s, q in ledger.state.positions.items() if q)
        new_cost = sum(o.filled_cost for o in ledger.state.working)
        new_notional = sum(o.filled_notional for o in ledger.state.working)
        day_cost = new_cost - cum_cost
        day_notional = new_notional - cum_notional
        cum_cost, cum_notional = new_cost, new_notional
        intents_total += rep.intents_submitted
        if AUTHORITY_NOTE in rep.note:
            authority_blocks += 1

        rows.append({
            "date": date, "nav_cents": nav,
            "cash_frac": cash_equiv / nav if nav else 1.0,
            "gross_frac": gross_val / nav if nav else 0.0,
            "drawdown_bps": dd,
            "drawdown_trigger_bps": cfg.mandate.drawdown_trigger_bps,
            "avg_corr": avg_corr, "vol_ann": vol_ann, "shrinkage": float(shrink),
            "intents_submitted": rep.intents_submitted,
            "intents_rejected": rep.intents_rejected,
            "filled_orders": rep.filled_orders,
            "cost_cents": day_cost, "executed_notional_cents": day_notional,
            "blocked": rep.blocked, "note": rep.note,
        })

    daily = pd.DataFrame(rows).set_index("date")
    if daily.empty:
        raise ValueError("no valid simulation days (all rows skipped)")
    ending_nav = int(daily["nav_cents"].iloc[-1])

    # Descriptive baselines over the identical window (entry at first decision-day
    # close, hold to last close). Net subtracts ONE entry cost; no exit/rebalance cost.
    first_close = panel.iloc[start_idx]
    last_close = panel.iloc[-1]
    gross_bh = (last_close / first_close - 1.0)
    bench = "SPY" if "SPY" in symbols else symbols[0]
    entry_cost = cost_bps / 10_000.0
    baselines = {
        "cash": 0.0,
        "benchmark_symbol": bench,
        "spy_buy_hold_gross": float(gross_bh[bench]),
        "spy_buy_hold_net": float(gross_bh[bench]) - entry_cost,
        "equal_weight_gross": float(gross_bh.mean()),
        "equal_weight_net": float(gross_bh.mean()) - entry_cost,
    }

    summary = {
        "period": (str(daily.index[0].date()), str(daily.index[-1].date())),
        "days": len(daily), "skipped_days": skipped,
        "starting_nav_cents": cash, "ending_nav_cents": ending_nav,
        "net_return": ending_nav / cash - 1.0,
        "total_cost_cents": cum_cost,
        "total_executed_notional_cents": cum_notional,
        "total_turnover": cum_notional / cash,
        "max_drawdown_bps": int(daily["drawdown_bps"].max()),
        "mean_cash_frac": float(daily["cash_frac"].mean()),
        "mean_gross_frac": float(daily["gross_frac"].mean()),
        "authority_blocks": authority_blocks,
        "intents_submitted_total": intents_total,
        "no_authority": intents_total == 0,
        "baselines": baselines,
        "registered_symbols": symbols,
        "mandate_version": cfg.mandate.version,
        "config_digest": cfg.mandate.digest(),
        "model_id": MODEL_ID,
        "tracker_snapshot": trackers[MODEL_ID].snapshot(),
        "workdir": workdir,
    }
    daily.to_csv(os.path.join(workdir, "daily.csv"))
    return {"daily": daily, "summary": summary, "system": sx}


def format_report(result: dict) -> str:
    s = result["summary"]
    b = s["baselines"]
    lines = []
    lines.append("=" * 64)
    lines.append("QUANT-M1 EXPLORATORY SIMULATION (research only; non-PIT panel)")
    lines.append("=" * 64)
    lines.append(f"  Period            : {s['period'][0]} -> {s['period'][1]}  ({s['days']} days)")
    lines.append(f"  Starting NAV      : ${s['starting_nav_cents']/100:,.2f}")
    lines.append(f"  Ending NAV        : ${s['ending_nav_cents']/100:,.2f}")
    lines.append(f"  Net return        : {s['net_return']:.2%}")
    lines.append(f"  Total costs       : ${s['total_cost_cents']/100:,.2f}")
    lines.append(f"  Total turnover    : {s['total_turnover']:.3f}x")
    lines.append(f"  Max drawdown      : {s['max_drawdown_bps']} bp")
    lines.append(f"  Mean cash / gross : {s['mean_cash_frac']:.2%} / {s['mean_gross_frac']:.2%}")
    lines.append(f"  Intents submitted : {s['intents_submitted_total']}")
    lines.append(f"  Authority blocks  : {s['authority_blocks']}/{s['days']}")
    lines.append("-" * 64)
    lines.append("  Descriptive baselines (same window; net = gross - one entry cost):")
    lines.append(f"    Cash                 : {b['cash']:.2%}")
    lines.append(f"    {b['benchmark_symbol']} buy&hold (gross/net): {b['spy_buy_hold_gross']:.2%} / {b['spy_buy_hold_net']:.2%}")
    lines.append(f"    Equal-weight (gross/net): {b['equal_weight_gross']:.2%} / {b['equal_weight_net']:.2%}")
    lines.append("-" * 64)
    if s["no_authority"]:
        lines.append("  RESULT: NO AUTHORITY / NO TRADES")
        lines.append("  No model earned positive authority. Trackers stayed in PROBATION")
        lines.append(f"  (model '{s['model_id']}': n={s['tracker_snapshot']['n']}, "
                     f"status={s['tracker_snapshot']['status']}).")
        lines.append("  No attribution callback is wired: the Phase 2 probe/bootstrap")
        lines.append("  protocol is not approved. This all-cash outcome is the intended")
        lines.append("  fail-closed behavior. It is NOT evidence of edge or of no edge.")
    else:
        lines.append("  TRADES OCCURRED: verify they came from earned authority, not")
        lines.append("  synthetic eligibility, before interpreting any return.")
    lines.append("-" * 64)
    lines.append("  Assumptions / placeholders (not validated forecasts):")
    lines.append("    - Covariance & MarketState: realized Ledoit-Wolf constant-correlation")
    lines.append("      estimates from one trailing window; not calibrated forecasts.")
    lines.append("    - ADV=$10M and unit beta per ETF (placeholders); participation")
    lines.append("      constraint is non-binding under them given the $20k order cap.")
    lines.append("    - liquidity_score=1.0, data_age_secs=0 (replay assumptions);")
    lines.append("      model_disagreement=0.0 (single model).")
    lines.append("    - +/-0.001 daily ER and 0.8/0.2 rank cuts: original uncalibrated")
    lines.append("      research placeholders, unchanged. kappa/probation unchanged.")
    lines.append("    - UNK:ALL shared issuer bucket and 40% issuer cap preserved.")
    lines.append("    - Panel is back-adjusted, non-PIT. No alpha/profitability claim.")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="evidence/panel_real_v1.csv")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--cash", type=int, default=10_000_000)  # $100,000 in cents
    parser.add_argument("--cov-window", type=int, default=63)
    parser.add_argument("--cost-bps", type=int, default=5)
    parser.add_argument("--min-trade-bps", type=int, default=5)
    parser.add_argument("--band-frac", type=float, default=0.25)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    print(f"Panel: {len(panel)} dates x {len(panel.columns)} ETFs "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    result = run_simulation(panel, days=args.days, cash=args.cash,
                            cov_window=args.cov_window, cost_bps=args.cost_bps,
                            min_trade_bps=args.min_trade_bps, band_frac=args.band_frac)
    print(format_report(result))
    print(f"Daily ledger: {os.path.join(result['summary']['workdir'], 'daily.csv')}")


if __name__ == "__main__":
    main()
