#!/usr/bin/env python3
"""Exploratory paper simulation. No alpha claim; Phase 2 SE remains disabled."""
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
from quant_m1.allocate.cost_aware import cost_aware_allocator
from quant_m1.orchestrator.paper_loop import run_paper_step
from quant_m1.portfolio.covariance import ledoit_wolf_constant_correlation
from quant_m1.research.survivor_factors import (
    load_survivor_factors, factor_integration_report,
)
from quant_m1.meta.selector import select_weights_with_survivors

MODEL_ID = "mom_21d_h1"
AUTHORITY_NOTE = "No model has earned positive authority"
TRADING_DAYS = 252


def load_panel(path: str) -> pd.DataFrame:
    close = pd.read_csv(path, index_col=0, parse_dates=True)
    close = close[sorted(close.columns.tolist())]
    if close.index.tzinfo is None:
        close.index = close.index.tz_localize("UTC")
    if close.index.has_duplicates:
        raise ValueError("panel contains duplicate timestamps")
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    return close


def _nav_cents(ledger, prices: dict) -> int:
    state = ledger.state
    return (
        state.cash_settled + state.receivable_total()
        + sum(q * prices[s] for s, q in state.positions.items() if q)
    )


def prospective_drawdown_bps(ledger, prices: dict) -> int:
    nav = _nav_cents(ledger, prices)
    peak = max(ledger.state.peak_nav, nav)
    return 0 if peak <= 0 else (peak - nav) * 10_000 // peak


def current_weights(ledger, prices: dict, symbols: list) -> np.ndarray:
    nav = _nav_cents(ledger, prices)
    if nav <= 0:
        return np.zeros(len(symbols))
    weights = np.array([
        ledger.state.positions.get(symbol, 0) * prices[symbol] / nav
        for symbol in symbols
    ])
    weights = np.clip(weights, 0.0, None)
    if weights.sum() > 1.0:
        weights /= weights.sum()
    return weights


def _load_survivors_safe(path: str | None):
    """Load survivor factors if available; return None otherwise."""
    if path is None:
        return None
    try:
        return load_survivor_factors(path)
    except (FileNotFoundError, Exception):
        return None


def run_simulation(
    panel: pd.DataFrame,
    *,
    days: int = 60,
    cash: int = 10_000_000,
    cov_window: int = 63,
    cost_bps: int = 5,
    min_trade_bps: int = 5,
    band_frac: float = 0.25,
    risk_aversion: float = 2.0,
    workdir: str | None = None,
    survivor_path: str | None = "evidence/survivor_factors.csv",
    blend_alpha: float = 0.5,
) -> dict:
    if isinstance(cash, bool) or not isinstance(cash, int) or cash <= 0:
        raise ValueError("cash must be positive integer cents")
    if days < 1 or cov_window < 3:
        raise ValueError("days must be >=1 and cov_window >=3")
    if not 0.0 <= band_frac <= 1.0:
        raise ValueError("band_frac must be in [0, 1]")
    if cost_bps < 0 or min_trade_bps < 0:
        raise ValueError("basis-point parameters must be nonnegative")

    symbols = sorted(panel.columns.tolist())
    if not symbols:
        raise ValueError("panel has no symbols")
    n_assets = len(symbols)
    min_history = max(cov_window + 1, 22)
    if len(panel) < min_history + 1:
        raise ValueError(f"panel too short: need >= {min_history + 1} rows")

    start_idx = max(len(panel) - days, min_history)
    first_row = panel.iloc[start_idx]
    if first_row.isna().any() or (first_row <= 0).any():
        raise ValueError("invalid first simulation-day prices")
    first_prices = {s: int(round(first_row[s] * 100)) for s in symbols}

    workdir = workdir or tempfile.mkdtemp(prefix="quant_sim_")
    os.makedirs(workdir, exist_ok=True)
    sx = build(
        workdir,
        cash=cash,
        positions={},
        marks=first_prices,
        instruments=etf_instruments(symbols),
    )
    cfg, ledger = sx["cfg"], sx["ledger"]
    if ledger.state.nav() != cash:
        raise ValueError("starting NAV differs from requested cash")

    trackers = {
        MODEL_ID: ContributionTracker(
            half_life_obs=60.0,
            kappa=1.0,
            probation_obs=30,
            horizon_days=1,
        )
    }

    # Load survivor factors once (monthly data, aligned by date)
    survivors = _load_survivors_safe(survivor_path)
    survivor_active = survivors is not None and len(survivors) > 0

    rows = []
    skipped = 0
    cumulative_cost = 0
    cumulative_notional = 0
    authority_blocks = 0
    intents_total = 0

    for i in range(start_idx, len(panel)):
        date = panel.index[i]
        row = panel.iloc[i]
        if row.isna().any() or (row <= 0).any():
            skipped += 1
            continue
        prices = {s: int(round(row[s] * 100)) for s in symbols}

        # Both estimates use information ending at i-1.
        returns = panel.iloc[i - cov_window:i].pct_change(fill_method=None).dropna().values
        covariance, shrinkage = ledoit_wolf_constant_correlation(returns)
        correlation = np.corrcoef(returns, rowvar=False)
        if n_assets > 1:
            avg_corr = float(
                correlation[~np.eye(n_assets, dtype=bool)].mean()
            )
        else:
            avg_corr = 0.0
        vol_ann = float(
            returns.mean(axis=1).std(ddof=1) * math.sqrt(TRADING_DAYS)
        )

        # Frozen, uncalibrated momentum placeholder; not a validated forecast.
        momentum = panel.iloc[i - 21:i].pct_change(fill_method=None).mean()
        ranks = momentum.rank(pct=True).values
        expected_return = np.where(
            ranks > 0.8, 0.001,
            np.where(ranks < 0.2, -0.001, 0.0),
        )

        drawdown = prospective_drawdown_bps(ledger, prices)
        market_state = MarketState(
            as_of=date.isoformat(),
            avg_corr=avg_corr,
            vol_ann=vol_ann,
            liquidity_score=1.0,
            data_age_secs=0,
            model_disagreement=0.0,
            n_assets=n_assets,
            drawdown_bps=drawdown,
            drawdown_trigger_bps=cfg.mandate.drawdown_trigger_bps,
        )

        # Survivor-aware regime weighting (falls back to pre-reg if no data)
        survivor_rep = None
        if survivor_active:
            # Use only survivor data available BEFORE this simulation date
            avail = survivors[survivors.index < date]
            if len(avail) >= 12:
                survivor_rep = factor_integration_report(
                    avail, regime=market_state.regime.value
                )

        sel_weights = select_weights_with_survivors(
            market_state, survivor_rep, blend_alpha=blend_alpha
        )

        forecast = {
            "model_signals": {MODEL_ID: expected_return},
            "cov": covariance,
            "risk_aversion": risk_aversion,
            "horizon_days": 1,
            "current_weights": current_weights(ledger, prices, symbols),
            "band_frac": band_frac,
            "selector_weights": sel_weights,
        }

        report = run_paper_step(
            sx,
            market_state,
            trackers,
            forecast,
            new_trade_date=ledger.state.trade_date + 1,
            new_mark_wall=int(date.timestamp()) + 1,
            new_marks=prices,
            allocator=cost_aware_allocator,
            min_trade_bps=min_trade_bps,
            cost_bps=cost_bps,
        )

        nav = ledger.state.nav()
        cash_value = ledger.state.cash_settled + ledger.state.receivable_total()
        gross_value = sum(
            q * prices[s] for s, q in ledger.state.positions.items() if q
        )
        filled_cost = sum(order.filled_cost for order in ledger.state.working)
        filled_notional = sum(
            order.filled_notional for order in ledger.state.working
        )
        day_cost = filled_cost - cumulative_cost
        day_notional = filled_notional - cumulative_notional
        cumulative_cost, cumulative_notional = filled_cost, filled_notional

        intents_total += report.intents_submitted
        if AUTHORITY_NOTE in report.note:
            authority_blocks += 1

        rows.append({
            "date": date,
            "nav_cents": nav,
            "cash_frac": cash_value / nav if nav else 1.0,
            "gross_frac": gross_value / nav if nav else 0.0,
            "drawdown_bps": drawdown,
            "drawdown_trigger_bps": cfg.mandate.drawdown_trigger_bps,
            "avg_corr": avg_corr,
            "vol_ann": vol_ann,
            "shrinkage": float(shrinkage),
            "intents_submitted": report.intents_submitted,
            "intents_rejected": report.intents_rejected,
            "filled_orders": report.filled_orders,
            "cost_cents": day_cost,
            "executed_notional_cents": day_notional,
            "blocked": report.blocked,
            "note": report.note,
            "survivor_active": survivor_rep is not None,
            "selector_weights": sel_weights,
        })

    daily = pd.DataFrame(rows).set_index("date")
    if daily.empty:
        raise ValueError("no valid simulation days")
    ending_nav = int(daily["nav_cents"].iloc[-1])

    first_close = panel.iloc[start_idx]
    last_close = panel.iloc[-1]
    buy_hold = last_close / first_close - 1.0
    benchmark = "SPY" if "SPY" in symbols else symbols[0]
    entry_cost = cost_bps / 10_000.0

    benchmark_gross = float(buy_hold[benchmark])
    benchmark_net = benchmark_gross - entry_cost

    baselines = {
        "cash": 0.0,
        "benchmark_symbol": benchmark,
        "benchmark_buy_hold_gross": benchmark_gross,
        "benchmark_buy_hold_net": benchmark_net,
        "equal_weight_gross": float(buy_hold.mean()),
        "equal_weight_net": float(buy_hold.mean()) - entry_cost,
    }

    if "SPY" in symbols:
        spy_gross = float(buy_hold["SPY"])
        baselines["spy_buy_hold_gross"] = spy_gross
        baselines["spy_buy_hold_net"] = spy_gross - entry_cost

    no_authority = not any(t.eligible for t in trackers.values())
    no_trades = intents_total == 0

    summary = {
        "period": (str(daily.index[0].date()), str(daily.index[-1].date())),
        "days": len(daily),
        "skipped_days": skipped,
        "starting_nav_cents": cash,
        "ending_nav_cents": ending_nav,
        "net_return": ending_nav / cash - 1.0,
        "total_cost_cents": cumulative_cost,
        "total_executed_notional_cents": cumulative_notional,
        "total_turnover": cumulative_notional / cash,
        "max_drawdown_bps": int(daily["drawdown_bps"].max()),
        "mean_cash_frac": float(daily["cash_frac"].mean()),
        "mean_gross_frac": float(daily["gross_frac"].mean()),
        "authority_blocks": authority_blocks,
        "intents_submitted_total": intents_total,
        "no_authority": no_authority,
        "no_trades": no_trades,
        "se_enabled": False,
        "baselines": baselines,
        "registered_symbols": symbols,
        "mandate_version": cfg.mandate.version,
        "config_digest": cfg.mandate.digest(),
        "model_id": MODEL_ID,
        "tracker_snapshot": trackers[MODEL_ID].snapshot(),
        "workdir": workdir,
        "allocator": "cost_aware_allocator",
        "survivor_wiring": survivor_active,
        "blend_alpha": blend_alpha,
    }
    daily.to_csv(os.path.join(workdir, "daily.csv"))
    return {"daily": daily, "summary": summary, "system": sx}


def format_report(result: dict) -> str:
    s = result["summary"]
    b = s["baselines"]
    lines = [
        "=" * 64,
        "QUANT-M1 EXPLORATORY SIMULATION — RESEARCH ONLY",
        f"Period: {s['period'][0]} -> {s['period'][1]} ({s['days']} days)",
        f"Starting NAV: ${s['starting_nav_cents']/100:,.2f}",
        f"Ending NAV: ${s['ending_nav_cents']/100:,.2f}",
        f"Net return: {s['net_return']:.2%}",
        f"Costs: ${s['total_cost_cents']/100:,.2f}",
        f"Turnover: {s['total_turnover']:.3f}x",
        f"Max drawdown: {s['max_drawdown_bps']} bp",
        f"Mean cash/gross: {s['mean_cash_frac']:.2%}/{s['mean_gross_frac']:.2%}",
        f"Intents: {s['intents_submitted_total']}",
        f"Authority-blocked days: {s['authority_blocks']}/{s['days']}",
        f"Allocator: {s['allocator']}",
        f"Survivor wiring: {s.get('survivor_wiring', False)} (alpha={s.get('blend_alpha', 0):.2f})",
        "-" * 64,
        f"Cash baseline: {b['cash']:.2%}",
        f"{b['benchmark_symbol']} buy-and-hold gross/net: "
        f"{b['benchmark_buy_hold_gross']:.2%}/"
        f"{b['benchmark_buy_hold_net']:.2%}",
        f"Equal-weight gross/net: "
        f"{b['equal_weight_gross']:.2%}/"
        f"{b['equal_weight_net']:.2%}",
        "-" * 64,
    ]
    if s["no_authority"] and s["no_trades"]:
        lines.extend([
            "RESULT: NO AUTHORITY / NO TRADES",
            "Trackers received no attribution observations and remain in probation.",
            "All-cash is fail-closed behavior, not a performance or edge result.",
        ])
    elif s["no_authority"]:
        lines.append("RESULT: NO AUTHORITY — investigate any submitted intents.")
    elif s["no_trades"]:
        lines.append("RESULT: AUTHORITY PRESENT / NO TRADES.")
    else:
        lines.append(
            "TRADES OCCURRED — verify earned authority and Risk Kernel approval."
        )
    lines.extend([
        "-" * 64,
        "Limitations:",
        "- Back-adjusted, non-PIT panel; no confirmatory alpha claim.",
        "- Momentum magnitudes/rank cuts are uncalibrated placeholders.",
        "- Liquidity, ADV, beta, data age, and disagreement use assumptions.",
        "- Phase 2 SE path remains disabled pending null-suite pass and owner approval.",
        "- Cost-aware allocator engaged; friction reduction is mechanical, not alpha.",
        "- Survivor factors are academic LS returns used as regime indicators only.",
        "=" * 64,
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="evidence/panel_real_v1.csv")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--cash", type=int, default=10_000_000)
    parser.add_argument("--cov-window", type=int, default=63)
    parser.add_argument("--cost-bps", type=int, default=5)
    parser.add_argument("--min-trade-bps", type=int, default=5)
    parser.add_argument("--band-frac", type=float, default=0.25)
    parser.add_argument("--survivor-path", default="evidence/survivor_factors.csv")
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    result = run_simulation(
        panel,
        days=args.days,
        cash=args.cash,
        cov_window=args.cov_window,
        cost_bps=args.cost_bps,
        min_trade_bps=args.min_trade_bps,
        band_frac=args.band_frac,
        survivor_path=args.survivor_path,
        blend_alpha=args.blend_alpha,
    )
    print(format_report(result))
    print(
        f"Daily ledger: "
        f"{os.path.join(result['summary']['workdir'], 'daily.csv')}"
    )


if __name__ == "__main__":
    main()
