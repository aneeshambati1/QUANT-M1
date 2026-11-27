#!/usr/bin/env python3
"""Honest exploratory simulation runner.

Uses evidence/panel_real_v1.csv, respects capital/position limits, and does NOT
artificially force tracker eligibility. If the system produces 0 trades, that
is the correct architectural outcome until models earn positive authority.

Usage:
    python scripts/run_simulation.py
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_m1.system import build
from quant_m1.meta.state import MarketState
from quant_m1.meta.contribution import ContributionTracker
from quant_m1.allocate import solve_allocation
from quant_m1.orchestrator.paper_loop import run_paper_step
from quant_m1.universe import register_etf_universe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="evidence/panel_real_v1.csv")
    parser.add_argument("--days", type=int, default=60)  # Short run for validation
    parser.add_argument("--cash", type=int, default=10_000_00)  # $100k in cents (respects per-order limits)
    args = parser.parse_args()

    # 1. Load real panel
    close = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    symbols = sorted(close.columns.tolist())  # Alphabetical sort matches generate_intents
    close = close[symbols]
    
    # Handle timestamps correctly (no tz_localize if already aware)
    if close.index.tzinfo is None:
        close.index = close.index.tz_localize("UTC")
        
    print(f"📊 Panel: {len(close)} dates × {len(symbols)} ETFs")
    print(f"   Range: {close.index[0].date()} to {close.index[-1].date()}")

    # 2. Build system with safe capital
    workdir = "/tmp/quant_sim_honest"
    os.makedirs(workdir, exist_ok=True)
    sx = build(workdir, cash=args.cash, positions={}, marks=None)
    
    # 3. Register universe properly (Instruments + Mandate Allowlist)
    register_etf_universe(sx, symbols)
    print(f"✅ Universe registered: {len(sx['instruments'])} instruments, mandate allowlist updated.")

    # 4. Initialize trackers honestly (NO synthetic authority injection)
    trackers = {e: ContributionTracker(half_life_obs=60.0, kappa=1.0, probation_obs=30) for e in symbols}
    print(f"⚠️  Trackers initialized honestly. They must earn authority through real contributions.")

    # 5. Run simulation
    sim_days = min(args.days, len(close))
    start_idx = len(close) - sim_days
    results = []
    authority_blocks = 0

    print(f"\n🚀 Running {sim_days}-day HONEST simulation...")
    for i in range(start_idx, len(close)):
        date = close.index[i]
        prices_cents = {k: int(round(v * 100)) for k, v in close.iloc[i].to_dict().items() if v > 0}
        if len(prices_cents) < len(symbols):
            continue

        ms = MarketState(
            as_of=date.isoformat(),  # Already UTC-aware
            avg_corr=0.4, vol_ann=0.15, liquidity_score=1.0,
            data_age_secs=10, model_disagreement=0.1, n_assets=len(symbols),
            drawdown_bps=0, drawdown_trigger_bps=5000,
        )
        
        # Calibrated forecast: simple momentum scaled to realistic daily bps
        if i >= 21:
            mom = close.iloc[i-21:i].pct_change().mean()
            ranks = mom.rank(pct=True).values
            er = np.where(ranks > 0.8, 0.001, np.where(ranks < 0.2, -0.001, 0.0))  # 10bps daily edge max
        else:
            er = np.zeros(len(symbols))
            
        fc = {
            "expected_returns": er,
            "cov": np.eye(len(symbols)) * 0.0002,  # Realistic daily variance
            "risk_aversion": 2.0,
            "horizon_days": 1,
        }

        rep = run_paper_step(
            sx, ms, trackers, fc,
            new_trade_date=sx["ledger"].state.trade_date + 1,
            new_mark_wall=int(date.timestamp()) + 1,
            new_marks=prices_cents,
            allocator=solve_allocation,
            min_trade_bps=5, cost_bps=5,
        )

        nav_cents = sx["ledger"].state.nav()
        results.append({"date": date, "nav": nav_cents / 100.0, "intents": rep.intents_submitted, "blocked": rep.blocked, "note": rep.note})
        
        if "No model has earned positive authority" in rep.note:
            authority_blocks += 1

        if (i - start_idx) % 10 == 0 and i > start_idx:
            print(f"  [{(i-start_idx)/sim_days*100:.0f}%] {date.date()} | NAV: ${nav_cents/100:,.2f} | Intents: {rep.intents_submitted}")

    # 6. Summary
    if results:
        df = pd.DataFrame(results).set_index("date")
        total_ret = (df["nav"].iloc[-1] / df["nav"].iloc[0]) - 1
        
        print("\n" + "=" * 60)
        print("📈 HONEST SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Period       : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Total Return : {total_ret:.2%}")
        print(f"  Active Days  : {(df['intents'] > 0).sum()}/{len(df)}")
        print(f"  Authority Blocks: {authority_blocks}/{len(df)}")
        print("=" * 60)
        
        if authority_blocks == len(df):
            print("\n✅ SYSTEM BEHAVIOR CORRECT:")
            print("   The system correctly withheld trading because models have not")
            print("   yet earned positive authority through real contributions.")
            print("   This is the intended fail-safe behavior.")
        elif (df['intents'] > 0).sum() > 0:
            print("\n⚠️  TRADES EXECUTED:")
            print("   Models earned authority and generated intents.")
            print("   Verify these are based on real contributions, not synthetic data.")
    else:
        print("\n❌ No results.")


if __name__ == "__main__":
    main()
