"""Pre-registered TSMOM Falsification Experiment.

Runs the TSMOM strategy on the frozen panel and applies the hard kill criteria:
1. Deflated Sharpe Ratio (DSR) >= 0.95
2. Net Sharpe >= 0.10
3. Max Drawdown <= 25%

If it fails, the strategy family is rejected for this universe.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure we can import from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_m1.data.freeze import load_frozen
from quant_m1.research.tsmom import generate_tsmom_weights
from quant_m1.research.dsr import deflated_sharpe


def run_falsification():
    print("=" * 70)
    print("TSMOM FALSIFICATION EXPERIMENT")
    print("=" * 70)

    # 1. Load the frozen panel
    try:
        panel = load_frozen("evidence", "panel_real_v1")
        print(f"Loaded frozen panel: {panel.content_hash[:16]}...")
    except Exception as e:
        print(f"FATAL: Could not load frozen panel. {e}")
        sys.exit(1)

    prices = panel.close

    # 2. Generate TSMOM weights
    print("\nGenerating TSMOM weights (12mo trend, 1mo skip, 10% vol target)...")
    weights = generate_tsmom_weights(prices, target_vol=0.10)

    # 3. Calculate daily returns
    daily_returns = prices.pct_change().fillna(0.0)

    # 4. Calculate gross portfolio returns
    # Shift weights by 1 to avoid lookahead bias (trade at close, realize next day)
    gross_returns = (weights.shift(1) * daily_returns).sum(axis=1)

    # 5. Apply transaction costs
    # Calculate daily turnover (sum of absolute weight changes)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    annual_turnover = turnover.sum() / (len(prices) / 252)

    # Assume 3 bp one-way cost (conservative based on cost_attribution)
    cost_bps = 3.0
    daily_costs = turnover * (cost_bps / 10_000.0)

    net_returns = gross_returns - daily_costs

    # 6. Calculate metrics
    ann_gross_return = gross_returns.mean() * 252
    ann_net_return = net_returns.mean() * 252
    ann_vol = net_returns.std() * np.sqrt(252)
    net_sharpe = ann_net_return / ann_vol if ann_vol > 0 else 0.0

    # Max Drawdown
    cum_returns = (1 + net_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # Deflated Sharpe Ratio (assuming 14 total trials now: 13 momentum + 1 TSMOM)
    n_trials = 14
    dsr = deflated_sharpe(
        net_sharpe,
        n_trials,
        skew=float(net_returns.skew()),
        kurtosis=float(net_returns.kurtosis()),
        years=10.0,
    )

    # 7. Print Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Annualized Gross Return: {ann_gross_return * 100:.2f}%")
    print(f"Annualized Net Return:   {ann_net_return * 100:.2f}%")
    print(f"Annualized Volatility:   {ann_vol * 100:.2f}%")
    print(f"Net Sharpe Ratio:        {net_sharpe:.3f}")
    print(f"Max Drawdown:            {max_dd * 100:.2f}%")
    print(f"Annualized Turnover:     {annual_turnover:.2f}x")
    print(f"Deflated Sharpe (N=14):  {dsr:.4f}")

    # 8. Apply Kill Criteria
    print("\n" + "=" * 70)
    print("KILL CRITERIA GATE")
    print("=" * 70)

    passed_dsr = dsr >= 0.95
    passed_sharpe = net_sharpe >= 0.10
    passed_dd = max_dd >= -0.25  # Max 25% drawdown

    print(f"1. DSR >= 0.95:        {'PASS' if passed_dsr else 'FAIL'} ({dsr:.4f})")
    print(f"2. Net Sharpe >= 0.10: {'PASS' if passed_sharpe else 'FAIL'} ({net_sharpe:.3f})")
    print(f"3. Max DD >= -25%:     {'PASS' if passed_dd else 'FAIL'} ({max_dd * 100:.2f}%)")

    if passed_dsr and passed_sharpe and passed_dd:
        print("\nVERDICT: SURVIVED. Proceed to paper trading validation.")
    else:
        print("\nVERDICT: FAILED. Strategy family rejected for this universe.")
        print("Do not run further parameter grids on this panel.")


if __name__ == "__main__":
    run_falsification()