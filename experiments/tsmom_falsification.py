"""Pre-registered TSMOM falsification experiment.

This is a research screen only. Passing its thresholds is not paper-trading
approval, proof of edge, or evidence of live-trading readiness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_m1.data.freeze import load_frozen
from quant_m1.research.tsmom import generate_tsmom_weights
from quant_m1.research.dsr import deflated_sharpe


N_TRIALS = 14
COST_BPS = 3.0


def run_falsification():
    print("=" * 70)
    print("TSMOM FALSIFICATION EXPERIMENT")
    print("RESEARCH SCREEN ONLY - NOT A TRADE AUTHORIZATION")
    print("=" * 70)

    try:
        panel = load_frozen("evidence", "panel_real_v1")
    except Exception as exc:
        raise RuntimeError(f"Could not load frozen panel: {exc}") from exc

    print(f"Loaded panel: {panel.content_hash[:16]}...")
    print(f"Evidence label: {panel.label}")

    prices = panel.close
    weights = generate_tsmom_weights(prices, target_vol=0.10)

    daily_returns = prices.pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Target weights formed at t close; held for return t+1.
    held_weights = weights.shift(1).fillna(0.0)
    gross_returns = (held_weights * daily_returns).sum(axis=1)

    # Turnover is measured when target weights change.
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    turnover = turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    daily_costs = turnover * (COST_BPS / 10_000.0)
    net_returns = gross_returns - daily_costs

    net_returns = net_returns.replace([np.inf, -np.inf], np.nan).dropna()
    gross_returns = gross_returns.reindex(net_returns.index)

    if len(net_returns) < 100:
        raise RuntimeError("Insufficient valid daily observations for this screen")

    ann_gross_return = float(gross_returns.mean() * 252)
    ann_net_return = float(net_returns.mean() * 252)
    ann_vol = float(net_returns.std(ddof=1) * np.sqrt(252))
    net_sharpe = ann_net_return / ann_vol if ann_vol > 0 else 0.0

    equity = (1.0 + net_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    # Pandas kurtosis() is excess kurtosis; DSR expects raw kurtosis.
    excess_kurtosis = float(net_returns.kurtosis())
    raw_kurtosis = excess_kurtosis + 3.0

    dsr_result = deflated_sharpe(
        sr_ann=net_sharpe,
        n_obs=len(net_returns),
        n_trials=N_TRIALS,
        skew=float(net_returns.skew()),
        kurtosis_raw=raw_kurtosis,
        periods_per_year=252,
    )
    dsr = float(dsr_result["dsr"])

    annual_turnover = float(turnover.sum() / (len(prices) / 252))

    print("\nRESULTS")
    print("-" * 70)
    print(f"Annualized gross return: {ann_gross_return * 100:.2f}%")
    print(f"Annualized net return:   {ann_net_return * 100:.2f}%")
    print(f"Annualized volatility:   {ann_vol * 100:.2f}%")
    print(f"Net Sharpe:              {net_sharpe:.3f}")
    print(f"Max drawdown:            {max_dd * 100:.2f}%")
    print(f"Annualized turnover:     {annual_turnover:.2f}x")
    print(f"Observations:             {len(net_returns)}")
    print(f"Trials assumed:           {N_TRIALS}")
    print(f"Raw kurtosis:             {raw_kurtosis:.4f}")
    print(f"Deflated Sharpe:          {dsr:.4f}")

    passed_dsr = dsr >= 0.95
    passed_sharpe = net_sharpe >= 0.10
    passed_dd = max_dd >= -0.25

    print("\nSCREEN")
    print("-" * 70)
    print(f"DSR >= 0.95:       {'PASS' if passed_dsr else 'FAIL'}")
    print(f"Net Sharpe >= 0.10:{'PASS' if passed_sharpe else 'FAIL'}")
    print(f"Max DD >= -25%:    {'PASS' if passed_dd else 'FAIL'}")

    if passed_dsr and passed_sharpe and passed_dd:
        print("\nRESULT: SCREEN SURVIVED.")
        print("This does NOT authorize paper trading.")
        print("Require independent review, cost stress, walk-forward tests,")
        print("and the complete research acceptance gate.")
    else:
        print("\nRESULT: SCREEN FAILED.")
        print("Preserve the negative result; do not optimize this panel post hoc.")


if __name__ == "__main__":
    run_falsification()
