"""Batch E smoke: sleeve + selector + price signals on the frozen panel.

RESEARCH ONLY. Builds a provisional cold-start tracker set, runs the regime
selector and price signal families each day (no lookahead), computes bounded
sleeve exploration weights, and writes a hashed report. It does NOT trade, does
NOT route to any venue, and does NOT update authority or feed the ProbeJournal.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.core import sha
from quant_m1.meta.contribution import ContributionTracker
from quant_m1.meta.sleeve import sleeve_weights, sleeve_report
from quant_m1.meta.selector import select_weights, renormalize_available
from quant_m1.meta.state import MarketState
from quant_m1.research.signals import momentum_signal, reversal_signal
from quant_m1.research.carry import carry_signal


def load_panel(path: str) -> pd.DataFrame:
    close = pd.read_csv(path, index_col=0, parse_dates=True)
    close = close[sorted(close.columns.tolist())]
    if close.index.tzinfo is None:
        close.index = close.index.tz_localize("UTC")
    return close


def run_smoke(panel: pd.DataFrame, days: int = 30, cov_window: int = 63) -> dict:
    symbols = sorted(panel.columns.tolist())
    n = len(symbols)
    start = max(len(panel) - days, cov_window + 1)
    trackers = {
        "mom_21": ContributionTracker(horizon_days=1),
        "rev_5": ContributionTracker(horizon_days=1),
    }
    rows = []
    for i in range(start, len(panel)):
        X = panel.iloc[i - cov_window:i].pct_change(fill_method=None).dropna().values
        vol_ann = float(X.mean(axis=1).std(ddof=1) * np.sqrt(252)) if X.shape[0] > 1 else 0.0
        corr = np.corrcoef(X, rowvar=False)
        avg_corr = float(corr[~np.eye(n, dtype=bool)].mean()) if n > 1 else 0.0
        ms = MarketState(as_of=panel.index[i].isoformat(), avg_corr=avg_corr,
                         vol_ann=vol_ann, liquidity_score=1.0, data_age_secs=0,
                         model_disagreement=0.0, n_assets=n,
                         drawdown_bps=0, drawdown_trigger_bps=1000)
        sel = select_weights(ms)
        avail = {"momentum": True, "reversal": True,
                 "carry": bool(carry_signal(None)["available"])}
        eff = renormalize_available(sel, avail)
        mom = momentum_signal(panel, i, 21)
        rev = reversal_signal(panel, i, 5)
        blended = eff.get("momentum", 0.0) * mom + eff.get("reversal", 0.0) * rev
        sw = sleeve_weights(trackers)
        rows.append({"date": str(panel.index[i].date()), "regime": ms.regime.value,
                     "selector": eff, "sleeve_total": sum(sw.values()),
                     "top_blended_asset": symbols[int(np.argmax(blended))] if n else None})
    rep = sleeve_report(sleeve_weights(trackers))
    rep["days"] = rows
    rep["n_days"] = len(rows)
    rep["live_trading"] = False
    rep["disclosure"] = ("Paper-only smoke of sleeve+selector+signals. Non-PIT panel. "
                         "No orders, no authority, no alpha claim.")
    rep["sha"] = sha({k: v for k, v in rep.items() if k != "sha"})
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="evidence/panel_real_v1.csv")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="evidence/batchE_sleeve_smoke.json")
    a = ap.parse_args()
    rep = run_smoke(load_panel(a.panel), days=a.days)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"days={rep['n_days']} sleeve_total={rep['total_sleeve_weight']:.4f}")
    print("LIVE_TRADING=False AUTHORITY_GRANTED=False PHASE2=disabled")
    print("RESEARCH ONLY — non-PIT panel; no profitability claim.")


if __name__ == "__main__":
    main()
