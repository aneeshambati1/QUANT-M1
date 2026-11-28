"""Exploratory 12-1 momentum and 5-day reversal trial on the frozen panel.

Research diagnostic ONLY. Non-PIT panel; NOT confirmatory; no alpha claim.
The deployable system is long-only; the long-short line below is a research
diagnostic, not a tradable book. New budget family, separate from the 13
market trials. No parameter tuning after seeing results.

Formation (12-1): close[t-skip] / close[t-skip-lookback] - 1, skip=21, lookback=252.
Reversal (5):     close[t] / close[t-5] - 1, hold 5 days, long the BOTTOM ranks.
Holds are non-overlapping; HAC (Newey-West) t-stat on the hold-return series.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.core import sha


def formation_return(close: pd.DataFrame, t: int, lookback: int, skip: int) -> pd.Series:
    a = close.iloc[t - skip]
    b = close.iloc[t - skip - lookback]
    return a / b - 1.0


def hac_tstat(returns: np.ndarray, lag: int) -> float:
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    mean = x.mean()
    e = x - mean
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    var_mean = s / n
    if not math.isfinite(var_mean) or var_mean <= 0:
        return float("nan")
    return mean / math.sqrt(var_mean)


def _hold_returns(close, start, lookback, skip, hold, long_top, n_leg):
    rets = []
    t = start
    while t + hold < len(close):
        sig = formation_return(close, t, lookback, skip)
        ranks = sig.rank(pct=True)
        if long_top:
            leg = ranks[ranks > 0.8].index
        else:
            leg = ranks[ranks < 0.2].index
        if len(leg) == 0:
            t += hold
            continue
        fwd = close.iloc[t + hold][leg] / close.iloc[t][leg] - 1.0
        rets.append(float(fwd.mean()))
        t += hold
    return np.asarray(rets)


def run_trial(panel: pd.DataFrame) -> dict:
    close = panel[sorted(panel.columns)]
    n = len(close)
    warm = 252 + 21 + 1
    if n < warm + 40:
        raise ValueError(f"panel too short for 12-1 trial: need >= {warm + 40} rows")
    mom = _hold_returns(close, warm, 252, 21, 21, True, 2)
    rev = _hold_returns(close, 21, 5, 0, 5, False, 2)
    report = {
        "panel_rows": n,
        "momentum_12_1": {"n_holds": int(len(mom)),
                          "mean_hold_return": float(mom.mean()) if len(mom) else None,
                          "hac_t": hac_tstat(mom, 3)},
        "reversal_5": {"n_holds": int(len(rev)),
                       "mean_hold_return": float(rev.mean()) if len(rev) else None,
                       "hac_t": hac_tstat(rev, 3)},
        "disclosure": "EXPLORATORY, non-PIT, long-only deployable; long-short not used; no alpha claim",
    }
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="evidence/panel_real_v1.csv")
    ap.add_argument("--out", default="evidence/momentum_12_1_report.json")
    a = ap.parse_args()
    close = pd.read_csv(a.panel, index_col=0, parse_dates=True)
    rep = run_trial(close)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2, sort_keys=True)
    m, r = rep["momentum_12_1"], rep["reversal_5"]
    print(f"12-1 momentum: n={m['n_holds']} mean={m['mean_hold_return']} HAC t={m['hac_t']:.3f}")
    print(f"5d reversal:   n={r['n_holds']} mean={r['mean_hold_return']} HAC t={r['hac_t']:.3f}")
    print("EXPLORATORY ONLY — non-PIT panel; no confirmatory alpha claim.")


if __name__ == "__main__":
    main()
