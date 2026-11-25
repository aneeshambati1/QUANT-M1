"""Pre-registered baseline hypothesis. One trial, declared before running.

HYPOTHESIS (frozen):
  signal  = 21-session trailing return
  target  = next 5-session return
  method  = cross-sectional Spearman rank IC per session
  universe= fixed 10-ETF list, no selection search
  cost    = IC is GROSS; cost is assessed separately, not inside IC

No parameter search is performed here. Changing lookback/horizon after
seeing results creates a new trial and must be registered as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import PricePanel, DataError
from .ic import measure_ic, ICReport

PRE_REGISTERED = {
    "signal": "trailing_return",
    "lookback_sessions": 21,
    "horizon_sessions": 5,
    "method": "cross_sectional_spearman",
    "trials_planned": 1,
}

UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "USO", "UUP")


def momentum_signal(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing return ending at row t. Uses only rows <= t."""
    if lookback < 1:
        raise DataError("lookback must be >= 1")
    return close.pct_change(lookback)


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from t's close to (t+horizon)'s close. Uses rows > t."""
    if horizon < 1:
        raise DataError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def run_baseline(panel: PricePanel, *, lookback: int = 21, horizon: int = 5,
                 evidence_label: str | None = None) -> ICReport:
    s = momentum_signal(panel.close, lookback)
    f = forward_return(panel.close, horizon)
    sv, fv = s.to_numpy(dtype=float), f.to_numpy(dtype=float)
    usable = np.isfinite(sv).all(axis=1) & np.isfinite(fv).all(axis=1)
    if usable.sum() < 30:
        raise DataError(f"only {int(usable.sum())} usable sessions; need >= 30")
    label = evidence_label or panel.label
    return measure_ic(sv[usable], fv[usable],
                      horizon=f"{horizon}d", evidence_label=label)