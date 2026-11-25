"""Price panel loading with explicit bar semantics.

BAR SEMANTICS (fixes the bar-open ambiguity): every index timestamp is the
SESSION CLOSE. The close value is knowable only at or after that timestamp.
A signal computed from row t therefore uses information available at t's
close, and a forward return from t spans t's close to (t+h)'s close.

Daily closes are used deliberately. Intraday bars require a bar-end
availability timestamp that most free vendors do not supply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataError(Exception):
    pass


@dataclass(frozen=True)
class PricePanel:
    close: pd.DataFrame
    source: str
    vintage: str
    pit: bool
    label: str

    def as_dict(self) -> dict:
        return {"source": self.source, "vintage": self.vintage,
                "pit": self.pit, "label": self.label,
                "n_rows": int(self.close.shape[0]),
                "n_symbols": int(self.close.shape[1]),
                "start": str(self.close.index[0]),
                "end": str(self.close.index[-1])}


def panel_from_frame(df: pd.DataFrame, *, source: str, vintage: str,
                     pit: bool, label: str) -> PricePanel:
    if df is None or df.empty:
        raise DataError("empty price frame")
    if df.shape[1] < 2:
        raise DataError("need at least 2 symbols for cross-sectional IC")
    if df.shape[0] < 40:
        raise DataError("need at least 40 sessions")
    idx = pd.DatetimeIndex(df.index)
    if not idx.is_monotonic_increasing:
        raise DataError("index must be sorted ascending")
    if idx.has_duplicates:
        raise DataError("duplicate session timestamps")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    v = out.to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise DataError("no finite prices")
    if np.any(finite <= 0.0):
        raise DataError("prices must be strictly positive")
    if out.isna().all().any():
        raise DataError("symbol column is entirely NaN")
    if not label or not label.strip():
        raise DataError("label is required")
    return PricePanel(out, source, vintage, pit, label)


def load_csv(path: str, *, vintage: str, label: str) -> PricePanel:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return panel_from_frame(df, source=f"csv:{path}", vintage=vintage,
                            pit=False, label=label)


def load_yfinance(symbols, start: str, end: str) -> PricePanel:
    """PROVISIONAL source. Back-adjusted closes, no vintage history, NOT PIT."""
    import yfinance as yf
    raw = yf.download(list(symbols), start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or len(raw) == 0:
        raise DataError("yfinance returned no rows")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all")
    return panel_from_frame(
        close, source="yfinance", vintage=f"downloaded:{pd.Timestamp.utcnow().isoformat()}",
        pit=False,
        label="PROVISIONAL - yfinance back-adjusted, NON-PIT, gross of cost")