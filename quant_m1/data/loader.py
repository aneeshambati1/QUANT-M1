"""Price panel loading with explicit bar semantics and content identity.

BAR SEMANTICS: every index timestamp is the SESSION CLOSE. The close value
is knowable only at or after that timestamp. A signal from row t uses
information available at t's close; a forward return spans t to t+h closes.

CONTENT IDENTITY: content_hash is SHA-256 over the canonical index, columns,
and the raw little-endian float64 values. It changes when prices change even
if source, dates, symbols and row count are identical. This is what makes a
silent vendor revision detectable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataError(Exception):
    pass


def content_hash(df: pd.DataFrame) -> str:
    idx = [t.isoformat() for t in pd.DatetimeIndex(df.index)]
    cols = [str(c) for c in df.columns]
    vals = np.ascontiguousarray(df.to_numpy(dtype="<f8"))
    h = hashlib.sha256()
    h.update(json.dumps({"index": idx, "columns": cols},
                        sort_keys=True, separators=(",", ":")).encode())
    h.update(vals.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PricePanel:
    close: pd.DataFrame
    source: str
    vintage: str
    pit: bool
    label: str
    content_hash: str = ""

    def as_dict(self) -> dict:
        return {"source": self.source, "vintage": self.vintage,
                "pit": self.pit, "label": self.label,
                "content_hash": self.content_hash,
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
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
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
    return PricePanel(out, source, vintage, pit, label, content_hash(out))


def load_csv(path: str, *, vintage: str, label: str) -> PricePanel:
    df = pd.read_csv(path, index_col=0, parse_dates=True,
                     float_precision="round_trip")
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
        close, source="yfinance",
        vintage=f"downloaded:{pd.Timestamp.utcnow().isoformat()}", pit=False,
        label="PROVISIONAL - yfinance back-adjusted, NON-PIT, gross of cost")