"""Price panel loading: bar semantics, canonical order, content identity.

BAR SEMANTICS: every index timestamp is the SESSION CLOSE. A signal from row t
uses information available at t's close; a forward return spans t to t+h closes.

CANONICAL ORDER: columns are sorted by name at construction. Column order is
not data, and an unstable vendor column order must never look like a price
revision. content_hash is therefore order-invariant by construction.

CONTENT IDENTITY: SHA-256 over the canonical index, sorted column names, and
the raw little-endian float64 values in that order. It changes if and only if
dates, symbols, or prices change.

column_hashes() gives per-symbol hashes so a genuine vendor revision can be
attributed to specific symbols rather than detected only in aggregate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataError(Exception):
    pass


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        raise DataError("duplicate column names")
    out = df.copy()
    out.columns = cols
    return out.loc[:, sorted(cols)]


def _hash_parts(index, names: list[str], values: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"index": list(index), "columns": names},
                        sort_keys=True, separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return h.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    c = _canonical(df)
    idx = [t.isoformat() for t in pd.DatetimeIndex(c.index)]
    return _hash_parts(idx, [str(x) for x in c.columns], c.to_numpy(dtype="<f8"))


def column_hashes(df: pd.DataFrame) -> dict:
    """Per-symbol hash. Use to attribute a revision to specific symbols."""
    c = _canonical(df)
    idx = [t.isoformat() for t in pd.DatetimeIndex(c.index)]
    return {str(col): _hash_parts(idx, [str(col)],
                                  c[[col]].to_numpy(dtype="<f8"))
            for col in c.columns}


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
    out = _canonical(df)
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