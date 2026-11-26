"""Price panel loading: bar semantics, canonical order, content identity."""
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
    h.update(json.dumps(
        {"index": list(index), "columns": names},
        sort_keys=True, separators=(",", ":"),
    ).encode())
    h.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return h.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    c = _canonical(df)
    idx = [t.isoformat() for t in pd.DatetimeIndex(c.index)]
    return _hash_parts(idx, list(c.columns), c.to_numpy(dtype="<f8"))


def column_hashes(df: pd.DataFrame) -> dict:
    c = _canonical(df)
    idx = [t.isoformat() for t in pd.DatetimeIndex(c.index)]
    return {
        str(col): _hash_parts(
            idx, [str(col)], c[[col]].to_numpy(dtype="<f8")
        )
        for col in c.columns
    }


@dataclass(frozen=True)
class PricePanel:
    close: pd.DataFrame
    source: str
    vintage: str
    pit: bool
    label: str
    content_hash: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "vintage": self.vintage,
            "pit": self.pit,
            "label": self.label,
            "content_hash": self.content_hash,
            "n_rows": int(self.close.shape[0]),
            "n_symbols": int(self.close.shape[1]),
            "start": str(self.close.index[0]),
            "end": str(self.close.index[-1]),
        }


def panel_from_frame(
    df: pd.DataFrame,
    *,
    source: str,
    vintage: str,
    pit: bool,
    label: str,
) -> PricePanel:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataError("empty or invalid price frame")
    if df.shape[1] < 2:
        raise DataError("need at least 2 symbols for cross-sectional IC")
    if df.shape[0] < 40:
        raise DataError("need at least 40 sessions")
    if not isinstance(pit, (bool, np.bool_)):
        raise DataError("pit must be an explicit boolean")
    if not isinstance(source, str) or not source.strip():
        raise DataError("source is required")
    if not isinstance(vintage, str) or not vintage.strip():
        raise DataError("vintage is required")
    if not isinstance(label, str) or not label.strip():
        raise DataError("label is required")

    idx = pd.DatetimeIndex(df.index)
    if not idx.is_monotonic_increasing:
        raise DataError("index must be sorted ascending")
    if idx.has_duplicates:
        raise DataError("duplicate session timestamps")
    if idx.hasnans:
        raise DataError("index contains NaT")

    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = _canonical(df)
    out.index = idx

    values = out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataError("prices contain NaN or infinite values")
    if np.any(values <= 0.0):
        raise DataError("prices must be strictly positive")

    return PricePanel(
        out, source, vintage, bool(pit), label, content_hash(out)
    )


def load_csv(path: str, *, vintage: str, label: str) -> PricePanel:
    df = pd.read_csv(
        path, index_col=0, parse_dates=True, float_precision="round_trip"
    )
    return panel_from_frame(
        df, source=f"csv:{path}", vintage=vintage, pit=False, label=label
    )


def load_yfinance(symbols, start: str, end: str) -> PricePanel:
    """Provisional source: adjusted closes, no historical PIT guarantee."""
    import yfinance as yf

    raw = yf.download(
        list(symbols), start=start, end=end,
        auto_adjust=True, progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise DataError("yfinance returned no rows")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all")

    return panel_from_frame(
        close,
        source="yfinance",
        vintage=f"downloaded:{pd.Timestamp.utcnow().isoformat()}",
        pit=False,
        label="PROVISIONAL - yfinance back-adjusted, NON-PIT, gross of cost",
    )
