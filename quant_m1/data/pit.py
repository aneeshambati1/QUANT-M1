"""
Point-in-time bars.

No lookahead:
  - ts <= decision_ts
  - known_at <= decision_ts

Query-window invariance:
  changing the requested start date must not change the as-of history.

Ordering is by NORMALISED UTC instant, never by raw timestamp string, so mixed
offsets sort correctly. Duplicate (symbol, ts) revisions resolve to the latest
vintage known at the decision time; conflicting identical-known_at duplicates
are rejected rather than silently chosen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List


def _parse_utc(s: str) -> datetime:
    if not isinstance(s, str) or not s.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return dt.astimezone(timezone.utc)


def _finite_pos(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_factor: float
    known_at: str
    source: str
    vintage_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        for name in ("open", "high", "low", "close", "adj_factor"):
            _finite_pos(getattr(self, name), name)
        if isinstance(self.volume, bool) or not isinstance(self.volume, (int, float)):
            raise ValueError("volume must be a finite number")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and >= 0")
        if self.high < self.low:
            raise ValueError("high < low")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open/close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open/close")
        if _parse_utc(self.known_at) < _parse_utc(self.ts):
            raise ValueError("known_at must be >= ts")


def _resolve(bars: List[Bar]) -> List[Bar]:
    """Latest known_at per ts wins; deterministic tie-break on vintage_id."""
    by_ts: dict[datetime, Bar] = {}
    for b in bars:
        key = _parse_utc(b.ts)
        prev = by_ts.get(key)
        if prev is None:
            by_ts[key] = b
            continue
        bk, pk = _parse_utc(b.known_at), _parse_utc(prev.known_at)
        if bk > pk:
            by_ts[key] = b
        elif bk == pk:
            if b == prev:
                continue
            if b.vintage_id != prev.vintage_id:
                by_ts[key] = max((b, prev), key=lambda x: x.vintage_id)
            else:
                raise ValueError(
                    f"conflicting duplicate bar for {b.symbol} at {b.ts}")
    return [by_ts[k] for k in sorted(by_ts)]


def as_of_bars(bars: Iterable[Bar], symbol: str, decision_ts: str) -> List[Bar]:
    cut = _parse_utc(decision_ts)
    out = [
        b for b in bars
        if b.symbol == symbol
        and _parse_utc(b.ts) <= cut
        and _parse_utc(b.known_at) <= cut
    ]
    return _resolve(out)


def history_before(bars: Iterable[Bar], symbol: str, decision_ts: str) -> List[Bar]:
    cut = _parse_utc(decision_ts)
    out = [
        b for b in bars
        if b.symbol == symbol
        and _parse_utc(b.ts) < cut
        and _parse_utc(b.known_at) <= cut
    ]
    return _resolve(out)


def query_window_invariant(
    bars: Iterable[Bar],
    symbol: str,
    decision_ts: str,
    start_a: str,
    start_b: str,
) -> bool:
    bars = list(bars)
    a = [b for b in as_of_bars(bars, symbol, decision_ts) if _parse_utc(b.ts) >= _parse_utc(start_a)]
    b = [b for b in as_of_bars(bars, symbol, decision_ts) if _parse_utc(b.ts) >= _parse_utc(start_b)]
    common_start = max(_parse_utc(start_a), _parse_utc(start_b))
    a = [x for x in a if _parse_utc(x.ts) >= common_start]
    b = [x for x in b if _parse_utc(x.ts) >= common_start]
    return a == b
