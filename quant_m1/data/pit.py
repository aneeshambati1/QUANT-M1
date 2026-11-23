"""
Point-in-time bars.

No lookahead:
  - ts <= decision_ts
  - known_at <= decision_ts

Query-window invariance:
  changing the requested start date must not change the as-of history.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return dt.astimezone(timezone.utc)


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
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("prices must be positive")
        if self.high < self.low:
            raise ValueError("high < low")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open/close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open/close")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")
        if self.adj_factor <= 0:
            raise ValueError("adj_factor must be positive")
        if _parse_utc(self.known_at) < _parse_utc(self.ts):
            raise ValueError("known_at must be >= ts")


def as_of_bars(bars: Iterable[Bar], symbol: str, decision_ts: str) -> List[Bar]:
    cut = _parse_utc(decision_ts)
    out = [
        b for b in bars
        if b.symbol == symbol
        and _parse_utc(b.ts) <= cut
        and _parse_utc(b.known_at) <= cut
    ]
    return sorted(out, key=lambda b: b.ts)


def history_before(bars: Iterable[Bar], symbol: str, decision_ts: str) -> List[Bar]:
    cut = _parse_utc(decision_ts)
    out = [
        b for b in bars
        if b.symbol == symbol
        and _parse_utc(b.ts) < cut
        and _parse_utc(b.known_at) <= cut
    ]
    return sorted(out, key=lambda b: b.ts)


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
