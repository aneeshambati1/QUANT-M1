"""
Typed evidence boundary for AI/news/macro.

AI output may produce Evidence.
AI output may NOT produce OrderIntent, qty, side, price, or order_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EvidenceType(Enum):
    MACRO_RELEASE = "macro_release"
    FILING = "filing"
    EARNINGS_SURPRISE = "earnings_surprise"
    SENTIMENT_SHIFT = "sentiment_shift"
    REGIME_CHANGE = "regime_change"


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Evidence:
    type: EvidenceType
    symbol: Optional[str]
    release_ts: str
    known_at: str
    confidence: float
    direction: float
    magnitude: float
    source: str
    raw_ref: str
    vintage_id: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")
        if not (-1.0 <= self.direction <= 1.0):
            raise ValueError("direction must be in [-1,1]")
        if self.magnitude < 0:
            raise ValueError("magnitude must be >= 0")
        if _parse_utc(self.known_at) < _parse_utc(self.release_ts):
            raise ValueError("known_at must be >= release_ts")
        if not self.source or not self.raw_ref or not self.vintage_id:
            raise ValueError("source/raw_ref/vintage_id required")

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "symbol": self.symbol,
            "release_ts": self.release_ts,
            "known_at": self.known_at,
            "confidence": self.confidence,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "source": self.source,
            "raw_ref": self.raw_ref,
            "vintage_id": self.vintage_id,
        }
