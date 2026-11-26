"""
Typed evidence boundary for AI/news/macro.

AI output may produce Evidence.
AI output may NOT produce OrderIntent, qty, side, price, or order_id.
"""
from __future__ import annotations

import math
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


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _parse_utc(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 timestamp")

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp") from exc

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")

    if dt.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must be expressed in UTC")

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
        if not isinstance(self.type, EvidenceType):
            raise ValueError("type must be an EvidenceType")

        if self.symbol is not None:
            if not isinstance(self.symbol, str) or not self.symbol.strip():
                raise ValueError("symbol must be None or a non-empty string")

        release_dt = _parse_utc(self.release_ts, "release_ts")
        known_dt = _parse_utc(self.known_at, "known_at")

        if known_dt < release_dt:
            raise ValueError("known_at must be >= release_ts")

        if not _finite_number(self.confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        if not _finite_number(self.direction):
            raise ValueError("direction must be finite")
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError("direction must be in [-1, 1]")

        if not _finite_number(self.magnitude):
            raise ValueError("magnitude must be finite")
        if self.magnitude < 0.0:
            raise ValueError("magnitude must be >= 0")

        for name, value in (
            ("source", self.source),
            ("raw_ref", self.raw_ref),
            ("vintage_id", self.vintage_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

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
