"""Session and shift scheduling. The continuous-running and night-shift layer.

ASSUMPTION: hour boundaries below are UTC and reflect northern-hemisphere
WINTER offsets for XNYS. They are NOT DST-aware and NOT exchange-calendar
driven. A production deployment must derive phases from the exchange calendar
(settlement.ExchangeCalendar) because the US market close shifts between 20:00
and 21:00 UTC across the DST transition. This module is deterministic and
dependency-free so it can be tested without exchange_calendars installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    NIGHT_RESEARCH = "night_research"
    PRE_MARKET = "pre_market"
    MARKET_HOURS = "market_hours"
    POST_MARKET = "post_market"


@dataclass(frozen=True)
class SessionSchedule:
    market_open_utc_hour: int = 14
    market_close_utc_hour: int = 21
    night_start_utc_hour: int = 0
    night_end_utc_hour: int = 6

    def __post_init__(self) -> None:
        for name in ("market_open_utc_hour", "market_close_utc_hour",
                     "night_start_utc_hour", "night_end_utc_hour"):
            v = getattr(self, name)
            if not isinstance(v, int) or not 0 <= v <= 23:
                raise ValueError(f"{name} must be an int hour in [0,23]")
        if self.market_open_utc_hour >= self.market_close_utc_hour:
            raise ValueError("market_open_utc_hour must be < market_close_utc_hour")

    def _in_night(self, hour: int) -> bool:
        if self.night_start_utc_hour <= self.night_end_utc_hour:
            return self.night_start_utc_hour <= hour < self.night_end_utc_hour
        return hour >= self.night_start_utc_hour or hour < self.night_end_utc_hour

    def phase_at(self, hour_utc: int) -> Phase:
        if not isinstance(hour_utc, int) or not 0 <= hour_utc <= 23:
            raise ValueError("hour_utc must be an int in [0,23]")
        if self.market_open_utc_hour <= hour_utc < self.market_close_utc_hour:
            return Phase.MARKET_HOURS
        if self._in_night(hour_utc):
            return Phase.NIGHT_RESEARCH
        if hour_utc < self.market_open_utc_hour:
            return Phase.PRE_MARKET
        return Phase.POST_MARKET

    def should_run_cycle(self, hour_utc: int, is_session_day: bool) -> bool:
        """The loop is always allowed to THINK; whether it may TRADE is decided
        by the policy layer, not here. Research and market phases always run.
        On non-session days only the night research window runs.
        """
        phase = self.phase_at(hour_utc)
        if phase is Phase.NIGHT_RESEARCH:
            return True
        if not is_session_day:
            return False
        return phase in (Phase.PRE_MARKET, Phase.MARKET_HOURS, Phase.POST_MARKET)
