"""Session and shift scheduling. The continuous-running and night-shift layer.

SessionSchedule is a deterministic, dependency-free WINTER-offset fixture for
XNYS. It is NOT DST-aware and NOT exchange-calendar driven; it exists so tests
can run without exchange_calendars. For production timing use
CalendarSessionSchedule, which derives phases from actual exchange session
open/close instants (correct across DST, holidays, and early closes).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    NIGHT_RESEARCH = "night_research"
    PRE_MARKET = "pre_market"
    MARKET_HOURS = "market_hours"
    POST_MARKET = "post_market"
    CLOSED = "closed"


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


class CalendarSessionSchedule:
    """Exchange-calendar-driven schedule (DST, holidays, early closes).

    Derives market phases from the actual XNYS session open/close instants in
    UTC, so it is correct across DST transitions, exchange holidays, and early
    closes. Requires exchange_calendars (a declared dependency). Non-session
    days outside the night-research window return Phase.CLOSED. Timestamps must
    be timezone-aware; naive inputs are rejected.
    """

    def __init__(self, calendar_name: str = "XNYS",
                 night_start_utc_hour: int = 0, night_end_utc_hour: int = 6):
        import exchange_calendars as xcals
        for name, v in (("night_start_utc_hour", night_start_utc_hour),
                        ("night_end_utc_hour", night_end_utc_hour)):
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 23:
                raise ValueError(f"{name} must be an int hour in [0,23]")
        self._cal = xcals.get_calendar(calendar_name)
        self.night_start_utc_hour = night_start_utc_hour
        self.night_end_utc_hour = night_end_utc_hour

    def _in_night(self, hour: int) -> bool:
        if self.night_start_utc_hour <= self.night_end_utc_hour:
            return self.night_start_utc_hour <= hour < self.night_end_utc_hour
        return hour >= self.night_start_utc_hour or hour < self.night_end_utc_hour

    @staticmethod
    def _to_utc(ts):
        import pandas as pd
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return ts.tz_convert("UTC")

    def is_session_day(self, d) -> bool:
        import pandas as pd
        return bool(self._cal.is_session(pd.Timestamp(d)))

    def phase_at_ts(self, ts) -> Phase:
        import pandas as pd
        utc = self._to_utc(ts)
        if self._in_night(utc.hour):
            return Phase.NIGHT_RESEARCH
        local = utc.tz_convert(self._cal.tz)
        session = pd.Timestamp(local.date())
        if not self._cal.is_session(session):
            return Phase.CLOSED
        open_utc = self._cal.session_open(session).tz_convert("UTC")
        close_utc = self._cal.session_close(session).tz_convert("UTC")
        if utc < open_utc:
            return Phase.PRE_MARKET
        if utc < close_utc:
            return Phase.MARKET_HOURS
        return Phase.POST_MARKET

    def should_run_cycle(self, ts) -> bool:
        phase = self.phase_at_ts(ts)
        if phase is Phase.NIGHT_RESEARCH:
            return True
        if phase is Phase.CLOSED:
            return False
        return True
