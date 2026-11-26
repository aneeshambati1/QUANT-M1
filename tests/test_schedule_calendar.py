"""Calendar-driven schedule: DST, holidays, early closes, weekends."""
from __future__ import annotations

import pandas as pd
import pytest

from quant_m1.orchestrator.schedule import CalendarSessionSchedule, Phase, SessionSchedule


@pytest.fixture(scope="module")
def cal():
    return CalendarSessionSchedule("XNYS")


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_winter_open_close(cal):
    assert cal.phase_at_ts(_ts("2024-01-16 14:29")) is Phase.PRE_MARKET
    assert cal.phase_at_ts(_ts("2024-01-16 14:30")) is Phase.MARKET_HOURS
    assert cal.phase_at_ts(_ts("2024-01-16 20:59")) is Phase.MARKET_HOURS
    assert cal.phase_at_ts(_ts("2024-01-16 21:00")) is Phase.POST_MARKET


def test_summer_dst_open_close(cal):
    assert cal.phase_at_ts(_ts("2024-07-16 13:29")) is Phase.PRE_MARKET
    assert cal.phase_at_ts(_ts("2024-07-16 13:30")) is Phase.MARKET_HOURS
    assert cal.phase_at_ts(_ts("2024-07-16 19:59")) is Phase.MARKET_HOURS
    assert cal.phase_at_ts(_ts("2024-07-16 20:00")) is Phase.POST_MARKET


def test_early_close(cal):
    assert cal.phase_at_ts(_ts("2024-07-03 16:59")) is Phase.MARKET_HOURS
    assert cal.phase_at_ts(_ts("2024-07-03 17:00")) is Phase.POST_MARKET


def test_holiday_closed(cal):
    assert cal.phase_at_ts(_ts("2024-07-04 15:00")) is Phase.CLOSED
    assert cal.phase_at_ts(_ts("2024-07-04 02:00")) is Phase.NIGHT_RESEARCH


def test_weekend_closed(cal):
    assert cal.phase_at_ts(_ts("2024-07-06 15:00")) is Phase.CLOSED


def test_naive_timestamp_rejected(cal):
    with pytest.raises(ValueError):
        cal.phase_at_ts(pd.Timestamp("2024-07-16 15:00"))


def test_should_run_cycle(cal):
    assert cal.should_run_cycle(_ts("2024-07-04 15:00")) is False
    assert cal.should_run_cycle(_ts("2024-07-04 02:00")) is True
    assert cal.should_run_cycle(_ts("2024-07-16 15:00")) is True


def test_session_schedule_fixture_unchanged():
    s = SessionSchedule()
    assert s.phase_at(15) is Phase.MARKET_HOURS
    assert s.phase_at(2) is Phase.NIGHT_RESEARCH
