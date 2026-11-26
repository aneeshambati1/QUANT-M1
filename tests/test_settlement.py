"""Holiday-aware settlement authority tests (XNYS sessions, T+2/T+1 transition)."""
from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from quant_m1.research.settlement import settlement_lag, settlement_date, is_settled


def test_settlement_lag_before_transition():
    assert settlement_lag(date(2024, 5, 27)) == 2
    assert settlement_lag(date(2020, 1, 1)) == 2


def test_settlement_lag_on_and_after_transition():
    assert settlement_lag(date(2024, 5, 28)) == 1
    assert settlement_lag(date(2025, 1, 1)) == 1


def test_settlement_date_skips_weekend_and_holiday():
    # 2024-05-24 Fri, T+2: sessions 05-28 (05-27 Memorial Day closed) then 05-29.
    assert settlement_date(date(2024, 5, 24)) == pd.Timestamp("2024-05-29")


def test_settlement_date_after_transition():
    assert settlement_date(date(2024, 5, 28)) == pd.Timestamp("2024-05-29")


def test_settlement_date_friday_to_monday():
    assert settlement_date(date(2024, 6, 7)) == pd.Timestamp("2024-06-10")


def test_settlement_date_skips_july4():
    # 2024-07-03 Wed (early close, still a session), T+1 -> 2024-07-05 Fri.
    assert settlement_date(date(2024, 7, 3)) == pd.Timestamp("2024-07-05")


def test_holiday_trade_date_rejected():
    for d in [date(2024, 5, 27), date(2024, 7, 4), date(2024, 3, 29)]:
        with pytest.raises(ValueError):
            settlement_date(d)


def test_weekend_trade_date_rejected():
    with pytest.raises(ValueError):
        settlement_date(date(2024, 5, 25))


def test_is_settled_true():
    assert is_settled(date(2024, 5, 20), date(2024, 5, 22)) is True


def test_is_settled_false():
    assert is_settled(date(2024, 5, 20), date(2024, 5, 21)) is False


def test_settlement_date_accepts_session_timestamp():
    assert settlement_date(pd.Timestamp("2024-05-28 14:30:00")) == pd.Timestamp("2024-05-29")
