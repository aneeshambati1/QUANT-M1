"""Tests for unified settlement authority with historical T+2/T+1 transition."""
from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from quant_m1.research.settlement import (
    settlement_lag, settlement_date, is_settled, T1_TRANSITION,
)


def test_settlement_lag_before_transition():
    assert settlement_lag(date(2024, 5, 27)) == 2
    assert settlement_lag(date(2020, 1, 1)) == 2
    assert settlement_lag(date(2015, 6, 15)) == 2


def test_settlement_lag_on_and_after_transition():
    assert settlement_lag(date(2024, 5, 28)) == 1
    assert settlement_lag(date(2024, 5, 29)) == 1
    assert settlement_lag(date(2025, 1, 1)) == 1


def test_settlement_date_weekday():
    # 2024-05-27 is Monday, before transition -> T+2 = Wednesday 2024-05-29
    settle = settlement_date(date(2024, 5, 27))
    assert settle == pd.Timestamp("2024-05-29")


def test_settlement_date_after_transition():
    # 2024-05-28 is Tuesday, on transition -> T+1 = Wednesday 2024-05-29
    settle = settlement_date(date(2024, 5, 28))
    assert settle == pd.Timestamp("2024-05-29")


def test_settlement_date_weekend_skip():
    # 2024-05-24 is Friday, before transition -> T+2 = Tuesday 2024-05-28 (skip weekend)
    settle = settlement_date(date(2024, 5, 24))
    assert settle == pd.Timestamp("2024-05-28")


def test_is_settled_true():
    assert is_settled(date(2024, 5, 20), date(2024, 5, 22)) is True
    assert is_settled(date(2024, 5, 20), date(2024, 5, 23)) is True


def test_is_settled_false():
    assert is_settled(date(2024, 5, 20), date(2024, 5, 21)) is False


def test_settlement_date_accepts_timestamp():
    ts = pd.Timestamp("2024-05-27 14:30:00")
    settle = settlement_date(ts)
    assert settle == pd.Timestamp("2024-05-29")