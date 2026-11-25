"""Unified settlement authority with historical T+2/T+1 transition.

Before 2024-05-28: US equity settlement was T+2.
On and after 2024-05-28: US equity settlement is T+1.

This module provides a single source of truth for settlement lag. 
SETTLEMENT_DATE INVARIANT: cash from a sale is available on the first
exchange session >= trade_date + settlement_lag(trade_date).
"""
from __future__ import annotations

import pandas as pd
from datetime import date, datetime

T1_TRANSITION = pd.Timestamp("2024-05-28").date()


def settlement_lag(trade_date: date | datetime | pd.Timestamp) -> int:
    """Return settlement lag in business days for the given trade date.

    Returns 2 before 2024-05-28, 1 on or after.
    """
    if isinstance(trade_date, (datetime, pd.Timestamp)):
        trade_date = trade_date.date() if hasattr(trade_date, 'date') else trade_date
    if not isinstance(trade_date, date):
        trade_date = pd.Timestamp(trade_date).date()

    if trade_date >= T1_TRANSITION:
        return 1
    return 2


def settlement_date(trade_date: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Compute the settlement date for a trade executed on trade_date.

    Uses standard US business days (skips weekends). Returns the first 
    exchange session that is >= trade_date + settlement_lag business days.
    """
    if not isinstance(trade_date, pd.Timestamp):
        td = pd.Timestamp(trade_date).normalize()
    else:
        td = trade_date.normalize()

    lag = settlement_lag(td)
    # bdate_range includes the start date. periods=lag+1 gives the lag-th 
    # business day after the start date.
    return pd.bdate_range(start=td, periods=lag + 1)[-1]


def is_settled(trade_date: date | datetime | pd.Timestamp,
               check_date: date | datetime | pd.Timestamp) -> bool:
    """Return True if a trade executed on trade_date is settled by check_date."""
    settle = settlement_date(trade_date)
    check = pd.Timestamp(check_date).normalize()
    return check >= settle