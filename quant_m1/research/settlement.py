"""Unified settlement authority with historical T+2/T+1 transition.

Before 2024-05-28: US equity settlement was T+2.
On and after 2024-05-28: US equity settlement is T+1.

SETTLEMENT_DATE INVARIANT: cash from a sale is available after settlement_lag
EXCHANGE SESSIONS (XNYS), not calendar/business days. Weekends AND exchange
holidays are skipped. A trade_date that is not an exchange session is rejected.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

T1_TRANSITION = pd.Timestamp("2024-05-28").date()


@lru_cache(maxsize=1)
def _calendar():
    return xcals.get_calendar("XNYS")


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def settlement_lag(trade_date) -> int:
    """Return settlement lag in exchange sessions: 2 before 2024-05-28, else 1."""
    return 1 if _to_date(trade_date) >= T1_TRANSITION else 2


def settlement_date(trade_date) -> pd.Timestamp:
    """First exchange session that is settlement_lag sessions after trade_date.

    Rejects a trade_date that is not an XNYS session (weekend/holiday).
    """
    d = _to_date(trade_date)
    ts = pd.Timestamp(d)
    cal = _calendar()
    if not cal.is_session(ts):
        raise ValueError(f"trade_date {d.isoformat()} is not an XNYS exchange session")
    cur = ts
    for _ in range(settlement_lag(d)):
        cur = cal.next_session(cur)
    return pd.Timestamp(cur.date())


def is_settled(trade_date, check_date) -> bool:
    """True if a trade executed on trade_date is settled by check_date."""
    settle = settlement_date(trade_date)
    check = pd.Timestamp(_to_date(check_date))
    return check >= settle
