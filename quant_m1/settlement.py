"""
T+1 settlement book for QUANT-M1.

This module is intentionally independent of strategy logic.
It enforces:
  - sell proceeds are unsettled until the next exchange session
  - same-day settlement is impossible
  - duplicate exec_id cannot create duplicate settlement
  - settlement is idempotent
  - amounts are integer minor units to avoid float cash drift

US securities T+1 effective May 28, 2024. Historical tests before that date
must explicitly model T+2 if claiming historical fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class PendingSettlement:
    exec_id: str
    trade_date: str          # ISO date, must be an exchange session
    available_date: str      # ISO date, strictly after trade_date
    amount_minor: int        # integer cents / minor units, net of sell costs
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.exec_id:
            raise ValueError("exec_id must be non-empty")
        if isinstance(self.amount_minor, bool) or not isinstance(self.amount_minor, int):
            raise ValueError("amount_minor must be an integer (minor units)")
        if self.amount_minor <= 0:
            raise ValueError("settlement amount must be positive")
        td = date.fromisoformat(self.trade_date)
        ad = date.fromisoformat(self.available_date)
        if ad <= td:
            raise ValueError("available_date must be strictly after trade_date")
        if self.currency != "USD":
            raise ValueError("M1 mandate is USD base")


class ExchangeCalendar:
    """Exchange session calendar. Default XNYS."""

    def __init__(self, name: str = "XNYS") -> None:
        self._cal = xcals.get_calendar(name)

    def is_session(self, d: str) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(d)))

    def next_session(self, d: str) -> str:
        return self._cal.next_session(pd.Timestamp(d)).date().isoformat()

    def previous_session(self, d: str) -> str:
        return self._cal.previous_session(pd.Timestamp(d)).date().isoformat()


class SettlementBook:
    """
    Append-only settlement book.

    Lifecycle:
      on_sell_fill(...) -> pending
      settle(as_of)     -> released when available_date <= as_of

    Duplicate exec_id with identical payload is idempotent.
    Duplicate exec_id with different payload raises.
    """

    def __init__(
        self,
        calendar: ExchangeCalendar | None = None,
        pending: Tuple[PendingSettlement, ...] = (),
    ) -> None:
        self.calendar = calendar or ExchangeCalendar("XNYS")
        self._seen: Dict[str, PendingSettlement] = {}
        self._pending: Dict[str, PendingSettlement] = {}
        self._released: Dict[str, PendingSettlement] = {}

        for p in pending:
            if not isinstance(p, PendingSettlement):
                raise ValueError("pending entries must be PendingSettlement")
            if p.exec_id in self._seen:
                if self._seen[p.exec_id] != p:
                    raise ValueError(f"SETTLEMENT_EXEC_ID_CONFLICT: {p.exec_id}")
                continue
            self._seen[p.exec_id] = p
            self._pending[p.exec_id] = p

    def on_sell_fill(
        self,
        exec_id: str,
        trade_date: str,
        net_proceeds_minor: int,
        currency: str = "USD",
    ) -> PendingSettlement:
        if isinstance(net_proceeds_minor, bool) or not isinstance(net_proceeds_minor, int):
            raise ValueError("net_proceeds_minor must be an integer (minor units)")
        if not self.calendar.is_session(trade_date):
            raise ValueError(f"trade_date {trade_date} is not an exchange session")

        available = self.calendar.next_session(trade_date)
        p = PendingSettlement(
            exec_id=exec_id,
            trade_date=trade_date,
            available_date=available,
            amount_minor=net_proceeds_minor,
            currency=currency,
        )

        if exec_id in self._seen:
            if self._seen[exec_id] != p:
                raise ValueError(f"SETTLEMENT_EXEC_ID_CONFLICT: {exec_id}")
            return p  # idempotent

        self._seen[exec_id] = p
        self._pending[exec_id] = p
        return p

    def settle(self, as_of_date: str) -> int:
        """Release matured settlements. Returns total minor units released."""
        as_of = date.fromisoformat(as_of_date)
        total = 0
        due = [
            p for p in self._pending.values()
            if date.fromisoformat(p.available_date) <= as_of
        ]
        for p in due:
            total += p.amount_minor
            del self._pending[p.exec_id]
            self._released[p.exec_id] = p
        return total

    def unsettled_total(self) -> int:
        return sum(p.amount_minor for p in self._pending.values())

    def snapshot(self) -> Tuple[PendingSettlement, ...]:
        return tuple(sorted(self._pending.values(), key=lambda p: (p.available_date, p.exec_id)))

    def released_snapshot(self) -> Tuple[PendingSettlement, ...]:
        return tuple(sorted(self._released.values(), key=lambda p: (p.available_date, p.exec_id)))
