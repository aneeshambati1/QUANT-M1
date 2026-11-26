"""Core SettlementBook: duplicate/conflict integrity and integer minor units."""
from __future__ import annotations

import pytest

from quant_m1.settlement import ExchangeCalendar, PendingSettlement, SettlementBook


def _book():
    return SettlementBook(ExchangeCalendar("XNYS"))


def test_duplicate_pending_identical_is_idempotent():
    p = PendingSettlement("e1", "2024-05-28", "2024-05-29", 1000)
    assert SettlementBook(pending=(p, p)).unsettled_total() == 1000


def test_duplicate_pending_conflict_raises():
    p1 = PendingSettlement("e1", "2024-05-28", "2024-05-29", 1000)
    p2 = PendingSettlement("e1", "2024-05-28", "2024-05-29", 2000)
    with pytest.raises(ValueError):
        SettlementBook(pending=(p1, p2))


def test_on_sell_fill_float_proceeds_rejected():
    with pytest.raises(ValueError):
        _book().on_sell_fill("e1", "2024-05-28", 1000.5)


def test_on_sell_fill_bool_proceeds_rejected():
    with pytest.raises(ValueError):
        _book().on_sell_fill("e1", "2024-05-28", True)


def test_pending_float_amount_rejected():
    with pytest.raises(ValueError):
        PendingSettlement("e1", "2024-05-28", "2024-05-29", 1000.5)


def test_settlement_releases_on_available_date():
    b = _book()
    b.on_sell_fill("e1", "2024-05-28", 1000)
    assert b.settle("2024-05-28") == 0
    assert b.settle("2024-05-29") == 1000
    assert b.settle("2024-05-29") == 0
