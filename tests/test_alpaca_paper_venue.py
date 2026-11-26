"""AlpacaPaperVenue tests. All use mocked SDK — no real API calls."""
from __future__ import annotations
import os, sys, types
from unittest.mock import MagicMock, patch
import pytest

# Fake alpaca module hierarchy BEFORE importing the venue
_a = types.ModuleType("alpaca")
_at = types.ModuleType("alpaca.trading")
_atc = types.ModuleType("alpaca.trading.client")
_atr = types.ModuleType("alpaca.trading.requests")
_ate = types.ModuleType("alpaca.trading.enums")
_ac = types.ModuleType("alpaca.common")
_ace = types.ModuleType("alpaca.common.exceptions")

class _TC:
    def __init__(self, *a, **kw): pass
class _LOR:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
class _OS:
    BUY = "buy"; SELL = "sell"
class _TIF:
    DAY = "day"
class _AE(Exception): pass

_atc.TradingClient = _TC
_atr.LimitOrderRequest = _LOR
_atr.GetAssetsRequest = MagicMock
_ate.OrderSide = _OS
_ate.TimeInForce = _TIF
_ace.APIError = _AE

for name, mod in [("alpaca",_a),("alpaca.trading",_at),("alpaca.trading.client",_atc),
                  ("alpaca.trading.requests",_atr),("alpaca.trading.enums",_ate),
                  ("alpaca.common",_ac),("alpaca.common.exceptions",_ace)]:
    sys.modules[name] = mod

from quant_m1.venue_adapters.alpaca_paper import AlpacaPaperVenue, _require_paper_env

@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PK_TEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK_TEST")

@pytest.fixture
def venue():
    v = AlpacaPaperVenue()
    v._initialized = True
    v._api = MagicMock()
    return v

def test_missing_env_raises():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ALPACA_API_KEY", None)
        os.environ.pop("ALPACA_SECRET_KEY", None)
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            _require_paper_env()

def test_diverge_raises(venue):
    with pytest.raises(NotImplementedError):
        venue.diverge("SPY", 100)

def test_hardcoded_paper_endpoint():
    import inspect
    assert "paper=True" in inspect.getsource(AlpacaPaperVenue._ensure_api)

def test_send_ack(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00, cost_bps=5)
    assert venue.orders["O1"].status == "ACKNOWLEDGED"
    assert venue.outbox[0].kind == "ACK"

def test_send_dup_idempotent(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    assert len(venue.outbox) == 1

def test_send_dup_diff_raises(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    with pytest.raises(ValueError):
        venue.send("O1", "SPY", "SELL", 10, 500_00)

def test_fill_updates(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00, cost_bps=5)
    ev = venue.fill("O1", 10, price=500_00)
    assert ev.kind == "FILL"
    assert venue.positions.get("SPY") == 10
    assert venue.cash < 0

def test_partial_fill(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.fill("O1", 5, price=500_00)
    assert venue.orders["O1"].status == "PARTIALLY_FILLED"
    venue.fill("O1", 5, price=500_00)
    assert venue.orders["O1"].status == "FILLED"

def test_sell_reduces(venue):
    venue.positions["SPY"] = 10
    venue.send("O1", "SPY", "SELL", 5, 500_00)
    venue.fill("O1", 5, price=500_00)
    assert venue.positions["SPY"] == 5

def test_snapshot_structure(venue):
    s = venue.snapshot()
    assert s["complete"] is True and "orders" in s and "cash" in s

def test_snapshot_none_when_absent(venue):
    venue.present = False
    assert venue.snapshot() is None

def test_cancel(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.cancel_ack("O1")
    assert venue.orders["O1"].status == "CANCELLED"

def test_reject(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.reject("O1")
    assert venue.orders["O1"].status == "REJECTED"

def test_timeout(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.timeout("O1")
    assert venue.orders["O1"].status == "UNKNOWN"

def test_invalid_fill(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    with pytest.raises(ValueError):
        venue.fill("O1", 0, price=500_00)

def test_fill_after_cancel(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    venue.cancel_ack("O1")
    with pytest.raises(ValueError):
        venue.fill("O1", 10, price=500_00)

def test_poll_discovers_fill(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00, cost_bps=5)
    m = MagicMock(); m.filled_qty = "10"; m.filled_avg_price = "500.00"; m.status = "filled"
    venue._api.get_order_by_client_order_id.return_value = m
    evs = venue.poll_fills()
    assert len(evs) == 1 and evs[0].kind == "FILL"
    assert venue.orders["O1"].status == "FILLED"

def test_poll_no_change(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    m = MagicMock(); m.filled_qty = "0"; m.status = "new"
    venue._api.get_order_by_client_order_id.return_value = m
    assert venue.poll_fills() == []

def test_poll_cancel(venue):
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    m = MagicMock(); m.filled_qty = "0"; m.status = "cancelled"
    venue._api.get_order_by_client_order_id.return_value = m
    assert any(e.kind == "CANCEL_ACK" for e in venue.poll_fills())

def test_insufficient_power_rejects(venue):
    venue._api.submit_order.side_effect = Exception("insufficient buying power")
    venue.send("O1", "SPY", "BUY", 10, 500_00)
    assert venue.orders["O1"].status == "REJECTED"
