"""Alpaca Paper Trading venue adapter.

SAFETY CONTRACT:
  - Connects ONLY to paper-api.alpaca.markets (hardcoded paper=True).
  - Live endpoint is structurally unreachable.
  - API keys read from env vars ALPACA_API_KEY and ALPACA_SECRET_KEY.
    NEVER stored in code, config, or logs.
  - The Risk Kernel retains FINAL VETO. This adapter is a dumb executor.
  - Prices converted to integer cents internally (matches ledger).

REQUIREMENTS: pip install alpaca-py

ENVIRONMENT (Colab Secrets or shell):
  ALPACA_API_KEY=<paper key ID>
  ALPACA_SECRET_KEY=<paper secret>
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..core import Money, Shares, ceil_bps
from ..venue import ExecEvent, VenueOrder


def _require_paper_env() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
            "Use Colab Secrets. NEVER commit keys to code."
        )
    return key, secret


@dataclass
class AlpacaPaperVenue:
    """Drop-in replacement for MockVenue. Same interface, real paper broker."""
    orders: dict[str, VenueOrder] = field(default_factory=dict)
    positions: dict[str, Shares] = field(default_factory=dict)
    cash: Money = 0
    outbox: list[ExecEvent] = field(default_factory=list)
    _eid: int = 0
    present: bool = True
    snapshot_available: bool = True
    _seq: int = 0
    _api: object = field(default=None, repr=False)
    _initialized: bool = False

    def _ensure_api(self) -> None:
        if self._initialized:
            return
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")
        key, secret = _require_paper_env()
        # HARDCODED paper=True. No config can change this.
        self._api = TradingClient(key, secret, paper=True)
        self._initialized = True
        self._sync_positions()
        self._sync_cash()

    def _sync_positions(self) -> None:
        self._ensure_api()
        try:
            positions = self._api.get_all_positions()
            self.positions = {}
            for p in positions:
                qty = int(float(p.qty))
                if qty != 0:
                    self.positions[p.symbol] = qty
        except Exception:
            pass

    def _sync_cash(self) -> None:
        self._ensure_api()
        try:
            account = self._api.get_account()
            self.cash = int(round(float(account.cash) * 100))
        except Exception:
            pass

    def _next_eid(self) -> str:
        self._eid += 1
        return f"E{self._eid:06d}"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, order_id: str, kind: str, cum: Shares, px: Money) -> ExecEvent:
        ev = ExecEvent(self._next_eid(), order_id, kind, cum, px, self._next_seq())
        self.outbox.append(ev)
        return ev

    def send(self, order_id: str, symbol: str, side: str, qty: Shares,
             price: Money, cost_bps: int = 0) -> None:
        if not self.present:
            raise ConnectionError("venue unavailable")
        self._ensure_api()
        old = self.orders.get(order_id)
        if old:
            same = (old.symbol, old.side, old.qty, old.price, old.cost_bps) == (
                symbol, side, qty, price, cost_bps)
            if not same:
                raise ValueError("client order ID reused with different terms")
            return
        if side not in {"BUY", "SELL"} or qty <= 0 or price <= 0:
            raise ValueError("invalid order")
        limit_price = round(price / 100.0, 2)
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            alpaca_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
            req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=alpaca_side,
                type="limit", limit_price=limit_price,
                time_in_force=TimeInForce.DAY, client_order_id=order_id,
            )
            self._api.submit_order(req)
        except Exception as e:
            err = str(e).lower()
            if "insufficient" in err or "buying power" in err:
                vo = VenueOrder(order_id, symbol, side, qty, 0, price,
                                "REJECTED", 0, 0, cost_bps)
                self.orders[order_id] = vo
                self._emit(order_id, "REJECT", 0, price)
                return
            raise
        vo = VenueOrder(order_id, symbol, side, qty, 0, price,
                        "ACKNOWLEDGED", 0, 0, cost_bps)
        self.orders[order_id] = vo
        self._emit(order_id, "ACK", 0, price)

    def poll_fills(self) -> list[ExecEvent]:
        """Check Alpaca for new fills. Call each session step."""
        self._ensure_api()
        events = []
        try:
            for oid, vo in list(self.orders.items()):
                if vo.status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                    continue
                try:
                    ao = self._api.get_order_by_client_order_id(oid)
                except Exception:
                    continue
                filled_qty = int(float(ao.filled_qty or 0))
                if filled_qty > vo.cum_qty:
                    delta = filled_qty - vo.cum_qty
                    avg = float(ao.filled_avg_price or vo.price / 100.0)
                    px_cents = int(round(avg * 100))
                    vo.cum_qty = filled_qty
                    vo.filled_notional += delta * px_cents
                    vo.filled_cost += ceil_bps(delta * px_cents, vo.cost_bps)
                    vo.status = "FILLED" if filled_qty == vo.qty else "PARTIALLY_FILLED"
                    signed = delta if vo.side == "BUY" else -delta
                    self.positions[vo.symbol] = self.positions.get(vo.symbol, 0) + signed
                    self.cash += (-signed * px_cents) - ceil_bps(delta * px_cents, vo.cost_bps)
                    events.append(self._emit(oid, "FILL", filled_qty, px_cents))
                status = str(ao.status).lower()
                if status == "cancelled" and vo.status not in {"CANCELLED", "FILLED"}:
                    vo.status = "CANCELLED"
                    events.append(self._emit(oid, "CANCEL_ACK", vo.cum_qty, vo.price))
                elif status == "rejected" and vo.status not in {"REJECTED", "FILLED"}:
                    vo.status = "REJECTED"
                    events.append(self._emit(oid, "REJECT", 0, vo.price))
                elif status == "expired" and vo.status not in {"EXPIRED", "FILLED"}:
                    vo.status = "EXPIRED"
                    events.append(self._emit(oid, "TIMEOUT", vo.cum_qty, vo.price))
        except Exception:
            self.present = False
        return events

    def fill(self, order_id: str, q: Shares, price: Money | None = None,
             dup: bool = False, stale_cum: int | None = None) -> ExecEvent:
        """Local fill for testing. Real fills come from poll_fills()."""
        o = self.orders[order_id]
        px = o.price if price is None else price
        if q <= 0 or px <= 0 or o.status in {"CANCELLED", "REJECTED"} or o.cum_qty + q > o.qty:
            raise ValueError("invalid fill")
        o.cum_qty += q
        o.filled_notional += q * px
        fc = ceil_bps(q * px, o.cost_bps)
        o.filled_cost += fc
        o.status = "FILLED" if o.cum_qty == o.qty else "PARTIALLY_FILLED"
        signed = q if o.side == "BUY" else -q
        self.positions[o.symbol] = self.positions.get(o.symbol, 0) + signed
        self.cash += (-signed * px) - fc
        cum = o.cum_qty if stale_cum is None else stale_cum
        ev = self._emit(order_id, "FILL", cum, px)
        if dup:
            self.outbox.append(ev)
        return ev

    def cancel_ack(self, order_id: str) -> ExecEvent:
        self._ensure_api()
        o = self.orders[order_id]
        try:
            self._api.cancel_order_by_client_order_id(order_id)
        except Exception:
            pass
        if o.status not in {"FILLED", "REJECTED", "EXPIRED"}:
            o.status = "CANCELLED"
        return self._emit(order_id, "CANCEL_ACK", o.cum_qty, o.price)

    def cancel_reject(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        return self._emit(order_id, "CANCEL_REJECT", o.cum_qty, o.price)

    def reject(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        if o.cum_qty:
            raise ValueError("cannot reject an already-filled order")
        o.status = "REJECTED"
        return self._emit(order_id, "REJECT", 0, o.price)

    def timeout(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        if o.status not in {"FILLED", "CANCELLED", "REJECTED"}:
            o.status = "UNKNOWN"
        return self._emit(order_id, "TIMEOUT", o.cum_qty, o.price)

    def snapshot(self) -> dict | None:
        if not self.present or not self.snapshot_available:
            return None
        try:
            self._sync_positions()
            self._sync_cash()
            self._ensure_api()
            for oid, vo in list(self.orders.items()):
                if vo.status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                    continue
                try:
                    ao = self._api.get_order_by_client_order_id(oid)
                    fq = int(float(ao.filled_qty or 0))
                    st = str(ao.status).lower()
                    if st == "filled": vo.status, vo.cum_qty = "FILLED", fq
                    elif st == "cancelled": vo.status = "CANCELLED"
                    elif st == "rejected": vo.status = "REJECTED"
                    elif st == "expired": vo.status = "EXPIRED"
                    elif st == "partially_filled": vo.status, vo.cum_qty = "PARTIALLY_FILLED", fq
                except Exception:
                    pass
        except Exception:
            self.present = False
            return None
        return {
            "complete": True, "cash_is_t0": True,
            "orders": {k: {"cum_qty": v.cum_qty, "status": v.status, "qty": v.qty,
                           "symbol": v.symbol, "side": v.side, "price": v.price,
                           "filled_notional": v.filled_notional, "filled_cost": v.filled_cost}
                       for k, v in self.orders.items()},
            "positions": dict(self.positions), "cash": self.cash,
        }

    def diverge(self, symbol: str, delta: Shares) -> None:
        raise NotImplementedError("diverge() is a MockVenue test helper. Not available on real venues.")
