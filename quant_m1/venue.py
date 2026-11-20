"""Deterministic M1.1 mock venue. No network or real broker access."""
from __future__ import annotations
from dataclasses import dataclass, field
from .core import Money, Shares, ceil_bps


@dataclass(frozen=True)
class ExecEvent:
    exec_id: str
    order_id: str
    kind: str
    cum_qty: Shares = 0
    price: Money = 0
    seq: int = 0


@dataclass
class VenueOrder:
    order_id: str
    symbol: str
    side: str
    qty: Shares
    cum_qty: Shares = 0
    price: Money = 0
    status: str = "ACKNOWLEDGED"
    filled_notional: Money = 0
    filled_cost: Money = 0
    cost_bps: int = 0

    @property
    def average_fill_price(self) -> int:
        return self.filled_notional // self.cum_qty if self.cum_qty else 0


@dataclass
class MockVenue:
    orders: dict[str, VenueOrder] = field(default_factory=dict)
    positions: dict[str, Shares] = field(default_factory=dict)
    cash: Money = 0
    outbox: list[ExecEvent] = field(default_factory=list)
    _eid: int = 0
    present: bool = True
    snapshot_available: bool = True

    def _next_eid(self) -> str:
        self._eid += 1
        return f"E{self._eid:06d}"

    def send(self, order_id: str, symbol: str, side: str, qty: Shares,
             price: Money, cost_bps: int = 0) -> None:
        """Idempotent by client order ID."""
        if not self.present:
            raise ConnectionError("venue unavailable")
        old = self.orders.get(order_id)
        if old:
            same = (old.symbol, old.side, old.qty, old.price, old.cost_bps) == (
                symbol, side, qty, price, cost_bps)
            if not same:
                raise ValueError("client order ID reused with different terms")
            return
        if side not in {"BUY", "SELL"} or qty <= 0 or price <= 0:
            raise ValueError("invalid order")
        self.orders[order_id] = VenueOrder(
            order_id, symbol, side, qty, 0, price, "ACKNOWLEDGED", 0, 0, cost_bps
        )
        self.outbox.append(ExecEvent(self._next_eid(), order_id, "ACK", 0, price))

    def fill(self, order_id: str, q: Shares, price: Money | None = None,
             dup: bool = False, stale_cum: int | None = None) -> ExecEvent:
        o = self.orders[order_id]
        px = o.price if price is None else price
        if q <= 0 or px <= 0 or o.status in {"CANCELLED", "REJECTED"} or o.cum_qty + q > o.qty:
            raise ValueError("invalid fill")
        o.cum_qty += q
        o.filled_notional += q * px
        fill_cost = ceil_bps(q * px, o.cost_bps)
        o.filled_cost += fill_cost
        o.status = "FILLED" if o.cum_qty == o.qty else "PARTIALLY_FILLED"
        signed = q if o.side == "BUY" else -q
        self.positions[o.symbol] = self.positions.get(o.symbol, 0) + signed
        self.cash += (-signed * px) - fill_cost
        eid = self._next_eid()
        cum = o.cum_qty if stale_cum is None else stale_cum
        ev = ExecEvent(eid, order_id, "FILL", cum, px)
        self.outbox.append(ev)
        if dup:
            self.outbox.append(ev)
        return ev

    def cancel_ack(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        if o.status != "FILLED":
            o.status = "CANCELLED"
        ev = ExecEvent(self._next_eid(), order_id, "CANCEL_ACK", o.cum_qty, o.price)
        self.outbox.append(ev)
        return ev

    def cancel_reject(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        ev = ExecEvent(self._next_eid(), order_id, "CANCEL_REJECT", o.cum_qty, o.price)
        self.outbox.append(ev)
        return ev

    def reject(self, order_id: str) -> ExecEvent:
        o = self.orders[order_id]
        if o.cum_qty:
            raise ValueError("cannot reject an already-filled order")
        o.status = "REJECTED"
        ev = ExecEvent(self._next_eid(), order_id, "REJECT", 0, o.price)
        self.outbox.append(ev)
        return ev

    def snapshot(self) -> dict | None:
        if not self.present or not self.snapshot_available:
            return None
        return {
            "complete": True,
            "orders": {
                k: {
                    "cum_qty": v.cum_qty,
                    "status": v.status,
                    "qty": v.qty,
                    "symbol": v.symbol,
                    "side": v.side,
                    "price": v.price,
                    "filled_notional": v.filled_notional,
                    "filled_cost": v.filled_cost,
                }
                for k, v in self.orders.items()
            },
            "positions": dict(self.positions),
            "cash": self.cash,
        }

    def diverge(self, symbol: str, delta: Shares) -> None:
        self.positions[symbol] = self.positions.get(symbol, 0) + delta