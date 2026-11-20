"""Deterministic mock venue with fault injection. No network, no real broker."""
from __future__ import annotations
from dataclasses import dataclass, field
from .core import Money, Shares


@dataclass
class ExecEvent:
    exec_id: str
    order_id: str
    kind: str            # ACK|FILL|CANCEL_ACK|CANCEL_REJECT|REJECT
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


@dataclass
class MockVenue:
    orders: dict[str, VenueOrder] = field(default_factory=dict)
    positions: dict[str, Shares] = field(default_factory=dict)
    cash: Money = 0
    outbox: list[ExecEvent] = field(default_factory=list)
    _eid: int = 0
    present: bool = True          # False simulates "confirmed absent"
    snapshot_available: bool = True

    def _next_eid(self) -> str:
        self._eid += 1
        return f"E{self._eid:06d}"

    def send(self, order_id: str, symbol: str, side: str, qty: Shares, price: Money) -> None:
        self.orders[order_id] = VenueOrder(order_id, symbol, side, qty, 0, price)
        self.outbox.append(ExecEvent(self._next_eid(), order_id, "ACK", 0, price))

    def fill(self, order_id: str, q: Shares, dup: bool = False,
             stale_cum: int | None = None) -> ExecEvent:
        o = self.orders[order_id]
        o.cum_qty = min(o.qty, o.cum_qty + q)
        o.status = "FILLED" if o.cum_qty == o.qty else "PARTIALLY_FILLED"
        delta = q if o.side == "BUY" else -q
        self.positions[o.symbol] = self.positions.get(o.symbol, 0) + delta
        self.cash += (-delta * o.price)
        eid = self._next_eid()
        cum = o.cum_qty if stale_cum is None else stale_cum
        ev = ExecEvent(eid, order_id, "FILL", cum, o.price)
        self.outbox.append(ev)
        if dup:
            self.outbox.append(ExecEvent(eid, order_id, "FILL", cum, o.price))
        return ev

    def cancel_ack(self, order_id: str) -> ExecEvent:
        self.orders[order_id].status = "CANCELLED"
        ev = ExecEvent(self._next_eid(), order_id, "CANCEL_ACK",
                       self.orders[order_id].cum_qty, self.orders[order_id].price)
        self.outbox.append(ev); return ev

    def cancel_reject(self, order_id: str) -> ExecEvent:
        ev = ExecEvent(self._next_eid(), order_id, "CANCEL_REJECT",
                       self.orders[order_id].cum_qty, self.orders[order_id].price)
        self.outbox.append(ev); return ev

    def reject(self, order_id: str) -> ExecEvent:
        self.orders[order_id].status = "REJECTED"
        ev = ExecEvent(self._next_eid(), order_id, "REJECT", 0, self.orders[order_id].price)
        self.outbox.append(ev); return ev

    def snapshot(self) -> dict | None:
        if not self.snapshot_available:
            return None
        return {
            "orders": {k: {"cum_qty": v.cum_qty, "status": v.status, "qty": v.qty,
                           "symbol": v.symbol, "side": v.side, "price": v.price}
                       for k, v in self.orders.items()},
            "positions": dict(self.positions), "cash": self.cash,
        }

    def diverge(self, symbol: str, delta: Shares) -> None:
        self.positions[symbol] = self.positions.get(symbol, 0) + delta
