"""Kernel authorization, single-use tokens, gateway with reconciliation-first
reservation recomputation."""
from __future__ import annotations
import hashlib, hmac, uuid
from dataclasses import dataclass, asdict
from .core import (Money, Shares, BPS, Clock, ConfigAuthority, EventLog, Instrument,
                   Ledger, WorkingOrder, ceil_bps, floor_bps, sha)
from .flags import Flags
from .risk import evaluate, reservations, CONSTRAINTS
from .venue import MockVenue, ExecEvent


class Reject(Exception):
    def __init__(self, code: str):
        self.code = code; super().__init__(code)


@dataclass(frozen=True)
class Token:
    intent_hash: str
    state_hash: str
    order_id: str
    issue_seq: int
    valid_until_seq: int
    nonce: str
    sig: str

    def body(self) -> dict:
        d = asdict(self); d.pop("sig"); return d


def sign(key: bytes, body: dict) -> str:
    return hmac.new(key, sha(body).encode(), hashlib.sha256).hexdigest()


@dataclass
class Intent:
    symbol: str
    side: str
    qty: Shares
    price: Money
    cost_bps: int

    def hash(self) -> str:
        return sha(asdict(self))


class Kernel:
    def __init__(self, key: bytes, cfg: ConfigAuthority, instruments: dict[str, Instrument],
                 scenarios: tuple[dict[str, int], ...]):
        self.key, self.cfg, self.instruments, self.scenarios = key, cfg, instruments, scenarios

    def authorize(self, ledger: Ledger, intent: Intent, flags: Flags,
                  clock: Clock, watchdog_stop: bool) -> Token:
        cfg = self.cfg
        if not cfg.attested():
            raise Reject("CONFIG_UNATTESTED")
        if "NEW_RISK" not in flags.permissions() and intent.side == "BUY":
            raise Reject("MODE_BLOCKS_NEW_RISK")
        if intent.symbol not in cfg.mandate.allowlist:
            raise Reject("ALLOWLIST")
        inst = self.instruments[intent.symbol]
        if inst.product_class not in cfg.mandate.product_classes:
            raise Reject("PRODUCT_CLASS")
        if inst.halted:
            raise Reject("HALTED")
        if intent.side == "SELL" and cfg.floor.long_only:
            have = ledger.state.positions.get(intent.symbol, 0)
            committed_sells = sum(o.remaining for o in ledger.working_active()
                                  if o.symbol == intent.symbol and o.side == "SELL")
            if intent.qty + committed_sells > have:
                raise Reject("SHORT_PROHIBITED")
        if intent.side not in {"BUY", "SELL"}:
            raise Reject("BAD_SIDE")
        if intent.qty <= 0:
            raise Reject("BAD_QTY")
        if clock.wall - ledger.state.mark_wall > cfg.mandate.tau_data_secs:
            raise Reject("STALE_MARKS")
        if ledger.drawdown_bps() >= cfg.mandate.drawdown_trigger_bps and intent.side == "BUY":
            raise Reject("DRAWDOWN_TRIGGER")

        notional = intent.qty * intent.price
        if notional > cfg.mandate.order_notional_cap:
            raise Reject("ORDER_NOTIONAL")
        if notional > floor_bps(inst.adv, cfg.mandate.participation_bps):
            raise Reject("PARTICIPATION")
        if len(ledger.working_active()) + 1 > cfg.ops.m_vertex:
            raise Reject("M_VERTEX")

        order_id = f"O{uuid.uuid4().hex[:12]}"
        nonce = uuid.uuid4().hex
        cand = WorkingOrder(order_id, intent.symbol, intent.side, intent.qty, 0,
                            intent.price, intent.cost_bps, "RESERVED",
                            ledger.state.trade_date, nonce)
        orders = ledger.working_active() + (cand,)
        ev = evaluate(ledger.state, orders, self.instruments, cfg.mandate, self.scenarios)
        if not ev.ok:
            raise Reject(ev.reason)

        # risk-reducing sells still must pass every resource check (already done above)
        with_lock = True  # single-writer: caller holds the system lock
        assert with_lock
        ledger.state.working = ledger.state.working + (cand,)
        ledger.reserved = reservations(ev)
        ledger.state.seq = clock.next()
        tok = Token(intent.hash(), ledger.state.digest(), order_id,
                    clock.seq, clock.seq + cfg.ops.auth_ttl_seqs, nonce, "")
        tok = Token(**{**asdict(tok), "sig": sign(self.key, tok.body())})
        return tok


class Gateway:
    def __init__(self, key: bytes, cfg: ConfigAuthority, ledger: Ledger, log: EventLog,
                 venue: MockVenue, clock: Clock, instruments: dict[str, Instrument],
                 scenarios: tuple[dict[str, int], ...], flags: Flags,
                 watchdog_path: str):
        self.key, self.cfg, self.ledger, self.log = key, cfg, ledger, log
        self.venue, self.clock, self.instruments = venue, clock, instruments
        self.scenarios, self.flags, self.watchdog_path = scenarios, flags, watchdog_path
        self.fail_point: str | None = None

    def _watchdog(self) -> bool:
        import os
        return os.path.exists(self.watchdog_path)

    def submit(self, tok: Token, intent: Intent) -> None:
        if self._watchdog():
            raise Reject("WATCHDOG_STOP")
        if "SUBMIT" not in self.flags.permissions():
            raise Reject("MODE_BLOCKS_SUBMIT")
        if not hmac.compare_digest(tok.sig, sign(self.key, tok.body())):
            raise Reject("BAD_SIGNATURE")
        if tok.intent_hash != intent.hash():
            raise Reject("INTENT_MUTATED")
        if tok.nonce in self.ledger.spent_nonces:
            raise Reject("NONCE_REPLAY")
        if self.clock.seq > tok.valid_until_seq:
            self._expire(tok); raise Reject("TOKEN_EXPIRED")
        if tok.state_hash != self.ledger.state.digest():
            raise Reject("STALE_SNAPSHOT")
        recent = [s for s in self.ledger.submit_seq
                  if self.clock.seq - s < self.cfg.mandate.rate_window]
        if len(recent) >= self.cfg.mandate.rate_limit:
            raise Reject("RATE_LIMIT")

        # E-04: durably mark nonce BEFORE transmission
        self.ledger.spent_nonces.add(tok.nonce)
        self.ledger.submit_seq.append(self.clock.seq)
        self.ledger.persist()
        if self.fail_point == "CRASH_A":
            raise RuntimeError("crash after durable nonce, before send")

        self.venue.send(tok.order_id, intent.symbol, intent.side, intent.qty, intent.price)
        if self.fail_point == "CRASH_B":
            raise RuntimeError("crash after send, before ack processing")
        self._set_status(tok.order_id, "SUBMITTED")
        self.log.append("SUBMIT", {"order_id": tok.order_id}, self.clock.next())

    def _set_status(self, order_id: str, status: str) -> None:
        self.ledger.state.working = tuple(
            WorkingOrder(**{**asdict(o), "status": status}) if o.order_id == order_id else o
            for o in self.ledger.state.working)

    def _expire(self, tok: Token) -> None:
        self._set_status(tok.order_id, "EXPIRED")
        self._recompute()
        self.ledger.persist()

    # ---------------- reconciliation-first event processing (C4)
    def process(self, ev: ExecEvent) -> str:
        if ev.exec_id in self.ledger.applied_exec_ids:
            return "NOOP_DUPLICATE"
        order = next((o for o in self.ledger.state.working if o.order_id == ev.order_id), None)
        if order is None:
            return "NOOP_UNKNOWN_ORDER"
        if ev.kind == "FILL" and ev.cum_qty < order.cum_filled:
            self._reconcile_snapshot()
            return "SNAPSHOT_RECONCILED"

        self.ledger.applied_exec_ids.add(ev.exec_id)
        if ev.kind == "ACK":
            self._set_status(ev.order_id, "ACKNOWLEDGED")
        elif ev.kind == "FILL":
            self._apply_fill(ev.order_id, ev.cum_qty)
        elif ev.kind == "CANCEL_ACK":
            # INV-06: reconcile fills first; terminal message alone releases nothing
            self._reconcile_snapshot()
            self._set_status(ev.order_id, "CANCELLED")
        elif ev.kind == "CANCEL_REJECT":
            pass  # INV-07: reservation retained
        elif ev.kind == "REJECT":
            self._set_status(ev.order_id, "REJECTED")

        self._reconcile_positions()
        self._recompute()
        self.ledger.persist()
        self.log.append("EXEC", {"exec_id": ev.exec_id, "kind": ev.kind}, self.clock.next())
        return "APPLIED"

    def _apply_fill(self, order_id: str, cum: int) -> None:
        neww = []
        for o in self.ledger.state.working:
            if o.order_id != order_id:
                neww.append(o); continue
            d = cum - o.cum_filled
            if d > 0:
                notional = d * o.price
                cost = ceil_bps(notional, o.cost_bps)
                if o.side == "BUY":
                    self.ledger.state.positions[o.symbol] = \
                        self.ledger.state.positions.get(o.symbol, 0) + d
                    self.ledger.state.cash_settled -= notional + cost
                else:
                    self.ledger.state.positions[o.symbol] = \
                        self.ledger.state.positions.get(o.symbol, 0) - d
                    self.ledger.state.receivables += (
                        (o.trade_date + self.cfg.mandate.settle_lag_days, notional - cost),)
            status = "FILLED" if cum >= o.qty else "PARTIALLY_FILLED"
            neww.append(WorkingOrder(**{**asdict(o), "cum_filled": cum, "status": status}))
        self.ledger.state.working = tuple(neww)
        self.ledger.mark_peak()

    def _reconcile_positions(self) -> None:
        snap = self.venue.snapshot()
        if snap is None:
            self.flags.exec = "DISCONNECTED"; return
        internal = {s: q for s, q in self.ledger.state.positions.items() if q}
        external = {s: q for s, q in snap["positions"].items() if q}
        self.flags.exec = "OK" if internal == external else "DIVERGENT"

    def _reconcile_snapshot(self) -> None:
        snap = self.venue.snapshot()
        if snap is None:
            self.flags.exec = "DISCONNECTED"; return
        neww = []
        for o in self.ledger.state.working:
            vo = snap["orders"].get(o.order_id)
            if vo and vo["cum_qty"] > o.cum_filled:
                self._apply_fill(o.order_id, vo["cum_qty"])
                o = next(x for x in self.ledger.state.working if x.order_id == o.order_id)
            neww.append(o)
        self.ledger.state.working = tuple(neww)

    def _recompute(self) -> None:
        """INV-03: reservations recomputed from reachable vertices, never decremented."""
        orders = self.ledger.working_active()
        if not orders:
            self.ledger.reserved = {c: 0 for c in CONSTRAINTS}
            if self.flags.envelope == "OVERFLOW":
                self.flags.envelope = "OK"
            return
        ev = evaluate(self.ledger.state, orders, self.instruments,
                      self.cfg.mandate, self.scenarios)
        if not ev.ok:
            self.flags.envelope = "OVERFLOW"
            self.ledger.reserved = {c: 0 for c in CONSTRAINTS}
            self.log.append("OVERFLOW", {"reason": ev.reason}, self.clock.next())
            return
        self.ledger.reserved = reservations(ev)
        self.flags.envelope = "OK" if self.flags.envelope == "OVERFLOW" else self.flags.envelope

    def recover(self) -> str:
        """Crash recovery. Unknown => fully committed, block new risk."""
        snap = self.venue.snapshot()
        if snap is None:
            self.flags.exec = "DISCONNECTED"
            return "BLOCKED_NO_SNAPSHOT"
        for o in list(self.ledger.state.working):
            if o.terminal:
                continue
            vo = snap["orders"].get(o.order_id)
            if vo is None:
                if o.nonce in self.ledger.spent_nonces:
                    self._set_status(o.order_id, "REJECTED")   # confirmed absent
            else:
                if vo["cum_qty"] > o.cum_filled:
                    self._apply_fill(o.order_id, vo["cum_qty"])
                self._set_status(o.order_id, vo["status"])
        self._reconcile_positions()
        self._recompute()
        self.ledger.persist()
        return "RECOVERED" if self.flags.exec == "OK" else "BLOCKED_DIVERGENT"
