"""M1.2 authorization kernel and execution gateway.

Safety rules (unchanged):
- one shared lock serializes authorization, submission, events and recovery;
- every order is bound to an immutable intent and ledger snapshot;
- nonce is durably consumed before transmission;
- ambiguous broker state never releases a reservation;
- fills use execution prices, not the order limit price;
- cancellation/rejection is terminal only after broker snapshot confirmation.

M1.2 fixes:
  A1 reconciliation compares venue T+0 cash against settled + receivables
  A2 RESERVED / EXPIRED orders are ledger-only and excluded from venue reconciliation
  A3 every mutating path persists the ledger and records the exec_id
  A4 fill accounting is snapshot-authoritative (notional, not last-fill price)
  A5 receivable due date uses the CURRENT session, not the reservation date
  A6 m_vertex is passed from cfg.ops into risk.evaluate
  A7 a BUY fill that drives settled cash negative is applied then flagged DIVERGENT
  A8 Flags are persisted and can be restored
  A9 TIMEOUT handled explicitly as an unresolved, non-terminal state
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import uuid
from dataclasses import asdict, dataclass

from .core import (Money, Shares, Clock, ConfigAuthority, EventLog, Instrument,
                   Ledger, Store, WorkingOrder, ceil_bps, sha)
from .flags import Flags
from .risk import evaluate, reservations, CONSTRAINTS
from .venue import MockVenue, ExecEvent

# Ledger-only statuses: never transmitted, therefore absent from venue snapshots.
LEDGER_ONLY = frozenset({"RESERVED", "EXPIRED"})
TERMINAL_VENUE = frozenset({"CANCELLED", "REJECTED", "FILLED"})
KNOWN_KINDS = frozenset({"ACK", "FILL", "CANCEL_ACK", "CANCEL_REJECT", "REJECT", "TIMEOUT"})


class Reject(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


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
        d = asdict(self)
        d.pop("sig")
        return d


def sign(key: bytes, body: dict) -> str:
    return hmac.new(key, sha(body).encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Intent:
    symbol: str
    side: str
    qty: Shares
    price: Money
    cost_bps: int

    def validate(self) -> None:
        if self.side not in {"BUY", "SELL"}:
            raise Reject("BAD_SIDE")
        for name, value in (("qty", self.qty), ("price", self.price), ("cost_bps", self.cost_bps)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise Reject("BAD_" + name.upper() + "_TYPE")
        if self.qty <= 0:
            raise Reject("BAD_QTY")
        if self.price <= 0:
            raise Reject("BAD_PRICE")
        if not 0 <= self.cost_bps <= 10_000:
            raise Reject("BAD_COST")

    def hash(self) -> str:
        return sha(asdict(self))


class Kernel:
    def __init__(self, key: bytes, cfg: ConfigAuthority, instruments: dict[str, Instrument],
                 scenarios: tuple[dict[str, int], ...], lock=None):
        self.key, self.cfg, self.instruments, self.scenarios = key, cfg, instruments, scenarios
        self.lock = lock or threading.RLock()

    def authorize(self, ledger: Ledger, intent: Intent, flags: Flags,
                  clock: Clock, watchdog_stop: bool) -> Token:
        with self.lock:
            intent.validate()
            cfg = self.cfg
            if watchdog_stop:
                raise Reject("WATCHDOG_STOP")
            if not cfg.attested():
                raise Reject("CONFIG_UNATTESTED")
            if (ledger.state.mandate_version != cfg.mandate.version or
                    ledger.state.config_digest != cfg.mandate.digest()):
                raise Reject("CONFIG_STATE_MISMATCH")
            perms = flags.permissions()
            if "SUBMIT" not in perms:
                raise Reject("MODE_BLOCKS_SUBMIT")
            if intent.side == "BUY" and "NEW_RISK" not in perms:
                raise Reject("MODE_BLOCKS_NEW_RISK")
            if intent.symbol not in cfg.mandate.allowlist:
                raise Reject("ALLOWLIST")
            inst = self.instruments.get(intent.symbol)
            if inst is None:
                raise Reject("UNKNOWN_INSTRUMENT")
            if inst.product_class not in cfg.mandate.product_classes:
                raise Reject("PRODUCT_CLASS")
            if inst.halted:
                raise Reject("HALTED")
            if intent.side == "SELL" and cfg.floor.long_only:
                have = ledger.state.positions.get(intent.symbol, 0)
                committed = sum(o.remaining for o in ledger.working_active()
                                if o.symbol == intent.symbol and o.side == "SELL")
                if intent.qty + committed > have:
                    raise Reject("SHORT_PROHIBITED")
            if (clock.wall < ledger.state.mark_wall or
                    clock.wall - ledger.state.mark_wall > cfg.mandate.tau_data_secs):
                raise Reject("STALE_MARKS")
            if intent.side == "BUY" and ledger.drawdown_bps() >= cfg.mandate.drawdown_trigger_bps:
                raise Reject("DRAWDOWN_TRIGGER")
            notional = intent.qty * intent.price
            if notional > cfg.mandate.order_notional_cap:
                raise Reject("ORDER_NOTIONAL")
            if inst.adv <= 0 or notional > (inst.adv * cfg.mandate.participation_bps // 10_000):
                raise Reject("PARTICIPATION")
            active = ledger.working_active()
            if len(active) + 1 > cfg.ops.m_vertex:
                raise Reject("M_VERTEX")

            oid, nonce = f"O{uuid.uuid4().hex[:16]}", uuid.uuid4().hex
            cand = WorkingOrder(oid, intent.symbol, intent.side, intent.qty, 0,
                                intent.price, intent.cost_bps, "RESERVED",
                                ledger.state.trade_date, nonce)
            orders = active + (cand,)
            # A6: honor configured vertex budget, not risk.py's default.
            ev = evaluate(ledger.state, orders, self.instruments, cfg.mandate,
                          self.scenarios, m_vertex=cfg.ops.m_vertex)
            if not ev.ok:
                raise Reject(ev.reason)

            old_working = ledger.state.working
            old_reserved = dict(ledger.reserved)
            old_seq = ledger.state.seq
            try:
                ledger.state.working = ledger.state.working + (cand,)
                ledger.reserved = reservations(ev)
                ledger.state.seq = clock.next()
                tok0 = Token(intent.hash(), ledger.state.digest(), oid, clock.seq,
                             clock.seq + cfg.ops.auth_ttl_seqs, nonce, "")
                tok = Token(**{**asdict(tok0), "sig": sign(self.key, tok0.body())})
                ledger.persist()
                return tok
            except Exception:
                ledger.state.working = old_working
                ledger.reserved = old_reserved
                ledger.state.seq = old_seq
                raise


class Gateway:
    def __init__(self, key: bytes, cfg: ConfigAuthority, ledger: Ledger, log: EventLog,
                 venue: MockVenue, clock: Clock, instruments: dict[str, Instrument],
                 scenarios: tuple[dict[str, int], ...], flags: Flags,
                 watchdog_path: str, lock=None, flags_store: Store | None = None):
        self.key, self.cfg, self.ledger, self.log = key, cfg, ledger, log
        self.venue, self.clock, self.instruments = venue, clock, instruments
        self.scenarios, self.flags, self.watchdog_path = scenarios, flags, watchdog_path
        self.fail_point: str | None = None
        self.lock = lock or threading.RLock()
        # A8: flags are durable. Default path sits beside the watchdog file.
        self.flags_store = flags_store or Store(
            os.path.join(os.path.dirname(watchdog_path) or ".", "flags.json")
        )

    # ---------------- persistence helpers ----------------

    def _persist(self) -> None:
        self.ledger.persist()
        self.flags_store.save(asdict(self.flags))

    def restore_flags(self) -> bool:
        d = self.flags_store.load()
        if not d:
            return False
        known = set(Flags.__dataclass_fields__)
        for k, v in d.items():
            if k in known:
                setattr(self.flags, k, v)
        return True

    def _watchdog(self) -> bool:
        return os.path.exists(self.watchdog_path)

    def _find(self, order_id: str) -> WorkingOrder | None:
        return next((o for o in self.ledger.state.working if o.order_id == order_id), None)

    def _replace_order(self, order_id: str, **changes) -> None:
        found = False
        updated = []
        for o in self.ledger.state.working:
            if o.order_id == order_id:
                updated.append(WorkingOrder(**{**asdict(o), **changes}))
                found = True
            else:
                updated.append(o)
        if not found:
            raise Reject("UNKNOWN_ORDER")
        self.ledger.state.working = tuple(updated)

    # ---------------- submission ----------------

    def submit(self, tok: Token, intent: Intent) -> None:
        with self.lock:
            intent.validate()
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
            if not (tok.issue_seq <= self.clock.seq <= tok.valid_until_seq):
                self._expire(tok)
                raise Reject("TOKEN_EXPIRED")
            if tok.state_hash != self.ledger.state.digest():
                raise Reject("STALE_SNAPSHOT")
            reserved = self._find(tok.order_id)
            if reserved is None or reserved.status != "RESERVED":
                raise Reject("ORDER_NOT_RESERVED")
            if (reserved.symbol, reserved.side, reserved.qty, reserved.price,
                    reserved.cost_bps, reserved.nonce) != (
                    intent.symbol, intent.side, intent.qty, intent.price,
                    intent.cost_bps, tok.nonce):
                raise Reject("RESERVATION_INTENT_MISMATCH")
            recent = [s for s in self.ledger.submit_seq
                      if self.clock.seq - s < self.cfg.mandate.rate_window]
            if len(recent) >= self.cfg.mandate.rate_limit:
                raise Reject("RATE_LIMIT")

            # Durably consume the nonce BEFORE transmission.
            self.ledger.spent_nonces.add(tok.nonce)
            self.ledger.submit_seq.append(self.clock.seq)
            self._replace_order(tok.order_id, status="SUBMITTING")
            self._persist()
            if self.fail_point == "CRASH_A":
                raise RuntimeError("injected crash before venue send")
            try:
                self.venue.send(tok.order_id, intent.symbol, intent.side,
                                intent.qty, intent.price, intent.cost_bps)
            except Exception:
                self.flags.exec = "DISCONNECTED"
                self._recompute()
                self._persist()
                raise
            if self.fail_point == "CRASH_B":
                raise RuntimeError("injected crash after venue send")
            self._replace_order(tok.order_id, status="SUBMITTED")
            self._persist()
            self.log.append("SUBMIT", {"order_id": tok.order_id,
                                       "intent_hash": intent.hash()}, self.clock.next())

    def _expire(self, tok: Token) -> None:
        o = self._find(tok.order_id)
        if o and o.status == "RESERVED" and tok.nonce not in self.ledger.spent_nonces:
            self._replace_order(tok.order_id, status="EXPIRED")
            self._recompute()
            self._persist()

    # ---------------- event processing ----------------

    def process(self, ev: ExecEvent) -> str:
        with self.lock:
            event_hash = sha(asdict(ev))
            if ev.exec_id in self.ledger.applied_exec_ids:
                prior = self.ledger.applied_exec_hashes.get(ev.exec_id)
                if prior is not None and prior != event_hash:
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "EXEC_ID_PAYLOAD_CONFLICT"
                return "NOOP_DUPLICATE"

            o = self._find(ev.order_id)
            if o is None:
                self.flags.exec = "DIVERGENT"
                self._persist()
                return "UNKNOWN_ORDER_BLOCKED"
            if ev.kind not in KNOWN_KINDS:
                self.flags.exec = "DIVERGENT"
                self._persist()
                return "UNKNOWN_EVENT_BLOCKED"

            # A9: timeout is unresolved, never terminal, never releases reservation.
            if ev.kind == "TIMEOUT":
                self.flags.exec = "DISCONNECTED"
                if not o.terminal:
                    self._replace_order(ev.order_id, status="UNKNOWN")
                self._record_exec(ev, event_hash)
                self._recompute()
                self._persist()
                self.log.append("TIMEOUT", {"order_id": ev.order_id}, self.clock.next())
                return "TIMEOUT_UNRESOLVED_RESERVED"

            if ev.kind == "FILL":
                if ev.cum_qty > o.qty or ev.price <= 0:
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "INVALID_FILL_BLOCKED"
                snap = self.venue.snapshot()
                if snap is None or not snap.get("complete", False):
                    self.flags.exec = "DISCONNECTED"
                    self._persist()
                    return "FILL_UNCONFIRMED_RESERVED"
                vo = snap["orders"].get(ev.order_id)
                if vo is None:
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "FILL_UNCONFIRMED_RESERVED"
                # A4: the venue snapshot is authoritative for notional and cost.
                if vo["cum_qty"] < o.cum_filled or vo["cum_qty"] > o.qty:
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "INVALID_FILL_BLOCKED"
                if vo["cum_qty"] > o.cum_filled:
                    dn = vo["filled_notional"] - o.filled_notional
                    dc = vo["filled_cost"] - o.filled_cost
                    dq = vo["cum_qty"] - o.cum_filled
                    if dn <= 0 or dc < 0 or dn < dq:
                        self.flags.exec = "DIVERGENT"
                        self._persist()
                        return "INVALID_FILL_BLOCKED"
                    self._apply_snapshot_fill(ev.order_id, vo["cum_qty"], dn, dc)
                if ev.cum_qty < o.cum_filled:
                    # A3: stale-cum event. Reconcile, record, persist.
                    self._reconcile_snapshot()
                    self._record_exec(ev, event_hash)
                    self._recompute()
                    self._persist()
                    return "SNAPSHOT_RECONCILED"

            elif ev.kind == "ACK":
                if o.status in {"RESERVED", "SUBMITTING", "SUBMITTED"}:
                    self._replace_order(ev.order_id, status="ACKNOWLEDGED")

            elif ev.kind == "CANCEL_ACK":
                snap = self.venue.snapshot()
                if snap is None:
                    self.flags.exec = "DISCONNECTED"
                    self._persist()
                    return "CANCEL_UNCONFIRMED_RESERVED"
                vo = snap["orders"].get(ev.order_id)
                if not vo or vo["status"] != "CANCELLED":
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "CANCEL_UNCONFIRMED_RESERVED"
                if vo["cum_qty"] > o.cum_filled:
                    dn = vo["filled_notional"] - o.filled_notional
                    dc = vo["filled_cost"] - o.filled_cost
                    self._apply_snapshot_fill(ev.order_id, vo["cum_qty"], dn, dc)
                current = self._find(ev.order_id)
                if vo["cum_qty"] != current.cum_filled:
                    self.flags.exec = "DIVERGENT"
                    self._persist()
                    return "CANCEL_UNCONFIRMED_RESERVED"
                self._replace_order(ev.order_id, status="CANCELLED")

            elif ev.kind == "CANCEL_REJECT":
                pass

            elif ev.kind == "REJECT":
                snap = self.venue.snapshot()
                vo = snap["orders"].get(ev.order_id) if snap else None
                if not vo or vo["status"] != "REJECTED" or vo["cum_qty"] != 0:
                    self.flags.exec = "DIVERGENT" if snap else "DISCONNECTED"
                    self._persist()
                    return "REJECT_UNCONFIRMED_RESERVED"
                self._replace_order(ev.order_id, status="REJECTED")

            self._record_exec(ev, event_hash)
            self._reconcile_snapshot()
            self._recompute()
            self._persist()
            self.log.append("EXEC", {"exec_id": ev.exec_id, "kind": ev.kind,
                                     "order_id": ev.order_id, "cum_qty": ev.cum_qty,
                                     "price": ev.price}, self.clock.next())
            return "APPLIED"

    def _record_exec(self, ev: ExecEvent, event_hash: str) -> None:
        self.ledger.applied_exec_ids.add(ev.exec_id)
        self.ledger.applied_exec_hashes[ev.exec_id] = event_hash

    # ---------------- fill application ----------------

    def _apply_snapshot_fill(self, order_id: str, cum: int,
                             delta_notional: int, delta_cost: int) -> None:
        """Single authoritative fill path. Notional/cost come from the venue."""
        o = self._find(order_id)
        if o is None:
            raise Reject("UNKNOWN_ORDER")
        dq = cum - o.cum_filled
        if dq <= 0 or cum > o.qty or delta_notional <= 0 or delta_cost < 0:
            raise Reject("INVALID_SNAPSHOT_FILL")

        if o.side == "BUY":
            self.ledger.state.positions[o.symbol] = \
                self.ledger.state.positions.get(o.symbol, 0) + dq
            self.ledger.state.cash_settled -= delta_notional + delta_cost
            # A7: the fill is real, so apply it — but never leave it unflagged.
            if self.ledger.state.cash_settled < 0:
                self.flags.exec = "DIVERGENT"
                self.log.append("SETTLED_CASH_NEGATIVE",
                                {"order_id": order_id,
                                 "cash_settled": self.ledger.state.cash_settled},
                                self.clock.next())
        else:
            have = self.ledger.state.positions.get(o.symbol, 0)
            if dq > have:
                self.flags.exec = "DIVERGENT"
                raise Reject("FILL_WOULD_CREATE_SHORT")
            self.ledger.state.positions[o.symbol] = have - dq
            net = delta_notional - delta_cost
            if net > 0:
                # A5: settle from the CURRENT session, not the reservation date.
                self.ledger.record_sell_receivable(
                    self.ledger.state.trade_date, net, self.cfg.mandate.settle_lag_days
                )

        self._replace_order(
            order_id, cum_filled=cum,
            status="FILLED" if cum == o.qty else "PARTIALLY_FILLED",
            filled_notional=o.filled_notional + delta_notional,
            filled_cost=o.filled_cost + delta_cost,
        )
        self.ledger.mark_peak()

    def _apply_fill(self, order_id: str, cum: int, incremental_price: int) -> None:
        """Compatibility wrapper. Derives notional from price only as a fallback."""
        o = self._find(order_id)
        if o is None or cum < o.cum_filled or cum > o.qty:
            raise Reject("INVALID_CUMULATIVE_FILL")
        delta = cum - o.cum_filled
        if delta == 0:
            return
        notional = delta * incremental_price
        self._apply_snapshot_fill(order_id, cum, notional,
                                  ceil_bps(notional, o.cost_bps))

    # ---------------- reconciliation ----------------

    def _reconcile_snapshot(self) -> bool:
        snap = self.venue.snapshot()
        if snap is None or not snap.get("complete", False):
            self.flags.exec = "DISCONNECTED"
            return False

        # A1: venue cash is T+0. Ledger holds unsettled proceeds in receivables.
        ledger_cash_equivalent = (self.ledger.state.cash_settled
                                  + self.ledger.state.receivable_total())
        if snap["positions"] != self.ledger.state.positions or \
                snap["cash"] != ledger_cash_equivalent:
            self.flags.exec = "DIVERGENT"
            return False

        for o in self.ledger.state.working:
            # A2: ledger-only orders were never transmitted.
            if o.status in LEDGER_ONLY:
                continue
            vo = snap["orders"].get(o.order_id)
            if vo is None:
                if not o.terminal:
                    self.flags.exec = "DIVERGENT"
                    return False
                continue
            if vo["qty"] != o.qty or vo["symbol"] != o.symbol or vo["side"] != o.side:
                self.flags.exec = "DIVERGENT"
                return False
            if vo["cum_qty"] < o.cum_filled or vo["cum_qty"] > o.qty:
                self.flags.exec = "DIVERGENT"
                return False
            if vo["cum_qty"] > o.cum_filled:
                dn = vo["filled_notional"] - o.filled_notional
                dc = vo["filled_cost"] - o.filled_cost
                dq = vo["cum_qty"] - o.cum_filled
                if dn <= 0 or dc < 0 or dn < dq:
                    self.flags.exec = "DIVERGENT"
                    return False
                self._apply_snapshot_fill(o.order_id, vo["cum_qty"], dn, dc)
                o = self._find(o.order_id)
            if vo["filled_notional"] != o.filled_notional or vo["filled_cost"] != o.filled_cost:
                self.flags.exec = "DIVERGENT"
                return False
            if o.status == "UNKNOWN" or vo["status"] == "UNKNOWN":
                self.flags.exec = "DIVERGENT"
                return False
            if o.terminal:
                if vo["status"] != o.status:
                    self.flags.exec = "DIVERGENT"
                    return False
            elif vo["status"] in TERMINAL_VENUE:
                self._replace_order(o.order_id, status=vo["status"])

        self.flags.exec = "OK"
        return True

    def _recompute(self) -> None:
        active = self.ledger.working_active()
        if not active:
            self.ledger.reserved = {c: 0 for c in CONSTRAINTS}
            if self.flags.envelope == "OVERFLOW":
                self.flags.envelope = "OK"
            return
        # A6: honor configured vertex budget.
        ev = evaluate(self.ledger.state, active, self.instruments, self.cfg.mandate,
                      self.scenarios, m_vertex=self.cfg.ops.m_vertex)
        if not ev.ok:
            # Preserve existing reservations. Never zero them on failure.
            self.flags.envelope = "OVERFLOW"
            self.log.append("OVERFLOW", {"reason": ev.reason}, self.clock.next())
            return
        self.ledger.reserved = reservations(ev)
        if self.flags.envelope == "OVERFLOW":
            self.flags.envelope = "OK"

    # ---------------- recovery ----------------

    def recover(self) -> str:
        with self.lock:
            snap = self.venue.snapshot()
            if snap is None or not snap.get("complete", False):
                self.flags.exec = "DISCONNECTED"
                self._persist()
                return "BLOCKED_NO_AUTHORITATIVE_SNAPSHOT"
            unresolved = False
            for o in self.ledger.state.working:
                # A2: never transmitted -> nothing to resolve at the venue.
                if o.status in LEDGER_ONLY:
                    continue
                vo = snap["orders"].get(o.order_id)
                if vo is None:
                    if not o.terminal:
                        unresolved = True
                    continue
                if vo["cum_qty"] != o.cum_filled:
                    unresolved = True
                    continue
                if (vo["qty"], vo["symbol"], vo["side"]) != (o.qty, o.symbol, o.side):
                    unresolved = True
                    continue
                if vo["status"] == "UNKNOWN" or o.status == "UNKNOWN":
                    unresolved = True
                    continue
                if vo["status"] in TERMINAL_VENUE:
                    self._replace_order(o.order_id, status=vo["status"])
                elif not o.terminal:
                    self._replace_order(o.order_id, status=vo["status"])
            ok = self._reconcile_snapshot()
            if unresolved or not ok:
                self.flags.exec = "DIVERGENT"
                self._recompute()
                self._persist()
                return "BLOCKED_UNRESOLVED"
            self._recompute()
            self._persist()
            return "RECOVERED"
