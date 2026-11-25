"""M1.5 authorization kernel and execution gateway.

Changes over M1.4:
  - RED-TEAM Q: submit() re-checks NEW_RISK for BUY and re-checks mark staleness.
    A token authorized under permissive flags must not transmit after flags revoke
    new risk. Flags are not part of VerifiedState.digest(), so the token alone
    cannot enforce this.
  - RED-TEAM S: restore_flags() restores and verifies the EventLog BEFORE comparing
    audit heads. A missing/short chain no longer truncates the audit file.
  - RED-TEAM S: audit-head comparison uses chain membership, not equality. The log
    may legitimately be AHEAD of the composite (persist-then-append window); only a
    log that has LOST a recorded event is a divergence.
  - RED-TEAM S: mismatch paths no longer overwrite the composite. Evidence is
    preserved for operator inspection.
  - Composite atomic persistence, stale-o refresh, submit rollback, nonpositive
    sale proceeds handling retained from M1.4.

PRESERVED VERBATIM: CRASH_A / CRASH_B, RESERVATION_INTENT_MISMATCH,
_replace_order raising UNKNOWN_ORDER, Kernel.authorize persisting before return,
LEDGER_ONLY skip, sticky latch, all execution-path methods.

DEFERRED (requires Token schema change + full test review):
  - RED-TEAM R: operational-config version is not bound into the token.
  - RED-TEAM H: ledger control state (nonces, submit_seq, reserved) is not bound.
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

LEDGER_ONLY = frozenset({"RESERVED", "EXPIRED"})
TERMINAL_VENUE = frozenset({"CANCELLED", "REJECTED", "FILLED"})
KNOWN_KINDS = frozenset({"ACK", "FILL", "CANCEL_ACK", "CANCEL_REJECT", "REJECT", "TIMEOUT"})

COMPOSITE_SCHEMA_VERSION = 1
ZERO_HEAD = "0" * 64


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
    mandate_version: int
    config_digest: str
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
                tok0 = Token(
                    intent_hash=intent.hash(),
                    state_hash=ledger.state.digest(),
                    order_id=oid,
                    issue_seq=clock.seq,
                    valid_until_seq=clock.seq + cfg.ops.auth_ttl_seqs,
                    nonce=nonce,
                    mandate_version=cfg.mandate.version,
                    config_digest=cfg.mandate.digest(),
                    sig="",
                )
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
        base_dir = os.path.dirname(watchdog_path) or "."
        self.flags_store = flags_store or Store(os.path.join(base_dir, "flags.json"))
        self.composite_store = Store(os.path.join(base_dir, "composite_state.json"))
        self._divergence_latched = False

    # ---- durable composite write: ledger + flags + latch + audit head ----

    def _persist(self) -> None:
        payload = {
            "schema_version": COMPOSITE_SCHEMA_VERSION,
            "ledger": self.ledger.to_dict(),
            "flags": asdict(self.flags),
            "divergence_latched": self._divergence_latched,
            "ledger_digest": self.ledger.state.digest(),
            "audit_head": self.log.head(),
        }
        self.composite_store.save(payload)

    def _latch_silent(self, reason: str) -> None:
        """Latch WITHOUT appending to the audit chain and WITHOUT persisting.

        Used on restore failures, where appending would overwrite the very
        evidence that failed verification. The reason is surfaced via flags.exec
        and must be inspected by an operator against the on-disk artifacts.
        """
        self._divergence_latched = True
        self.flags.exec = "DIVERGENT"
        self._restore_fault = reason

    def restore_flags(self) -> bool:
        """Restore composite state, verifying the audit chain FIRST.

        Order matters: comparing a persisted audit head against an unrestored
        (empty) chain produces a false divergence on every restart, and the
        subsequent latch append truncates the audit file.
        """
        self._restore_fault = None

        # 1. Restore and verify the audit chain before any head comparison.
        try:
            self.log.restore()
        except ValueError:
            self._latch_silent("AUDIT_CHAIN_VERIFY_FAILED")
            return True

        d = self.composite_store.load()

        if d is None:
            # Legacy migration: ledger.json + flags.json.
            legacy_ledger = Ledger.restore(self.ledger.store)
            legacy_flags = self.flags_store.load()

            if legacy_ledger is None and not legacy_flags:
                return False

            if legacy_ledger is not None:
                self.ledger.state = legacy_ledger.state
                self.ledger.spent_nonces = legacy_ledger.spent_nonces
                self.ledger.applied_exec_ids = legacy_ledger.applied_exec_ids
                self.ledger.submit_seq = legacy_ledger.submit_seq
                self.ledger.reserved = legacy_ledger.reserved
                self.ledger.applied_exec_hashes = legacy_ledger.applied_exec_hashes
                self.ledger.last_settled_session = legacy_ledger.last_settled_session

            if legacy_flags:
                payload = legacy_flags.get("flags", legacy_flags)
                known = set(Flags.__dataclass_fields__)
                for k, v in payload.items():
                    if k in known:
                        setattr(self.flags, k, v)
                self._divergence_latched = bool(legacy_flags.get("divergence_latched", False))
                if (legacy_flags.get("ledger_digest") and
                        legacy_flags["ledger_digest"] != self.ledger.state.digest()):
                    self.latch_divergence("FLAGS_LEDGER_DIGEST_MISMATCH")

            self._persist()
            return True

        # 2. Schema mismatch: preserve the artifact, do not overwrite.
        if d.get("schema_version") != COMPOSITE_SCHEMA_VERSION:
            self._latch_silent("FLAGS_SCHEMA_UNKNOWN")
            return True

        ledger_data = d.get("ledger")
        if ledger_data is None:
            self._latch_silent("COMPOSITE_LEDGER_MISSING")
            return True

        restored = Ledger.from_dict(ledger_data, self.ledger.store)
        self.ledger.state = restored.state
        self.ledger.spent_nonces = restored.spent_nonces
        self.ledger.applied_exec_ids = restored.applied_exec_ids
        self.ledger.submit_seq = restored.submit_seq
        self.ledger.reserved = restored.reserved
        self.ledger.applied_exec_hashes = restored.applied_exec_hashes
        self.ledger.last_settled_session = restored.last_settled_session

        payload = d.get("flags", {})
        known = set(Flags.__dataclass_fields__)
        for k, v in payload.items():
            if k in known:
                setattr(self.flags, k, v)

        self._divergence_latched = bool(d.get("divergence_latched", False))
        if self._divergence_latched:
            self.flags.exec = "DIVERGENT"

        if d.get("ledger_digest") and d["ledger_digest"] != self.ledger.state.digest():
            self._latch_silent("FLAGS_LEDGER_DIGEST_MISMATCH")
            return True

        # 3. Audit head: the log may legitimately be AHEAD of the composite
        #    (persist-then-append window). Only a LOST event is a divergence.
        expected = d.get("audit_head")
        if expected and expected != ZERO_HEAD:
            chain_hashes = {r.get("hash") for r in self.log.chain}
            if expected not in chain_hashes:
                self._latch_silent("AUDIT_HEAD_MISSING_FROM_CHAIN")
                return True

        return True

    def latch_divergence(self, reason: str) -> None:
        self._divergence_latched = True
        self.flags.exec = "DIVERGENT"
        self.log.append("DIVERGENCE_LATCHED", {"reason": reason}, self.clock.next())

    def clear_divergence(self, operator_reason: str) -> None:
        self._divergence_latched = False
        self.flags.exec = "OK"
        self.log.append("DIVERGENCE_CLEARED", {"reason": operator_reason}, self.clock.next())
        self._persist()

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

    def submit(self, tok: Token, intent: Intent) -> None:
        with self.lock:
            intent.validate()
            if self._watchdog():
                raise Reject("WATCHDOG_STOP")

            # RED-TEAM Q: re-check the FULL permission set at transmission time.
            # Flags are not bound into VerifiedState.digest(), so a token issued
            # under permissive flags would otherwise survive a NEW_RISK revoke.
            perms = self.flags.permissions()
            if "SUBMIT" not in perms:
                raise Reject("MODE_BLOCKS_SUBMIT")
            if intent.side == "BUY" and "NEW_RISK" not in perms:
                raise Reject("MODE_BLOCKS_NEW_RISK")

            # RED-TEAM Q: re-check data freshness at transmission time.
            if (self.clock.wall < self.ledger.state.mark_wall or
                    self.clock.wall - self.ledger.state.mark_wall
                    > self.cfg.mandate.tau_data_secs):
                raise Reject("STALE_MARKS")

            if not hmac.compare_digest(tok.sig, sign(self.key, tok.body())):
                raise Reject("BAD_SIGNATURE")
            if tok.intent_hash != intent.hash():
                raise Reject("INTENT_MUTATED")
            if not self.cfg.attested():
                raise Reject("CONFIG_UNATTESTED")
            if (tok.mandate_version != self.cfg.mandate.version or
                    tok.config_digest != self.cfg.mandate.digest()):
                self._expire(tok)
                raise Reject("TOKEN_CONFIG_STALE")
            if (self.ledger.state.mandate_version != self.cfg.mandate.version or
                    self.ledger.state.config_digest != self.cfg.mandate.digest()):
                raise Reject("CONFIG_STATE_MISMATCH")
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

            old_nonces = set(self.ledger.spent_nonces)
            old_submit_seq = list(self.ledger.submit_seq)
            old_working = self.ledger.state.working

            try:
                self.ledger.spent_nonces.add(tok.nonce)
                self.ledger.submit_seq.append(self.clock.seq)
                self._replace_order(tok.order_id, status="SUBMITTING")
                self._persist()
            except Exception:
                self.ledger.spent_nonces = old_nonces
                self.ledger.submit_seq = old_submit_seq
                self.ledger.state.working = old_working
                raise

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

    def process(self, ev: ExecEvent) -> str:
        with self.lock:
            event_hash = sha(asdict(ev))
            if ev.exec_id in self.ledger.applied_exec_ids:
                prior = self.ledger.applied_exec_hashes.get(ev.exec_id)
                if prior is not None and prior != event_hash:
                    self.latch_divergence("EXEC_ID_PAYLOAD_CONFLICT")
                    self._persist()
                    return "EXEC_ID_PAYLOAD_CONFLICT"
                return "NOOP_DUPLICATE"

            o = self._find(ev.order_id)
            if o is None:
                self.latch_divergence("UNKNOWN_ORDER")
                self._persist()
                return "UNKNOWN_ORDER_BLOCKED"
            if ev.kind not in KNOWN_KINDS:
                self.latch_divergence("UNKNOWN_EVENT_KIND")
                self._persist()
                return "UNKNOWN_EVENT_BLOCKED"

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
                    self.latch_divergence("INVALID_FILL")
                    self._persist()
                    return "INVALID_FILL_BLOCKED"
                snap = self.venue.snapshot()
                if snap is None or not snap.get("complete", False):
                    self.flags.exec = "DISCONNECTED"
                    self._persist()
                    return "FILL_UNCONFIRMED_RESERVED"
                vo = snap["orders"].get(ev.order_id)
                if vo is None:
                    self.latch_divergence("FILL_WITHOUT_VENUE_ORDER")
                    self._persist()
                    return "FILL_UNCONFIRMED_RESERVED"
                if vo["cum_qty"] < o.cum_filled or vo["cum_qty"] > o.qty:
                    self.latch_divergence("VENUE_CUM_OUT_OF_RANGE")
                    self._persist()
                    return "INVALID_FILL_BLOCKED"
                if vo["cum_qty"] > o.cum_filled:
                    dn = vo["filled_notional"] - o.filled_notional
                    dc = vo["filled_cost"] - o.filled_cost
                    dq = vo["cum_qty"] - o.cum_filled
                    if dn <= 0 or dc < 0 or dn < dq:
                        self.latch_divergence("INVALID_VENUE_NOTIONAL")
                        self._persist()
                        return "INVALID_FILL_BLOCKED"
                    self._apply_snapshot_fill(ev.order_id, vo["cum_qty"], dn, dc)
                    o = self._find(ev.order_id)
                if ev.cum_qty < o.cum_filled:
                    reconciled = self._reconcile_snapshot()
                    self._record_exec(ev, event_hash)
                    self._recompute()
                    self._persist()
                    return "SNAPSHOT_RECONCILED" if reconciled else "SNAPSHOT_RECONCILED_DIVERGENT"

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
                    self.latch_divergence("CANCEL_UNCONFIRMED")
                    self._persist()
                    return "CANCEL_UNCONFIRMED_RESERVED"
                if vo["cum_qty"] > o.cum_filled:
                    dn = vo["filled_notional"] - o.filled_notional
                    dc = vo["filled_cost"] - o.filled_cost
                    self._apply_snapshot_fill(ev.order_id, vo["cum_qty"], dn, dc)
                current = self._find(ev.order_id)
                if vo["cum_qty"] != current.cum_filled:
                    self.latch_divergence("CANCEL_CUM_MISMATCH")
                    self._persist()
                    return "CANCEL_UNCONFIRMED_RESERVED"
                self._replace_order(ev.order_id, status="CANCELLED")

            elif ev.kind == "CANCEL_REJECT":
                pass

            elif ev.kind == "REJECT":
                snap = self.venue.snapshot()
                vo = snap["orders"].get(ev.order_id) if snap else None
                if not vo or vo["status"] != "REJECTED" or vo["cum_qty"] != 0:
                    if snap:
                        self.latch_divergence("REJECT_UNCONFIRMED")
                    else:
                        self.flags.exec = "DISCONNECTED"
                    self._persist()
                    return "REJECT_UNCONFIRMED_RESERVED"
                self._replace_order(ev.order_id, status="REJECTED")

            self._record_exec(ev, event_hash)
            reconciled = self._reconcile_snapshot()
            self._recompute()
            self._persist()
            self.log.append("EXEC", {"exec_id": ev.exec_id, "kind": ev.kind,
                                     "order_id": ev.order_id, "cum_qty": ev.cum_qty,
                                     "price": ev.price, "reconciled": reconciled},
                            self.clock.next())
            return "APPLIED" if reconciled else "APPLIED_RECONCILE_DIVERGENT"

    def _record_exec(self, ev: ExecEvent, event_hash: str) -> None:
        self.ledger.applied_exec_ids.add(ev.exec_id)
        self.ledger.applied_exec_hashes[ev.exec_id] = event_hash

    def _apply_snapshot_fill(self, order_id: str, cum: int,
                             delta_notional: int, delta_cost: int) -> None:
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
            if self.ledger.state.cash_settled < 0:
                self.latch_divergence("SETTLED_CASH_NEGATIVE")
        else:
            have = self.ledger.state.positions.get(o.symbol, 0)
            if dq > have:
                self.latch_divergence("FILL_WOULD_CREATE_SHORT")
                raise Reject("FILL_WOULD_CREATE_SHORT")
            self.ledger.state.positions[o.symbol] = have - dq
            net = delta_notional - delta_cost
            if net > 0:
                self.ledger.record_sell_receivable(
                    self.ledger.state.trade_date, net, self.cfg.mandate.settle_lag_days)
            elif net == 0:
                self.log.append("ZERO_NET_SALE_PROCEEDS",
                                {"order_id": order_id,
                                 "delta_notional": delta_notional,
                                 "delta_cost": delta_cost},
                                self.clock.next())
            else:
                self.latch_divergence("NEGATIVE_NET_SALE_PROCEEDS")

        self._replace_order(
            order_id, cum_filled=cum,
            status="FILLED" if cum == o.qty else "PARTIALLY_FILLED",
            filled_notional=o.filled_notional + delta_notional,
            filled_cost=o.filled_cost + delta_cost)
        self.ledger.mark_peak()

    def _reconcile_snapshot(self) -> bool:
        snap = self.venue.snapshot()
        if snap is None or not snap.get("complete", False):
            self.flags.exec = "DISCONNECTED"
            return False

        ledger_cash_equivalent = (self.ledger.state.cash_settled
                                  + self.ledger.state.receivable_total())
        if snap["positions"] != self.ledger.state.positions or \
                snap["cash"] != ledger_cash_equivalent:
            self.latch_divergence("CASH_OR_POSITION_MISMATCH")
            return False

        for o in self.ledger.state.working:
            if o.status in LEDGER_ONLY:
                continue
            vo = snap["orders"].get(o.order_id)
            if vo is None:
                if not o.terminal:
                    self.latch_divergence("MISSING_VENUE_ORDER")
                    return False
                continue
            if vo["qty"] != o.qty or vo["symbol"] != o.symbol or vo["side"] != o.side:
                self.latch_divergence("ORDER_TERMS_MISMATCH")
                return False
            if vo["cum_qty"] < o.cum_filled or vo["cum_qty"] > o.qty:
                self.latch_divergence("VENUE_CUM_OUT_OF_RANGE")
                return False
            if vo["cum_qty"] > o.cum_filled:
                dn = vo["filled_notional"] - o.filled_notional
                dc = vo["filled_cost"] - o.filled_cost
                dq = vo["cum_qty"] - o.cum_filled
                if dn <= 0 or dc < 0 or dn < dq:
                    self.latch_divergence("INVALID_VENUE_NOTIONAL")
                    return False
                self._apply_snapshot_fill(o.order_id, vo["cum_qty"], dn, dc)
                o = self._find(o.order_id)
            if vo["filled_notional"] != o.filled_notional or vo["filled_cost"] != o.filled_cost:
                self.latch_divergence("NOTIONAL_COST_MISMATCH")
                return False
            if o.status == "UNKNOWN" or vo["status"] == "UNKNOWN":
                self.flags.exec = "DIVERGENT"
                return False
            if o.terminal:
                if vo["status"] != o.status:
                    self.latch_divergence("TERMINAL_STATUS_MISMATCH")
                    return False
            elif vo["status"] in TERMINAL_VENUE:
                self._replace_order(o.order_id, status=vo["status"])

        self.flags.exec = "DIVERGENT" if self._divergence_latched else "OK"
        return not self._divergence_latched

    def _recompute(self) -> None:
        active = self.ledger.working_active()
        if not active:
            self.ledger.reserved = {c: 0 for c in CONSTRAINTS}
            if self.flags.envelope == "OVERFLOW":
                self.flags.envelope = "OK"
            return
        ev = evaluate(self.ledger.state, active, self.instruments, self.cfg.mandate,
                      self.scenarios, m_vertex=self.cfg.ops.m_vertex)
        if not ev.ok:
            self.flags.envelope = "OVERFLOW"
            self.log.append("OVERFLOW", {"reason": ev.reason}, self.clock.next())
            return
        self.ledger.reserved = reservations(ev)
        if self.flags.envelope == "OVERFLOW":
            self.flags.envelope = "OK"

    def recover(self) -> str:
        with self.lock:
            snap = self.venue.snapshot()
            if snap is None or not snap.get("complete", False):
                self.flags.exec = "DISCONNECTED"
                self._persist()
                return "BLOCKED_NO_AUTHORITATIVE_SNAPSHOT"
            unresolved = False
            for o in self.ledger.state.working:
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