"""M1.4 authorization kernel and execution gateway.

Changes over M1.3:
  - Composite atomic persistence: ledger + flags + latch + audit head in ONE file.
  - Matching composite restore path reconstructs ledger AND flags together.
  - Legacy flags.json migration on first restore.
  - submit() rolls back RAM state if pre-send persistence fails.
  - process() refreshes local order after applying venue-authoritative fill.
  - Nonpositive sale proceeds logged or latched instead of vanishing silently.
  - Obsolete price-based _apply_fill helper removed.
  - CRASH_A / CRASH_B injection points preserved verbatim.
  - Reservation-intent binding preserved verbatim.
  - Sticky divergence latch preserved.

Safety rules:
- one shared lock serializes authorization, submission, events and recovery;
- every order is bound to an immutable intent, ledger snapshot AND configuration;
- nonce is durably consumed before transmission;
- ambiguous broker state never releases a reservation;
- fills use venue-authoritative notional, not the order limit price;
- cancellation/rejection is terminal only after broker snapshot confirmation;
- reconciliation failure is never reported as APPLIED;
- a latched divergence is never cleared automatically.
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

    def restore_flags(self) -> bool:
        """Restore composite state if present; otherwise migrate legacy files.

        Returns True if a persisted state was found and restored.
        On schema mismatch or digest mismatch, latches divergence.
        """
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

            # Migrate: write composite so next restore uses the new path.
            self._persist()
            return True

        if d.get("schema_version") != COMPOSITE_SCHEMA_VERSION:
            self.latch_divergence("FLAGS_SCHEMA_UNKNOWN")
            self._persist()
            return True

        ledger_data = d.get("ledger")
        if ledger_data is None:
            self.latch_divergence("COMPOSITE_LEDGER_MISSING")
            self._persist()
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

        if d.get("ledger_digest") and d["ledger_digest"] != self.ledger.state.digest():
            self.latch_divergence("FLAGS_LEDGER_DIGEST_MISMATCH")

        if d.get("audit_head") and d["audit_head"] != self.log.head():
            self.latch_divergence("AUDIT_HEAD_MISMATCH")

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
            if "SUBMIT" not in self.flags.permissions():
                raise Reject("MODE_BLOCKS_SUBMIT")
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

            # Rollback on pre-send persistence failure.
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
                    # Refresh local order after replacement; stale-o bug fix.
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
            return "APPLIED" if reconcHere are the two complete file replacements. Copy each one in full and overwrite the corresponding file in your repo.

**UNEXECUTED — NO RESULT CLAIMED.** I have read your current `main` branch via `raw.githubusercontent.com`. These are designed against that source. I have not run them. Run `pytest -q` after replacing and report the actual count.

---

## File 1: `quant_m1/risk.py`

```python
"""M1.5 deterministic reachable-fill risk evaluation. Per-SimState marks.

Changes over M1.4:
  - Per-constraint risk-reducing exemption replaces the all-or-nothing gate.
  - A newly breached constraint always rejects.
  - A pre-existing breach may be waived only when its worst reachable usage
    does not worsen (loss metrics floored at zero for the comparison).
  - Hard floors (short reachable, negative settled cash, non-positive NAV)
    are NEVER exempted.
  - factor_abs remains ABSOLUTE net beta exposure in dollars. The mandate
    calibration determines core-book reachability; no duplicate metric.

usage(), simulate(), CONSTRAINTS, LOSS_CONSTRAINTS, and is_risk_reducing()
are unchanged from M1.4 so that reference.py's oracle stays valid.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .core import (Money, Shares, BPS, Mandate, Instrument, WorkingOrder,
                   VerifiedState, ceil_bps, floor_bps)


@dataclass(frozen=True)
class SimState:
    positions: dict[str, Shares]
    cash_settled: Money
    receivable_total: Money
    nav: Money
    marks: dict[str, Money]


def simulate(base: VerifiedState, orders: tuple[WorkingOrder, ...],
             nums: tuple[int, ...], den: int, instruments: dict[str, Instrument],
             settle_lag: int) -> SimState:
    if den <= 0 or len(nums) != len(orders) or any(n < 0 or n > den for n in nums):
        raise ValueError("invalid fill-fraction vector")
    if settle_lag < 1:
        raise ValueError("settle_lag must be >= 1: T+0 would invalidate this model")
    pos = dict(base.positions)
    cash = base.cash_settled
    recv = base.receivable_total()
    for o, num in zip(orders, nums):
        q = o.remaining * num // den
        if q == 0:
            continue
        if o.price <= 0 or o.cost_bps < 0:
            raise ValueError("invalid working order economics")
        notional = q * o.price
        cost = ceil_bps(notional, o.cost_bps)
        if o.side == "BUY":
            pos[o.symbol] = pos.get(o.symbol, 0) + q
            cash -= notional + cost
        elif o.side == "SELL":
            if q > pos.get(o.symbol, 0):
                raise ValueError("reachable fill creates short position")
            pos[o.symbol] = pos.get(o.symbol, 0) - q
            recv += notional - cost
        else:
            raise ValueError("unknown order side")
    if any(s not in base.marks for s, q in pos.items() if q):
        raise ValueError("missing mark for nonzero position")
    nav = cash + recv + sum(q * base.marks[s] for s, q in pos.items() if q)
    return SimState(pos, cash, recv, nav, dict(base.marks))


def vertices(m: int):
    if m < 0:
        raise ValueError("negative order count")
    return itertools.product((0, 1), repeat=m)


def scenario_nav(sim: SimState, marks: dict[str, int]) -> Money:
    if any(s not in marks for s, q in sim.positions.items() if q):
        raise ValueError("scenario missing a held instrument mark")
    return sim.cash_settled + sim.receivable_total + sum(
        q * marks[s] for s, q in sim.positions.items() if q)


@dataclass(frozen=True)
class Usage:
    gross: Money
    issuer_max: Money
    factor_abs: Money
    worst_loss: Money
    cvar: Money
    neg_cash: Money


def usage(sim: SimState, base_nav: Money, instruments: dict[str, Instrument],
          scenarios: tuple[dict[str, int], ...], q_bps: int) -> Usage:
    if not scenarios:
        raise ValueError("at least one stress scenario is required")
    if not 1 <= q_bps <= BPS:
        raise ValueError("CVaR tail fraction must be in (0, 10000] bps")
    gross = 0
    issuer: dict[str, int] = {}
    factor = 0
    for sym, q in sim.positions.items():
        if q == 0:
            continue
        if sym not in instruments or sym not in sim.marks:
            raise ValueError(f"missing instrument or mark: {sym}")
        inst = instruments[sym]
        val = q * sim.marks[sym]
        gross += abs(val)
        for iss, w in inst.lookthrough():
            issuer[iss] = issuer.get(iss, 0) + ceil_bps(abs(val), w)
        factor += inst.beta_bps * val
    losses = sorted((base_nav - scenario_nav(sim, sc) for sc in scenarios), reverse=True)
    k = max(1, (len(losses) * q_bps + BPS - 1) // BPS)
    k = min(k, len(losses))
    cvar = sum(losses[:k]) // k
    return Usage(gross, max(issuer.values(), default=0), abs(factor) // BPS,
                 max(losses), cvar, max(0, -sim.cash_settled))


def set_marks(_marks: dict[str, Money]) -> None:
    """Deprecated compatibility shim. Marks belong to each SimState."""
    return None


CONSTRAINTS = ("gross", "issuer_max", "factor_abs", "worst_loss", "cvar", "neg_cash")
LOSS_CONSTRAINTS = frozenset({"worst_loss", "cvar"})


def is_risk_reducing(worst: dict[str, int], base: dict[str, int]) -> bool:
    """True when no constraint usage increases. Loss metrics floored at 0.

    Retained for diagnostics and the Evaluation.risk_reducing field.
    The evaluate() exemption below is per-constraint, not all-or-nothing.
    """
    for c in CONSTRAINTS:
        w, b = worst[c], base[c]
        if c in LOSS_CONSTRAINTS:
            w, b = max(0, w), max(0, b)
        if w > b:
            return False
    return True


@dataclass(frozen=True)
class Evaluation:
    ok: bool
    reason: str
    max_usage: dict[str, int]
    base_usage: dict[str, int]
    nav_min: Money
    risk_reducing: bool
    n_vertices: int


def evaluate(base: VerifiedState, orders: tuple[WorkingOrder, ...],
             instruments: dict[str, Instrument], mandate: Mandate,
             scenarios: tuple[dict[str, int], ...],
             *, m_vertex: int = 6) -> Evaluation:
    m = len(orders)
    if m > m_vertex:
        return Evaluation(False, "M_VERTEX_EXCEEDED", {}, {}, 0, False, 0)
    if not scenarios:
        return Evaluation(False, "NO_STRESS_SCENARIOS", {}, {}, 0, False, 0)
    try:
        nav0 = base.nav()
        base_sim = simulate(base, orders, (0,) * m, 1, instruments, mandate.settle_lag_days)
        bu = usage(base_sim, nav0, instruments, scenarios, mandate.cvar_q_bps)
        base_d = {c: getattr(bu, c) for c in CONSTRAINTS}
        max_d = {c: -10**30 for c in CONSTRAINTS}
        nav_min = 10**30
        n = 0
        for v in vertices(m):
            n += 1
            sim = simulate(base, orders, v, 1, instruments, mandate.settle_lag_days)
            # HARD FLOORS: never exempted, even for risk-reducing trades.
            if any(q < 0 for q in sim.positions.values()):
                return Evaluation(False, "SHORT_REACHABLE", {}, {}, 0, False, n)
            if sim.cash_settled < 0:
                return Evaluation(False, "SETTLED_CASH_NEGATIVE", {}, {}, 0, False, n)
            nav_min = min(nav_min, sim.nav)
            u = usage(sim, nav0, instruments, scenarios, mandate.cvar_q_bps)
            for c in CONSTRAINTS:
                max_d[c] = max(max_d[c], getattr(u, c))
        if nav_min <= 0:
            return Evaluation(False, "NAV_NONPOSITIVE", max_d, base_d, nav_min, False, n)

        reducing = is_risk_reducing(max_d, base_d)

        checks = (
            ("gross", max_d["gross"], floor_bps(nav_min, mandate.gross_bps), "GROSS"),
            ("issuer_max", max_d["issuer_max"], floor_bps(nav_min, mandate.issuer_bps), "ISSUER"),
            ("factor_abs", max_d["factor_abs"], floor_bps(nav_min, mandate.factor_bps), "FACTOR"),
            ("worst_loss", max_d["worst_loss"], floor_bps(nav0, mandate.worst_loss_bps), "WORST_LOSS"),
            ("cvar", max_d["cvar"], floor_bps(nav0, mandate.cvar_bps), "CVAR"),
        )

        # 1. Any NEW breach rejects immediately, regardless of other dimensions.
        for c, got, lim, name in checks:
            if got > lim and base_d[c] <= lim:
                return Evaluation(False, f"LIMIT_{name}", max_d, base_d, nav_min, False, n)

        # 2. Pre-existing breaches may be waived only if not worsening.
        for c, got, lim, name in checks:
            if got > lim:
                w, b = max_d[c], base_d[c]
                if c in LOSS_CONSTRAINTS:
                    w, b = max(0, w), max(0, b)
                if w > b:
                    return Evaluation(False, f"LIMIT_{name}", max_d, base_d, nav_min, False, n)
                # Pre-existing breach, worst reachable usage did not worsen: waive.
                continue

        return Evaluation(True, "OK", max_d, base_d, nav_min, reducing, n)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return Evaluation(False, f"INVALID_RISK_INPUT:{type(exc).__name__}", {}, {}, 0, False, 0)


def reservations(ev: Evaluation) -> dict[str, int]:
    if not ev.ok:
        raise ValueError("cannot reserve against a failed risk evaluation")
    return {c: max(0, ev.max_usage[c] - ev.base_usage[c]) for c in CONSTRAINTS}
