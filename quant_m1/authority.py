"""M1.7 authorization kernel and execution gateway.

Changes over M1.6 (this repair pass):
  - P0 FIX: operator clearance uses an INDEPENDENT one-time key hash chain
    (OperatorConsole). The Gateway stores only sha256(current_operator_secret)
    and can no longer mint its own clear. The reveal is consumed and rotated;
    revealed secrets are never logged.
  - B1 FIX: Token binds ops_digest; submit rejects TOKEN_CONFIG_STALE on any
    operational-config change.
  - B2 FIX: composite binds Ledger.control_digest() (nonces, fills, reserved,
    session); tampering with replay-protection state is detected on restore.
  - B3 FIX: composite is HMAC-signed with the gateway key; an unsigned or
    altered composite latches COMPOSITE_SIGNATURE_INVALID.
  - FILL integrity: an event claiming cum_qty beyond the authoritative venue
    snapshot latches FILL_AHEAD_OF_VENUE.
  - Reconciliation: a venue order absent from the ledger latches
    UNEXPECTED_VENUE_ORDER.

Retained from M1.6/M1.5 (unchanged):
  - B15: divergence latch recovered from the audit chain, not the composite;
    missing composite with audit history is an integrity fault; composite/audit
    latch disagreement is a fault; audit chain is authoritative.
  - Q: submit() re-checks NEW_RISK for BUY and mark staleness at transmission.
  - S: restore_flags() verifies EventLog BEFORE head comparison; audit-head
    check uses chain membership; mismatch paths do not overwrite the composite.
  - CRASH_A / CRASH_B, RESERVATION_INTENT_MISMATCH, _replace_order raising
    UNKNOWN_ORDER, Kernel.authorize persisting before return, LEDGER_ONLY skip,
    timeout-stays-ambiguous, snapshot-authoritative fills.

Threat model: the gateway key protects tokens and the composite against parties
without the key. Full gateway-process compromise is out of scope, but the
operator hash chain specifically prevents a gateway-key holder from clearing a
divergence latch, because the Gateway never holds an operator secret.
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
OPERATOR_CLEAR_ACTION = "CLEAR_DIVERGENCE"


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
    ops_digest: str
    sig: str

    def body(self) -> dict:
        d = asdict(self)
        d.pop("sig")
        return d


def sign(key: bytes, body: dict) -> str:
    return hmac.new(key, sha(body).encode(), hashlib.sha256).hexdigest()


def operator_clear_body(reason: str, nonce: str, seq: int, next_hash: str) -> dict:
    """Canonical body for an operator clear command (one-time key + rotation)."""
    return {"action": OPERATOR_CLEAR_ACTION, "reason": reason,
            "nonce": nonce, "seq": seq, "next_hash": next_hash}


class OperatorConsole:
    """Independent operator authority holding the secret the Gateway never has.

    The Gateway stores only sha256(current_operator_secret). A clear reveals the
    current secret (hash-preimage proof) plus an HMAC over the body keyed by it,
    and the hash of the NEXT one-time secret. The Gateway verifies, consumes the
    reveal, and rotates to next_hash. A compromised Gateway holds only a hash and
    cannot forge a future clear. Revealed secrets are never logged. Rotation is a
    forward hash chain; each clear consumes one link.

    Key ownership: OperatorConsole alone holds secrets. Persistence contract: the
    Gateway persists only the current hash (signed composite + audit chain).
    """

    def __init__(self, secret: bytes | None = None):
        self._secret = secret if secret is not None else os.urandom(32)

    def current_hash(self) -> str:
        return hashlib.sha256(self._secret).hexdigest()

    def make_clear(self, reason: str, nonce: str, seq: int) -> dict:
        nxt = os.urandom(32)
        nh = hashlib.sha256(nxt).hexdigest()
        sig = sign(self._secret, operator_clear_body(reason, nonce, seq, nh))
        cmd = {"reason": reason, "nonce": nonce, "seq": seq,
               "key_reveal": self._secret.hex(), "next_hash": nh, "sig": sig}
        self._secret = nxt
        return cmd


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
                    ops_digest=cfg.ops.digest(),
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
                 watchdog_path: str, lock=None, flags_store: Store | None = None,
                 operator_hash: str | None = None):
        self.key, self.cfg, self.ledger, self.log = key, cfg, ledger, log
        self.venue, self.clock, self.instruments = venue, clock, instruments
        self.scenarios, self.flags, self.watchdog_path = scenarios, flags, watchdog_path
        self.fail_point: str | None = None
        self.lock = lock or threading.RLock()
        base_dir = os.path.dirname(watchdog_path) or "."
        self.flags_store = flags_store or Store(os.path.join(base_dir, "flags.json"))
        self.composite_store = Store(os.path.join(base_dir, "composite_state.json"))
        self._divergence_latched = False
        self._restore_fault = None
        self._operator_hash = operator_hash

    # ---- durable composite write: ledger + flags + latch + audit head ----

    def _persist(self) -> None:
        payload = {
            "schema_version": COMPOSITE_SCHEMA_VERSION,
            "ledger": self.ledger.to_dict(),
            "flags": asdict(self.flags),
            "divergence_latched": self._divergence_latched,
            "ledger_digest": self.ledger.state.digest(),
            "control_digest": self.ledger.control_digest(),
            "operator_hash": self._operator_hash,
            "audit_head": self.log.head(),
        }
        payload["composite_sig"] = sign(self.key, payload)
        self.composite_store.save(payload)

    def _latch_silent(self, reason: str) -> None:
        """Latch WITHOUT appending to audit and WITHOUT persisting.

        Used on restore failures, where appending would overwrite the very
        evidence that failed verification.
        """
        self._divergence_latched = True
        self.flags.exec = "DIVERGENT"
        self._restore_fault = reason

    def _latch_from_audit(self) -> bool:
        """Replay the audit chain to determine the authoritative latch state.

        A DIVERGENCE_LATCHED sets it; a DIVERGENCE_CLEARED clears it. The last
        one wins. This is independent of the composite file, so deleting the
        composite cannot erase a recorded latch.
        """
        latched = False
        for rec in self.log.chain:
            kind = rec.get("kind")
            if kind == "DIVERGENCE_LATCHED":
                latched = True
            elif kind == "DIVERGENCE_CLEARED":
                latched = False
        return latched

    def restore_flags(self) -> bool:
        """Restore composite state, verifying the audit chain FIRST.

        The audit chain is authoritative for the divergence latch. The
        composite is a convenience snapshot and must never override it.
        """
        self._restore_fault = None

        # 1. Restore and verify the audit chain before any head comparison.
        try:
            self.log.restore()
        except ValueError:
            self._latch_silent("AUDIT_CHAIN_VERIFY_FAILED")
            return True

        audit_latched = self._latch_from_audit()
        ah = self._operator_hash_from_audit()
        if ah is not None:
            self._operator_hash = ah
        has_history = bool(self.log.chain)

        d = self.composite_store.load()

        if d is None:
            # Legacy migration: ledger.json + flags.json.
            legacy_ledger = Ledger.restore(self.ledger.store)
            legacy_flags = self.flags_store.load()

            if legacy_ledger is None and not legacy_flags:
                # B15: prior activity with no composite is an integrity fault,
                # not a clean first start.
                if has_history:
                    self._latch_silent("COMPOSITE_MISSING_WITH_AUDIT_HISTORY")
                    return True
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

            # B15: the audit chain overrides any legacy flag value.
            if audit_latched:
                self._divergence_latched = True
                self.flags.exec = "DIVERGENT"

            if has_history:
                self._latch_silent("COMPOSITE_MISSING_WITH_AUDIT_HISTORY")
                return True

            self._persist()
            return True

        # 2. Schema mismatch: preserve the artifact, do not overwrite.
        if d.get("schema_version") != COMPOSITE_SCHEMA_VERSION:
            self._latch_silent("FLAGS_SCHEMA_UNKNOWN")
            return True

        supplied_sig = d.get("composite_sig")
        sig_body = {k: v for k, v in d.items() if k != "composite_sig"}
        if not isinstance(supplied_sig, str) or not hmac.compare_digest(
                supplied_sig, sign(self.key, sig_body)):
            self._latch_silent("COMPOSITE_SIGNATURE_INVALID")
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

        composite_latched = bool(d.get("divergence_latched", False))
        self._divergence_latched = composite_latched

        # B15: audit chain is authoritative. A composite claiming a clear
        # latch while the audit records a latch is a tamper/crash signal.
        if audit_latched:
            self._divergence_latched = True
            self.flags.exec = "DIVERGENT"
            if not composite_latched:
                self._restore_fault = "COMPOSITE_LATCH_DISAGREES_WITH_AUDIT"
        elif self._divergence_latched:
            self.flags.exec = "DIVERGENT"

        if d.get("ledger_digest") and d["ledger_digest"] != self.ledger.state.digest():
            self._latch_silent("FLAGS_LEDGER_DIGEST_MISMATCH")
            return True

        if d.get("control_digest") and d["control_digest"] != self.ledger.control_digest():
            self._latch_silent("FLAGS_CONTROL_DIGEST_MISMATCH")
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

    def _operator_hash_from_audit(self) -> str | None:
        """Authoritative current operator hash from the audit chain."""
        cur = None
        for rec in self.log.chain:
            if rec.get("kind") == "OPERATOR_REGISTERED":
                cur = rec.get("payload", {}).get("operator_hash")
            elif rec.get("kind") == "DIVERGENCE_CLEARED":
                nh = rec.get("payload", {}).get("next_hash")
                if nh:
                    cur = nh
        return cur

    def register_operator(self, operator_hash: str) -> None:
        """Register the hash of the independent operator's first one-time key."""
        with self.lock:
            if not isinstance(operator_hash, str) or len(operator_hash) != 64:
                raise Reject("OPERATOR_HASH_INVALID")
            self._operator_hash = operator_hash
            self.log.append("OPERATOR_REGISTERED",
                            {"operator_hash": operator_hash}, self.clock.next())
            self._persist()

    def clear_divergence(self, operator_reason: str, *, nonce: str | None = None,
                         seq: int | None = None, key_reveal: str | None = None,
                         next_hash: str | None = None, sig: str | None = None) -> None:
        """Clear a latch with an independent one-time operator command.

        The Gateway holds only the hash of the current operator secret, so it
        cannot self-clear. Requires hash-preimage possession, a valid HMAC over
        {action, reason, nonce, seq, next_hash}, the current seq, an unused nonce,
        and a valid rotation commitment. The reveal is never logged.
        """
        with self.lock:
            if not operator_reason or not operator_reason.strip():
                raise Reject("OPERATOR_REASON_REQUIRED")
            if (nonce is None or seq is None or key_reveal is None
                    or next_hash is None or sig is None):
                raise Reject("OPERATOR_AUTH_REQUIRED")
            if self._operator_hash is None:
                raise Reject("OPERATOR_NOT_REGISTERED")
            try:
                reveal = bytes.fromhex(key_reveal)
            except (ValueError, TypeError):
                raise Reject("OPERATOR_BAD_KEY")
            if hashlib.sha256(reveal).hexdigest() != self._operator_hash:
                raise Reject("OPERATOR_BAD_KEY")
            if not isinstance(next_hash, str) or len(next_hash) != 64:
                raise Reject("OPERATOR_BAD_NEXT_HASH")
            body = operator_clear_body(operator_reason, nonce, seq, next_hash)
            if not hmac.compare_digest(sig, sign(reveal, body)):
                raise Reject("OPERATOR_BAD_SIGNATURE")
            if seq != self.clock.seq:
                raise Reject("OPERATOR_SEQ_STALE")
            for rec in self.log.chain:
                if rec.get("kind") != "DIVERGENCE_CLEARED":
                    continue
                if rec.get("payload", {}).get("nonce") == nonce:
                    raise Reject("OPERATOR_NONCE_REPLAY")

            self._divergence_latched = False
            self.flags.exec = "OK"
            self._restore_fault = None
            self._operator_hash = next_hash
            self.log.append("DIVERGENCE_CLEARED",
                            {"reason": operator_reason, "nonce": nonce,
                             "operator_seq": seq, "next_hash": next_hash},
                            self.clock.next())
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

            # Q: re-check the FULL permission set at transmission.
            perms = self.flags.permissions()
            if "SUBMIT" not in perms:
                raise Reject("MODE_BLOCKS_SUBMIT")
            if intent.side == "BUY" and "NEW_RISK" not in perms:
                raise Reject("MODE_BLOCKS_NEW_RISK")

            # Q: re-check data freshness at transmission.
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
            if tok.ops_digest != self.cfg.ops.digest():
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
                if ev.cum_qty > vo["cum_qty"]:
                    self.latch_divergence("FILL_AHEAD_OF_VENUE")
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
                # Cancel rejected: the order remains live. No state change;
                # reconciliation below re-affirms the authoritative venue state.
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

        ledger_ids = {o.order_id for o in self.ledger.state.working}
        for vid in snap["orders"]:
            if vid not in ledger_ids:
                self.latch_divergence("UNEXPECTED_VENUE_ORDER")
                return False

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
