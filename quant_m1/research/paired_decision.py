"""Paired Counterfactual Decision Ledger (PCDL) - measurement only.

WHAT THIS IS
  Exact, reproducible ACCOUNTING of the utility difference between the active
  policy and injected comparator policies, CONDITIONAL ON THE PAPER SIMULATOR
  (limit-price fills, no market impact, no queue position, mock venue).

WHAT THIS IS NOT
  An exact real-world counterfactual. Pairing does not eliminate model risk,
  hypothetical-fill error, data bias, or uncertainty about how the alternative
  policy would have behaved in a live market.

PAIRING
  Removes the COMMON exposure component from the comparison. It does NOT make
  the differential independent of market returns: when books hold different
  positions, the P&L of that position difference IS the decision effect and
  remains market-driven. For equal variances Var(A-C) = 2*sigma^2*(1-rho);
  versus an independent two-sample comparison the variance-reduction factor is
  1/(1-rho). Pairing improves precision per episode; it does NOT create
  additional independent episodes.

NO DEFAULTS
  Utility, comparator set, horizon and dependence-block rule are injected and
  required. This module defines no objective, chooses no comparators, infers no
  regime blocks and makes no acceptance decision. A spec digest prevents silent
  drift WITHIN a spec; it does not license trying many specs. Every attempted
  specification must enter the existing trial-governance process and be retained
  in the hypothesis family.

CLEAN-START PRECONDITION
  The existing shadow constructor restores economic state but NOT working
  orders, reservations or replay-protection state. Episodes therefore require
  no active working orders and no outstanding reservations at start. Attempts
  violating this are recorded INVALID and never silently compared.

REPRODUCIBILITY
  Equality of canonical ECONOMIC outputs (integer NAV, cash, receivables,
  positions, filled cost/notional, turnover, session), NOT equality of audit
  records: order IDs and nonces are uuid4 and differ across replays. The runtime
  manifest records interpreter and library versions because requirements are
  minimum-version, not pinned.

SIMULATOR NOTE / EPISODE STRUCTURE
  Trades execute at the session's marks, so a DECISION-ONLY episode cannot
  realise directional P&L: its differential is dominated by costs. Moreover
  every comparator starts from the ACTIVE pre-state, so per-episode pairing
  measures the incremental decision from a COMMON state, never cumulative path
  divergence. To measure holding-period decision value, supply `valuation`:
  after the decision phase, BOTH books receive a hold-only session advance to
  the horizon marks. That realise the exposure taken during the decision phase.
  Without `valuation` the episode is decision-only and must be interpreted as
  cost/turnover/settlement measurement, not directional value.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from ..core import sha
from ..orchestrator.paper_loop import run_paper_step, _shadow_system

SPEC_SCHEMA_VERSION = 1
Comparator = Callable[[Mapping, Mapping, Mapping], tuple]


def runtime_manifest(solver: str = "SLSQP", extra: Optional[Mapping] = None) -> dict:
    import numpy, scipy, pandas
    m = {"python": platform.python_version(), "numpy": numpy.__version__,
         "scipy": scipy.__version__, "pandas": pandas.__version__,
         "solver": solver}
    if extra:
        m.update(dict(extra))
    return m


@dataclass(frozen=True)
class PairedSpec:
    """Pre-registered measurement specification. No defaults are supplied."""
    utility: Callable = field(repr=False, compare=False)
    utility_id: str = ""
    comparators: Mapping = field(repr=False, compare=False, default_factory=dict)
    horizon_id: str = ""
    block_rule_id: str = ""
    manifest: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.utility_id or not self.horizon_id or not self.block_rule_id:
            raise ValueError("utility_id, horizon_id and block_rule_id are required")
        if not self.comparators:
            raise ValueError("a frozen comparator set is required")
        if not self.manifest:
            raise ValueError("a runtime manifest is required")

    def identity(self) -> dict:
        return {"schema": SPEC_SCHEMA_VERSION, "utility_id": self.utility_id,
                "comparators": sorted(self.comparators), "horizon_id": self.horizon_id,
                "block_rule_id": self.block_rule_id, "manifest": dict(self.manifest)}

    def digest(self) -> str:
        return sha(self.identity())


@dataclass(frozen=True)
class BookResult:
    name: str
    start_nav: int
    end_nav: int
    cash_settled: int
    receivables: tuple
    positions: tuple
    filled_cost: int
    filled_notional: int
    turnover_shares: int
    trade_date: int
    mark_wall: int
    last_settled_session: int
    blocked: bool = False
    invalid_reason: str = ""

    def economic_payload(self) -> dict:
        return {"name": self.name, "start_nav": self.start_nav, "end_nav": self.end_nav,
                "cash_settled": self.cash_settled, "receivables": list(self.receivables),
                "positions": list(self.positions), "filled_cost": self.filled_cost,
                "filled_notional": self.filled_notional,
                "turnover_shares": self.turnover_shares, "trade_date": self.trade_date,
                "mark_wall": self.mark_wall,
                "last_settled_session": self.last_settled_session,
                "blocked": self.blocked, "invalid_reason": self.invalid_reason}

    def economic_digest(self) -> str:
        return sha(self.economic_payload())


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    spec_digest: str
    start: Mapping
    books: Mapping
    differentials: Mapping
    opportunity_cost: Mapping
    active_took_no_action: bool
    valid: bool
    invalid_reason: str = ""

    def payload(self) -> dict:
        return {"episode_id": self.episode_id, "spec_digest": self.spec_digest,
                "start": dict(self.start),
                "books": {k: v.economic_payload() for k, v in self.books.items()},
                "differentials": dict(self.differentials),
                "opportunity_cost": dict(self.opportunity_cost),
                "active_took_no_action": self.active_took_no_action,
                "valid": self.valid, "invalid_reason": self.invalid_reason}


def capture_start(sx) -> dict:
    led = sx["ledger"]
    return {"cash_settled": led.state.cash_settled,
            "positions": dict(led.state.positions),
            "marks": dict(led.state.marks),
            "receivables": tuple(led.state.receivables),
            "peak_nav": led.state.peak_nav,
            "trade_date": led.state.trade_date,
            "mark_wall": led.state.mark_wall,
            "last_settled_session": led.last_settled_session}


def clean_start(sx) -> tuple:
    led = sx["ledger"]
    if led.working_active():
        return False, "ACTIVE_WORKING_ORDERS_AT_START"
    if any(v != 0 for v in led.reserved.values()):
        return False, "OUTSTANDING_RESERVATIONS_AT_START"
    return True, ""


def _book(name: str, sx, start_nav: int, blocked: bool, reason: str = "") -> BookResult:
    led = sx["ledger"]
    orders = led.state.working
    return BookResult(
        name=name, start_nav=start_nav, end_nav=led.state.nav(),
        cash_settled=led.state.cash_settled,
        receivables=tuple(sorted(led.state.receivables)),
        positions=tuple(sorted((k, v) for k, v in led.state.positions.items() if v)),
        filled_cost=sum(o.filled_cost for o in orders),
        filled_notional=sum(o.filled_notional for o in orders),
        turnover_shares=sum(o.cum_filled for o in orders),
        trade_date=led.state.trade_date, mark_wall=led.state.mark_wall,
        last_settled_session=led.last_settled_session,
        blocked=blocked, invalid_reason=reason)


def hold_current(start: Mapping, trackers: Mapping, forecasts: Mapping) -> tuple:
    """Carry existing positions; submit no orders. NOT liquidation."""
    return {}, {}


def liquidate_to_cash(start: Mapping, trackers: Mapping, forecasts: Mapping) -> tuple:
    """Target zero weights: sells existing holdings, pays costs, T+1 receivables.

    Authority is unchanged (the active tracker set is reused); only the SIGNAL is
    zeroed. An eligible tracker set is required because the decision cycle
    produces no allocation at all when no model is eligible -- zeroing forecasts
    alone would silently mean HOLD, not liquidation. No authority is fabricated.
    """
    import numpy as np
    if not trackers:
        raise ValueError("liquidate_to_cash requires an eligible tracker set")
    f = dict(forecasts)
    if "model_signals" in f:
        f["model_signals"] = {k: np.zeros_like(np.asarray(v, dtype=float))
                              for k, v in f["model_signals"].items()}
        f.pop("expected_returns", None)
    elif "expected_returns" in f:
        f["expected_returns"] = np.zeros_like(np.asarray(f["expected_returns"], dtype=float))
    else:
        raise ValueError("liquidate_to_cash requires expected_returns or model_signals")
    return trackers, f


def run_episode(sx, market_state, trackers, forecasts, *, new_trade_date: int,
                new_mark_wall: int, new_marks: Mapping, allocator: Callable,
                spec: PairedSpec, episode_id: str,
                min_trade_bps: int = 10, cost_bps: int = 10,
                valuation: Optional[tuple] = None) -> EpisodeRecord:
    """Run the active book and every comparator from an identical clean start.

    `valuation` = (val_trade_date, val_mark_wall, val_marks). When supplied, both
    books receive a hold-only session advance to the horizon marks after the
    decision phase, realising the exposure taken. Two sessions are consumed.
    """
    import shutil, tempfile
    start = capture_start(sx)
    ok, why = clean_start(sx)
    digest = spec.digest()
    if not ok:
        return EpisodeRecord(episode_id, digest, start, {}, {}, {}, False, False, why)

    books, diffs, opp = {}, {}, {}
    invalid = ""
    _start_nav = sx["ledger"].state.nav()
    step = run_paper_step(sx, market_state, trackers, forecasts,
                          new_trade_date=new_trade_date, new_mark_wall=new_mark_wall,
                          new_marks=new_marks, allocator=allocator,
                          min_trade_bps=min_trade_bps, cost_bps=cost_bps)
    if valuation is not None and not step.blocked:
        vd, vw, vm = valuation
        step = run_paper_step(sx, market_state, {}, {}, new_trade_date=vd,
                              new_mark_wall=vw, new_marks=vm, allocator=allocator,
                              min_trade_bps=min_trade_bps, cost_bps=cost_bps)
    active = _book("active", sx, _start_nav, step.blocked, step.note if step.blocked else "")
    books["active"] = active
    if step.blocked:
        invalid = "ACTIVE_BOOK_BLOCKED"

    for name, cmp_fn in spec.comparators.items():
        tmp = tempfile.mkdtemp(prefix="pcdl_")
        try:
            sh = _shadow_system(sx, start, tmp)
            sh["ledger"].last_settled_session = start["last_settled_session"]
            c_trackers, c_forecasts = cmp_fn(start, trackers, forecasts)
            c_step = run_paper_step(sh, market_state, c_trackers, c_forecasts,
                                    new_trade_date=new_trade_date,
                                    new_mark_wall=new_mark_wall, new_marks=new_marks,
                                    allocator=allocator, min_trade_bps=min_trade_bps,
                                    cost_bps=cost_bps)
            if valuation is not None and not c_step.blocked:
                vd, vw, vm = valuation
                c_step = run_paper_step(sh, market_state, {}, {}, new_trade_date=vd,
                                        new_mark_wall=vw, new_marks=vm,
                                        allocator=allocator,
                                        min_trade_bps=min_trade_bps, cost_bps=cost_bps)
            br = _book(name, sh, _start_nav, c_step.blocked,
                       c_step.note if c_step.blocked else "")
        except Exception as exc:
            br = BookResult(name, active.start_nav, active.start_nav, 0, (), (), 0, 0, 0,
                            new_trade_date, new_mark_wall, start["last_settled_session"],
                            True, f"COMPARATOR_ERROR:{type(exc).__name__}")
            invalid = invalid or f"COMPARATOR_FAILED:{name}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        books[name] = br
        if br.blocked and not invalid:
            invalid = f"COMPARATOR_BLOCKED:{name}"

    if not invalid:
        u_active = float(spec.utility(active))
        for name, br in books.items():
            if name == "active":
                continue
            d = u_active - float(spec.utility(br))
            diffs[name] = d
            opp[name] = -d
    no_action = active.turnover_shares == 0
    return EpisodeRecord(episode_id, digest, start, books, diffs, opp,
                         no_action, not invalid, invalid)


def verify_reproducibility(record: EpisodeRecord, sx, market_state, trackers, forecasts,
                           *, new_trade_date: int, new_mark_wall: int, new_marks: Mapping,
                           allocator: Callable, spec: PairedSpec,
                           min_trade_bps: int = 10, cost_bps: int = 10,
                           valuation: Optional[tuple] = None) -> tuple:
    """Replay EVERY book (including active) in shadows from the recorded start.

    Compares canonical ECONOMIC digests exactly. Random order IDs/nonces are
    excluded by construction. A mismatch also indicates an incomplete start
    snapshot, so this doubles as a check on the shadow-constructor contract.
    """
    import shutil, tempfile
    if not record.valid:
        return False, {"episode": "INVALID_EPISODE_NOT_REPLAYABLE"}
    start = dict(record.start)
    policies = {"active": (trackers, forecasts)}
    for name, fn in spec.comparators.items():
        policies[name] = fn(start, trackers, forecasts)
    mismatches = {}
    for name, (tr, fc) in policies.items():
        tmp = tempfile.mkdtemp(prefix="pcdl_replay_")
        try:
            sh = _shadow_system(sx, start, tmp)
            sh["ledger"].last_settled_session = start["last_settled_session"]
            _sn = sh["ledger"].state.nav()
            st = run_paper_step(sh, market_state, tr, fc,
                                new_trade_date=new_trade_date,
                                new_mark_wall=new_mark_wall, new_marks=new_marks,
                                allocator=allocator, min_trade_bps=min_trade_bps,
                                cost_bps=cost_bps)
            if valuation is not None and not st.blocked:
                vd, vw, vm = valuation
                st = run_paper_step(sh, market_state, {}, {}, new_trade_date=vd,
                                    new_mark_wall=vw, new_marks=vm,
                                    allocator=allocator,
                                    min_trade_bps=min_trade_bps, cost_bps=cost_bps)
            br = _book(name, sh, _sn, st.blocked, st.note if st.blocked else "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if br.economic_digest() != record.books[name].economic_digest():
            mismatches[name] = "ECONOMIC_OUTPUT_MISMATCH"
    return (not mismatches), mismatches


class PairedLedger:
    """Append-only hash-chained episode records. Invalid attempts are retained."""

    def __init__(self, path: str):
        self.path = path
        self._chain = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._chain.append(json.loads(line))

    def append(self, record: EpisodeRecord) -> str:
        prev = self._chain[-1]["hash"] if self._chain else "0" * 64
        body = {"seq": len(self._chain), "prev": prev, "record": record.payload()}
        h = sha(body)
        entry = {**body, "hash": h}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._chain.append(entry)
        return h

    def records(self) -> tuple:
        return tuple(e["record"] for e in self._chain)

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for i, e in enumerate(self._chain):
            body = {"seq": e["seq"], "prev": e["prev"], "record": e["record"]}
            if e["seq"] != i or e["prev"] != prev or sha(body) != e["hash"]:
                return False
            prev = e["hash"]
        return True
