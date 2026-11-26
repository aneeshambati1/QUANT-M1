"""Restartable paper-session daemon (O-07) with venue reconciliation (O-08).

This is a BOUNDED, IDEMPOTENT, RESTARTABLE batch runner -- not a long-lived
process holding state. An external scheduler (cron/systemd) invokes `run` with
a supplied session schedule. The daemon never invents market data, never
bypasses the Risk Kernel, and never routes live orders. PAPER / RESEARCH ONLY.

O-08 VENUE RECONCILIATION
  A real broker's execution IDs are globally unique and never reused across
  sessions. The mock venue's `_eid` counter resets on construction, so a naive
  restart re-issues E000001... which the restored ledger already recorded with
  different payloads. The Gateway correctly latches EXEC_ID_PAYLOAD_CONFLICT.
  That is the replay protection WORKING, not a bug to weaken.

  The fix is to make the simulated venue faithful: the daemon persists the
  venue's execution sequence in its own checkpoint and restores it on restart,
  then rebuilds the venue's economic state from the restored ledger and calls
  Gateway.recover(). Replay protection in core/authority/venue is UNCHANGED.

  A tampered checkpoint that lowers the sequence causes ID reuse, which the
  Gateway detects and latches -- fail closed, exactly as before.

HALTING
  Any divergence latch, reconciliation failure, or step exception HALTS the
  daemon and persists the halt. Subsequent runs refuse to trade until the
  operator clears the divergence through the existing independent operator
  path and the halt is explicitly reset. The daemon cannot clear its own halt.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

from ..core import Store
from ..system import build
from .paper_loop import run_paper_step

CHECKPOINT_SCHEMA = 1


@dataclass
class DaemonReport:
    sessions_attempted: int = 0
    sessions_completed: int = 0
    halted: bool = False
    halt_reason: str = ""
    reports: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"sessions_attempted": self.sessions_attempted,
                "sessions_completed": self.sessions_completed,
                "halted": self.halted, "halt_reason": self.halt_reason,
                "n_reports": len(self.reports)}


def _blocked_of(rep) -> bool:
    b = getattr(rep, "blocked", None)
    if b is not None:
        return bool(b)
    inner = getattr(rep, "step", None)
    if inner is not None:
        return bool(getattr(inner, "blocked", False))
    valid = getattr(rep, "valid", None)
    if valid is not None:
        return not bool(valid)
    return False


class PaperDaemon:
    """Bounded, restartable paper-session runner."""

    def __init__(self, workdir: str, *, cash: int = 10_000_000,
                 marks: Optional[Mapping] = None,
                 step_fn: Optional[Callable] = None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self._store = Store(os.path.join(workdir, "daemon_checkpoint.json"))
        self.sx = build(workdir, cash=cash, positions={}, marks=marks)
        self._step_fn = step_fn or run_paper_step
        self.halted = False
        self.halt_reason = ""
        self.sessions_run = 0
        self.reconcile_status = ""

        ck = self._store.load()
        self._eid = 0
        self._seq = 0
        if ck is not None:
            if ck.get("schema") != CHECKPOINT_SCHEMA:
                self.halted = True
                self.halt_reason = "CHECKPOINT_SCHEMA_UNKNOWN"
            else:
                self._eid = int(ck.get("venue_eid", 0))
                self._seq = int(ck.get("venue_seq", 0))
                self.sessions_run = int(ck.get("sessions_run", 0))
                if ck.get("halted"):
                    self.halted = True
                    self.halt_reason = str(ck.get("halt_reason", "PRIOR_HALT"))
        self._reconcile_venue()

    def _exec_seq_floor(self) -> int:
        """Highest numeric suffix among already-applied simulated exec IDs.

        The simulated venue issues IDs of the form 'E%06d'. A restored sequence
        below this floor would re-issue an ID the ledger already recorded, so the
        daemon refuses to start rather than waiting for a collision to be caught
        downstream. IDs not matching the simulated format are ignored.
        """
        floor = 0
        for eid in self.sx["ledger"].applied_exec_ids:
            mt = re.fullmatch(r"E(\d+)", str(eid))
            if mt:
                floor = max(floor, int(mt.group(1)))
        return floor

    def _reconcile_venue(self) -> None:
        """Rebuild venue economic state from the restored ledger; restore exec seq."""
        led = self.sx["ledger"]
        v = self.sx["venue"]
        floor = self._exec_seq_floor()
        if self._eid < floor:
            self.halted = True
            self.halt_reason = f"CHECKPOINT_EXEC_SEQ_REGRESSION:{self._eid}<{floor}"
            self.reconcile_status = "SKIPPED_EXEC_SEQ_REGRESSION"
            self._persist()
            return
        v.orders.clear()
        v.outbox.clear()
        v.positions = dict(led.state.positions)
        v.cash = led.state.cash_settled + led.state.receivable_total()
        v._eid = self._eid
        v._seq = self._seq
        if self.halted:
            self.reconcile_status = "SKIPPED_HALTED"
            return
        status = self.sx["gw"].recover()
        self.reconcile_status = status
        if status != "RECOVERED":
            self.halted = True
            self.halt_reason = f"RECONCILE_FAILED:{status}"
            self._persist()

    def _persist(self) -> None:
        v = self.sx["venue"]
        self._store.save({
            "schema": CHECKPOINT_SCHEMA,
            "venue_eid": v._eid, "venue_seq": v._seq,
            "last_trade_date": self.sx["ledger"].state.trade_date,
            "last_mark_wall": self.sx["ledger"].state.mark_wall,
            "sessions_run": self.sessions_run,
            "halted": self.halted, "halt_reason": self.halt_reason,
        })

    def reset_halt(self, *, operator_reason: str) -> None:
        """Explicitly reset a daemon halt. Requires the divergence to be cleared
        first through the existing independent operator path; this method cannot
        clear a Gateway divergence latch."""
        if not operator_reason or not operator_reason.strip():
            raise ValueError("operator_reason required")
        if self.sx["gw"]._divergence_latched:
            raise ValueError("GATEWAY_STILL_DIVERGENT: clear via operator path first")
        self.halted = False
        self.halt_reason = ""
        self._reconcile_venue()
        self._persist()

    def run(self, sessions: Iterable, trackers: Mapping, *, allocator: Callable,
            limit: Optional[int] = None, min_trade_bps: int = 10,
            cost_bps: int = 10) -> DaemonReport:
        """Run supplied sessions in order. Stops fail-closed on any block.

        `sessions` yields (market_state, forecasts, trade_date, mark_wall, marks).
        The daemon does not generate market data.
        """
        out = DaemonReport()
        if self.halted:
            out.halted = True
            out.halt_reason = self.halt_reason
            return out
        for (ms, fc, date, wall, mk) in sessions:
            if limit is not None and out.sessions_attempted >= limit:
                break
            out.sessions_attempted += 1
            try:
                rep = self._step_fn(self.sx, ms, trackers, fc,
                                    new_trade_date=date, new_mark_wall=wall,
                                    new_marks=mk, allocator=allocator,
                                    min_trade_bps=min_trade_bps, cost_bps=cost_bps)
            except Exception as exc:
                self.halted = True
                self.halt_reason = f"STEP_ERROR:{type(exc).__name__}"
                self._persist()
                break
            out.reports.append(rep)
            if _blocked_of(rep) or self.sx["gw"]._divergence_latched:
                self.halted = True
                self.halt_reason = "SESSION_BLOCKED_OR_DIVERGENT"
                self._persist()
                break
            out.sessions_completed += 1
            self.sessions_run += 1
            self._persist()
        out.halted = self.halted
        out.halt_reason = self.halt_reason
        return out
