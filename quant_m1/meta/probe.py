"""Phase 2 research-only probe attribution (v3 + audit repairs 3A/3C/3D).

Repairs over the first implementation:
  3A. Probe observations go to a SEPARATE horizon-keyed research journal
      (ProbeJournal), NEVER into the live ContributionTracker. The candidate's
      measurement procedure is therefore identical for all 399 observations,
      regardless of any status change under the original kappa rule. The
      certificate is evaluated from the journal via a dedicated tracker.
  3C. Family assignments are PERSISTED in the journal and bound to the
      promotion ledger at registration. A family closed by a failed
      certificate or terminal event stays closed; later correlation drift
      cannot reopen it.
  3D. Probes REFUSE to run when the real system is frozen, divergent, killed,
      or has unresolved working orders. Blocked shadow outcomes are rejected
      (no observation is recorded from a blocked pair).

ISOLATION: disposable shadows with own ledger/gateway/MockVenue/workdir; the
real system digest is guarded before/after. Probe decisions never reach the
real gateway or any live endpoint. Probe coefficients never become live
weights. _ProbeGate is a counterfactual construct used ONLY inside shadow
tracker maps.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Mapping, Optional

import numpy as np

from ..core import Store, sha
from ..orchestrator.paper_loop import _shadow_system, run_paper_step
from .contribution import ContributionTracker

PI_PROBE = 0.05
RHO_DUP = 0.95


class _ProbeGate:
    """Research-only counterfactual gate. NEVER used in real tracker maps."""

    def __init__(self, horizon_days: int):
        self.horizon_days = horizon_days
        self.eligible = True

    def raw_weight(self) -> float:
        return 1.0

    def snapshot(self) -> dict:
        return {"probe_gate": True, "horizon_days": self.horizon_days}


class ProbeJournal:
    """Durable, horizon-keyed research history of paired probe observations.

    One consistent measurement procedure per (candidate, horizon). Hash-chained.
    Family assignments are persisted here and never recomputed from scratch.
    """

    def __init__(self, path: str):
        self.store = Store(path)
        self.chain = self.store.load() or []
        self._verify()
        self.observations: dict = {}   # (candidate,horizon) -> [float]
        self.families: dict = {}       # candidate -> family (persistent)
        for rec in self.chain:
            p = rec["payload"]
            if rec["kind"] == "FAMILY":
                self.families[p["candidate"]] = p["family"]
            elif rec["kind"] == "OBS":
                key = (p["candidate"], p["horizon"])
                self.observations.setdefault(key, []).append(p["value"])

    def _verify(self) -> None:
        prev = "0" * 64
        for rec in self.chain:
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev") != prev or rec.get("hash") != sha(body):
                raise ValueError("PROBE_JOURNAL_CHAIN_INVALID")
            prev = rec["hash"]

    def _append(self, kind: str, payload: dict) -> None:
        prev = self.chain[-1]["hash"] if self.chain else "0" * 64
        rec = {"kind": kind, "payload": payload, "prev": prev}
        rec["hash"] = sha(rec)
        self.chain.append(rec)
        self.store.save(self.chain)

    def set_family(self, candidate: str, family: str) -> None:
        if candidate in self.families:
            if self.families[candidate] != family:
                raise ValueError("FAMILY_ALREADY_ASSIGNED_IMMUTABLE")
            return
        self.families[candidate] = family
        self._append("FAMILY", {"candidate": candidate, "family": family})

    def record(self, candidate: str, horizon: int, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("observation must be finite")
        self.observations.setdefault((candidate, horizon), []).append(float(value))
        self._append("OBS", {"candidate": candidate, "horizon": horizon,
                             "value": float(value)})

    def n_obs(self, candidate: str, horizon: int) -> int:
        return len(self.observations.get((candidate, horizon), []))

    def certificate_tracker(self, candidate: str, horizon: int) -> ContributionTracker:
        """Rebuild a dedicated tracker from the journal for the ONE fixed
        certificate evaluation. Never the live tracker."""
        t = ContributionTracker(horizon_days=horizon)
        for v in self.observations.get((candidate, horizon), []):
            t.update(v, horizon_days=horizon)
        return t


def assign_families(signals: Mapping[str, np.ndarray],
                    rho_dup: float = RHO_DUP,
                    existing: Optional[Mapping[str, str]] = None) -> dict:
    """Deterministic family assignment honoring PERSISTED assignments.

    Existing assignments are immutable. New candidates join the first earlier
    representative (sorted IDs) with Pearson correlation >= rho_dup computed
    on the CURRENT time-aligned signal window; once assigned, never changed.
    """
    fam = dict(existing) if existing else {}
    names = sorted(signals)
    reps = sorted({fam[c] for c in names if c in fam})
    for name in names:
        if name in fam:
            continue
        v = np.asarray(signals[name], dtype=float)
        placed = False
        for r in reps:
            if r not in signals:
                continue
            u = np.asarray(signals[r], dtype=float)
            if v.std() > 0 and u.std() > 0:
                c = float(np.corrcoef(u, v)[0, 1])
                if np.isfinite(c) and c >= rho_dup:
                    fam[name] = r
                    placed = True
                    break
        if not placed:
            fam[name] = name
            reps.append(name)
            reps.sort()
    return fam


def probe_coefficient(candidate: str, families: Mapping[str, str],
                      active: Mapping[str, float], pi: float = PI_PROBE) -> float:
    rep = families.get(candidate, candidate)
    if candidate != rep:
        return 0.0
    if rep in active and active[rep] > 0:
        return 0.0
    return pi


def _start_snapshot(sx) -> dict:
    st = sx["ledger"].state
    return {"cash_settled": st.cash_settled, "positions": dict(st.positions),
            "marks": dict(st.marks), "receivables": tuple(st.receivables),
            "peak_nav": st.peak_nav, "trade_date": st.trade_date,
            "mark_wall": st.mark_wall}


def assert_probe_ready(sx) -> None:
    """Refuse probes when the real system is not in a clean operational state."""
    gw, flags, ledger = sx["gw"], sx["flags"], sx["ledger"]
    if gw._divergence_latched:
        raise RuntimeError("PROBE_REFUSED_DIVERGENT")
    if flags.killed or flags.frozen:
        raise RuntimeError("PROBE_REFUSED_KILLED_OR_FROZEN")
    if flags.exec != "OK":
        raise RuntimeError("PROBE_REFUSED_EXEC_NOT_OK")
    if ledger.working_active():
        raise RuntimeError("PROBE_REFUSED_UNRESOLVED_WORKING_ORDERS")


def probe_pair(real_sx, market_state, base_fc: Mapping, mu_active: np.ndarray,
               probe_blend: np.ndarray, *, new_trade_date: int,
               new_mark_wall: int, new_marks: Mapping[str, int],
               allocator, min_trade_bps: int, cost_bps: int,
               horizon_days: int = 1) -> float:
    """Paired after-cost observation: (NAV_probe - NAV_base)/NAV_start.
    Both shadows run the full paper path. Blocked outcomes are REJECTED."""
    assert_probe_ready(real_sx)
    start = _start_snapshot(real_sx)
    if start["cash_settled"] + sum(
            q * start["marks"][s] for s, q in start["positions"].items() if q) <= 0:
        raise ValueError("starting NAV must be positive")
    real_digest = real_sx["ledger"].state.digest()
    gate = {"__probe_gate__": _ProbeGate(horizon_days)}
    navs = []
    for blend in (mu_active, probe_blend):
        tmp = tempfile.mkdtemp(prefix="quant_probe_")
        try:
            shadow = _shadow_system(real_sx, start, tmp)
            fc = {**base_fc, "expected_returns": np.asarray(blend, dtype=float)}
            fc.pop("model_signals", None)
            rep = run_paper_step(
                shadow, market_state, gate, fc,
                new_trade_date=new_trade_date, new_mark_wall=new_mark_wall,
                new_marks=new_marks, allocator=allocator,
                min_trade_bps=min_trade_bps, cost_bps=cost_bps,
            )
            if rep.blocked:
                raise RuntimeError("PROBE_SHADOW_BLOCKED_NO_OBSERVATION")
            navs.append(rep.model_after)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if real_sx["ledger"].state.digest() != real_digest:
        raise RuntimeError("PROBE_ISOLATION_VIOLATION: real ledger mutated")
    nav0 = real_sx["ledger"].state.nav()
    return (navs[1] - navs[0]) / nav0


def run_probe_session(real_sx, market_state, trackers: Mapping,
                      model_signals: Mapping[str, np.ndarray],
                      active_weights: Mapping[str, float], base_fc: Mapping,
                      *, new_trade_date: int, new_mark_wall: int,
                      new_marks: Mapping[str, int], allocator,
                      journal: ProbeJournal, ledger=None,
                      min_trade_bps: int = 5, cost_bps: int = 5,
                      pi: float = PI_PROBE, horizon_days: int = 1) -> dict:
    """One frozen-baseline probe session. Observations go to the JOURNAL only.

    Candidates are selected by the LEDGER (registered, untested, open family),
    NOT by live tracker eligibility — so the measurement procedure cannot
    silently change when the original kappa rule flips a tracker's status.
    The real portfolio is never traded here.
    """
    assert_probe_ready(real_sx)
    names = sorted(model_signals)
    n_assets = len(next(iter(model_signals.values())))
    mu = np.zeros(n_assets)
    for a, w in active_weights.items():
        if w > 0 and a in model_signals:
            mu = mu + w * np.asarray(model_signals[a], dtype=float)
    families = assign_families(model_signals, existing=journal.families)
    for c in names:
        journal.set_family(c, families[c])
    report = {"observations": {}, "guarded": [], "coefficients": {}, "skipped": []}
    for c in names:
        if ledger is not None:
            key = (c, horizon_days)
            if key not in ledger.cells or key in ledger.tested:
                report["skipped"].append(c)
                continue
            if ledger.cells[key]["family"] in ledger.closed_families:
                report["guarded"].append(c)
                continue
        pic = probe_coefficient(c, families, active_weights, pi)
        report["coefficients"][c] = pic
        if pic <= 0.0:
            report["guarded"].append(c)
            continue
        blend = (1.0 - pic) * mu + pic * np.asarray(model_signals[c], dtype=float)
        obs = probe_pair(
            real_sx, market_state, base_fc, mu, blend,
            new_trade_date=new_trade_date, new_mark_wall=new_mark_wall,
            new_marks=new_marks, allocator=allocator,
            min_trade_bps=min_trade_bps, cost_bps=cost_bps,
            horizon_days=horizon_days,
        )
        journal.record(c, horizon_days, obs)
        report["observations"][c] = obs
    return report
