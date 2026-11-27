"""Phase 2 research-only probe attribution (v3 Amendments A/B).

Frozen-baseline, parallel paired add-one-in at a forecast-blend coefficient
pi=0.05. Family-level coefficient cap: only the family representative gets pi;
duplicates get pi_c=0, so their probe blend equals the baseline and their
paired observation is identically zero (fail-closed, not promotable).

ISOLATION: probes run in disposable shadow systems with their own ledger,
gateway, MockVenue and workdir. The real system is never mutated; a digest
guard asserts this. Probe decisions never reach the real gateway or any live
endpoint. Probe coefficients never become live weights.

The _ProbeGate below is a counterfactual research construct used ONLY inside
shadow tracker maps to open the allocation gate for a prescribed blend. It is
never inserted into real trackers and grants no real authority.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Mapping, Optional

import numpy as np

from ..orchestrator.paper_loop import _shadow_system, run_paper_step

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


def assign_families(signals: Mapping[str, np.ndarray],
                    rho_dup: float = RHO_DUP) -> dict:
    """Deterministic family assignment: sorted IDs; a signal joins the first
    earlier representative with Pearson correlation >= rho_dup."""
    names = sorted(signals)
    reps: list = []
    fam: dict = {}
    for name in names:
        v = np.asarray(signals[name], dtype=float)
        placed = False
        for r in reps:
            u = np.asarray(signals[r], dtype=float)
            if v.std() > 0 and u.std() > 0:
                c = float(np.corrcoef(u, v)[0, 1])
                if np.isfinite(c) and c >= rho_dup:
                    fam[name] = r
                    placed = True
                    break
        if not placed:
            reps.append(name)
            fam[name] = name
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


def probe_pair(real_sx, market_state, base_fc: Mapping, mu_active: np.ndarray,
               probe_blend: np.ndarray, *, new_trade_date: int,
               new_mark_wall: int, new_marks: Mapping[str, int],
               allocator, min_trade_bps: int, cost_bps: int,
               horizon_days: int = 1) -> float:
    """Paired after-cost observation: (NAV_probe - NAV_base)/NAV_start.
    Both shadows run the full paper path (policy, kernel veto, costs)."""
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
                      min_trade_bps: int = 5, cost_bps: int = 5,
                      pi: float = PI_PROBE, horizon_days: int = 1) -> dict:
    """One frozen-baseline probe session for probationary candidates.

    mu_active is frozen from the earned active weights at entry. Each
    representative candidate is evaluated add-one-in; duplicates get pi_c=0
    and are NOT fed observations (guard recorded). Representative observations
    are fed to the real probationary tracker via update() (approved Phase 2
    mechanism). The real portfolio is never traded by this function.
    """
    names = sorted(model_signals)
    n_assets = len(next(iter(model_signals.values())))
    mu = np.zeros(n_assets)
    for a, w in active_weights.items():
        if w > 0 and a in model_signals:
            mu = mu + w * np.asarray(model_signals[a], dtype=float)
    families = assign_families(model_signals)
    report = {"observations": {}, "guarded": [], "coefficients": {}}
    for c in names:
        t = trackers.get(c)
        if t is None or getattr(t, "eligible", False):
            continue  # only probationary candidates are probed
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
        t.update(obs, horizon_days=horizon_days)
        report["observations"][c] = obs
    return report
