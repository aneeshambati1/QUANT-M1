"""Research-only Phase 2 paired probe attribution.

Probe outcomes are recorded only in ProbeJournal. They never update the live
ContributionTracker or reach a live execution path. Promotion authority is
owned by PromotionLedger; this module does not grant it.
"""
from __future__ import annotations

import shutil
import tempfile
from threading import RLock
from typing import Mapping, Optional

import numpy as np

from ..core import Store, sha
from ..orchestrator.paper_loop import _shadow_system, run_paper_step
from .contribution import ContributionTracker

PI_PROBE = 0.05
RHO_DUP = 0.95


class _ProbeGate:
    """Counterfactual gate used only in disposable probe shadows."""

    def __init__(self, horizon_days: int):
        self.horizon_days = horizon_days
        self.eligible = True

    def raw_weight(self) -> float:
        return 1.0

    def snapshot(self) -> dict:
        return {"probe_gate": True, "horizon_days": self.horizon_days}


class ProbeJournal:
    """Hash-chained research journal with immutable family assignments."""

    def __init__(self, path: str):
        self.store = Store(path)
        self._lock = RLock()
        self.chain = self.store.load() or []
        self._verify()
        self.observations: dict[tuple[str, int], list[float]] = {}
        self.families: dict[str, str] = {}

        for rec in self.chain:
            p = rec["payload"]
            if rec["kind"] == "FAMILY":
                candidate, family = p["candidate"], p["family"]
                old = self.families.get(candidate)
                if old is not None and old != family:
                    raise ValueError("PROBE_FAMILY_HISTORY_CONFLICT")
                self.families[candidate] = family
            elif rec["kind"] == "OBS":
                key = (p["candidate"], p["horizon"])
                self.observations.setdefault(key, []).append(float(p["value"]))
            else:
                raise ValueError("PROBE_JOURNAL_UNKNOWN_EVENT")

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
        next_chain = [*self.chain, rec]
        self.store.save(next_chain)
        self.chain = next_chain

    def set_family(self, candidate: str, family: str) -> None:
        if not candidate or not family:
            raise ValueError("candidate and family must be nonempty")
        with self._lock:
            old = self.families.get(candidate)
            if old is not None:
                if old != family:
                    raise ValueError("FAMILY_ALREADY_ASSIGNED_IMMUTABLE")
                return
            self._append("FAMILY", {"candidate": candidate, "family": family})
            self.families[candidate] = family

    def record(self, candidate: str, horizon: int, value: float) -> None:
        if not candidate:
            raise ValueError("candidate must be nonempty")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if not np.isfinite(value):
            raise ValueError("observation must be finite")
        with self._lock:
            if candidate not in self.families:
                raise ValueError("PROBE_CANDIDATE_FAMILY_NOT_REGISTERED")
            payload = {
                "candidate": candidate,
                "horizon": horizon,
                "value": float(value),
            }
            self._append("OBS", payload)
            self.observations.setdefault((candidate, horizon), []).append(
                float(value)
            )

    def n_obs(self, candidate: str, horizon: int) -> int:
        return len(self.observations.get((candidate, horizon), []))

    def certificate_tracker(
        self, candidate: str, horizon: int
    ) -> ContributionTracker:
        tracker = ContributionTracker(horizon_days=horizon)
        for value in self.observations.get((candidate, horizon), []):
            tracker.update(value, horizon_days=horizon)
        return tracker


def assign_families(
    signals: Mapping[str, np.ndarray],
    rho_dup: float = RHO_DUP,
    existing: Optional[Mapping[str, str]] = None,
) -> dict:
    if not np.isfinite(rho_dup) or not -1.0 <= rho_dup <= 1.0:
        raise ValueError("rho_dup must be finite and in [-1, 1]")

    fam = dict(existing or {})
    names = sorted(signals)
    reps = sorted({fam[name] for name in names if name in fam})

    for name in names:
        if name in fam:
            continue
        values = np.asarray(signals[name], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid signal vector: {name}")

        assigned = False
        for representative in reps:
            if representative not in signals:
                continue
            other = np.asarray(signals[representative], dtype=float)
            if other.shape != values.shape:
                continue
            if values.std() <= 0 or other.std() <= 0:
                continue
            corr = float(np.corrcoef(other, values)[0, 1])
            if np.isfinite(corr) and corr >= rho_dup:
                fam[name] = representative
                assigned = True
                break

        if not assigned:
            fam[name] = name
            reps.append(name)
            reps.sort()

    return fam


def probe_coefficient(
    candidate: str,
    families: Mapping[str, str],
    active: Mapping[str, float],
    pi: float = PI_PROBE,
) -> float:
    if not np.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be finite and in [0, 1]")
    representative = families.get(candidate, candidate)
    if candidate != representative:
        return 0.0
    if active.get(representative, 0.0) > 0:
        return 0.0
    return float(pi)


def _start_snapshot(sx) -> dict:
    state = sx["ledger"].state
    return {
        "cash_settled": state.cash_settled,
        "positions": dict(state.positions),
        "marks": dict(state.marks),
        "receivables": tuple(state.receivables),
        "peak_nav": state.peak_nav,
        "trade_date": state.trade_date,
        "mark_wall": state.mark_wall,
    }


def assert_probe_ready(sx) -> None:
    gateway, flags, ledger = sx["gw"], sx["flags"], sx["ledger"]
    if gateway._divergence_latched:
        raise RuntimeError("PROBE_REFUSED_DIVERGENT")
    if flags.killed or flags.frozen:
        raise RuntimeError("PROBE_REFUSED_KILLED_OR_FROZEN")
    if flags.exec != "OK":
        raise RuntimeError("PROBE_REFUSED_EXEC_NOT_OK")
    if ledger.working_active():
        raise RuntimeError("PROBE_REFUSED_UNRESOLVED_WORKING_ORDERS")


def probe_pair(
    real_sx,
    market_state,
    base_fc: Mapping,
    mu_active: np.ndarray,
    probe_blend: np.ndarray,
    *,
    new_trade_date: int,
    new_mark_wall: int,
    new_marks: Mapping[str, int],
    allocator,
    min_trade_bps: int,
    cost_bps: int,
    horizon_days: int = 1,
) -> float:
    assert_probe_ready(real_sx)
    start = _start_snapshot(real_sx)
    nav0 = real_sx["ledger"].state.nav()
    if not np.isfinite(nav0) or nav0 <= 0:
        raise ValueError("starting NAV must be positive")

    real_digest = real_sx["ledger"].state.digest()
    gate = {"__probe_gate__": _ProbeGate(horizon_days)}
    navs = []

    try:
        for blend in (mu_active, probe_blend):
            tmp = tempfile.mkdtemp(prefix="quant_probe_")
            try:
                shadow = _shadow_system(real_sx, start, tmp)
                forecast = {
                    **base_fc,
                    "expected_returns": np.asarray(blend, dtype=float),
                }
                forecast.pop("model_signals", None)
                report = run_paper_step(
                    shadow,
                    market_state,
                    gate,
                    forecast,
                    new_trade_date=new_trade_date,
                    new_mark_wall=new_mark_wall,
                    new_marks=new_marks,
                    allocator=allocator,
                    min_trade_bps=min_trade_bps,
                    cost_bps=cost_bps,
                )
                if report.blocked:
                    raise RuntimeError("PROBE_SHADOW_BLOCKED_NO_OBSERVATION")
                navs.append(float(report.model_after))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if real_sx["ledger"].state.digest() != real_digest:
            raise RuntimeError("PROBE_ISOLATION_VIOLATION: real ledger mutated")

    return (navs[1] - navs[0]) / nav0


def run_probe_session(
    real_sx,
    market_state,
    trackers: Mapping,
    model_signals: Mapping[str, np.ndarray],
    active_weights: Mapping[str, float],
    base_fc: Mapping,
    *,
    new_trade_date: int,
    new_mark_wall: int,
    new_marks: Mapping[str, int],
    allocator,
    journal: ProbeJournal,
    ledger=None,
    min_trade_bps: int = 5,
    cost_bps: int = 5,
    pi: float = PI_PROBE,
    horizon_days: int = 1,
) -> dict:
    """Run one session; ledger registration is mandatory and probes are shadow-only."""
    assert_probe_ready(real_sx)
    if ledger is None:
        raise RuntimeError("PROBE_REFUSED_NO_PROMOTION_LEDGER")
    if not model_signals:
        return {"observations": {}, "guarded": [], "coefficients": {}, "skipped": []}
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
        raise ValueError("horizon_days must be an integer >= 1")

    names = sorted(model_signals)
    first = np.asarray(model_signals[names[0]], dtype=float)
    if first.ndim != 1 or not np.all(np.isfinite(first)):
        raise ValueError("invalid model signal vector")
    mu = np.zeros_like(first)
    for asset, weight in active_weights.items():
        if weight > 0 and asset in model_signals:
            signal = np.asarray(model_signals[asset], dtype=float)
            if signal.shape != mu.shape or not np.all(np.isfinite(signal)):
                raise ValueError(f"invalid signal vector: {asset}")
            mu += weight * signal

    families = assign_families(model_signals, existing=journal.families)
    for candidate in names:
        journal.set_family(candidate, families[candidate])

    report = {
        "observations": {},
        "guarded": [],
        "coefficients": {},
        "skipped": [],
    }

    for candidate in names:
        key = (candidate, horizon_days)
        if key not in ledger.cells or key in ledger.tested:
            report["skipped"].append(candidate)
            continue
        cell = ledger.cells[key]
        if cell["family"] in ledger.closed_families:
            report["guarded"].append(candidate)
            continue

        coefficient = probe_coefficient(candidate, families, active_weights, pi)
        report["coefficients"][candidate] = coefficient
        if coefficient <= 0:
            report["guarded"].append(candidate)
            continue

        signal = np.asarray(model_signals[candidate], dtype=float)
        if signal.shape != mu.shape or not np.all(np.isfinite(signal)):
            raise ValueError(f"invalid signal vector: {candidate}")
        blend = (1.0 - coefficient) * mu + coefficient * signal

        observation = probe_pair(
            real_sx,
            market_state,
            base_fc,
            mu,
            blend,
            new_trade_date=new_trade_date,
            new_mark_wall=new_mark_wall,
            new_marks=new_marks,
            allocator=allocator,
            min_trade_bps=min_trade_bps,
            cost_bps=cost_bps,
            horizon_days=horizon_days,
        )
        journal.record(candidate, horizon_days, observation)
        report["observations"][candidate] = observation

    return report
