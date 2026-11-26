"""Paper daemon tests: restart continuity (O-08), fail-closed halting (O-07)."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation
from quant_m1.authority import Intent
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.orchestrator.daemon import PaperDaemon
from quant_m1.system import build

M0 = {"A": 10_000, "B": 10_000, "C": 10_000}
UP = {"A": 10_500, "B": 10_400, "C": 10_200}


def _state():
    return MarketState(as_of="2026-11-26T15:00:00+00:00", avg_corr=0.4, vol_ann=0.15,
                       liquidity_score=1.0, data_age_secs=10, model_disagreement=0.1,
                       n_assets=3)


def _active():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200):
        t.update(0.01)
    return t


def _fc():
    return {"expected_returns": np.array([0.06, 0.04, 0.02]),
            "cov": np.eye(3) * 0.04, "horizon_days": 1, "risk_aversion": 4.0}


def test_first_start_reconciles_and_runs(tmp_path):
    da = PaperDaemon(str(tmp_path), cash=2_000_000)
    assert da.reconcile_status == "RECOVERED" and not da.halted
    rep = da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
                 allocator=solve_allocation)
    assert rep.sessions_completed == 1 and not rep.halted


def test_restart_continues_without_exec_id_conflict(tmp_path):
    """O-08: restoring the venue exec sequence prevents EXEC_ID_PAYLOAD_CONFLICT."""
    d = str(tmp_path)
    da = PaperDaemon(d, cash=2_000_000)
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    da2 = PaperDaemon(d, cash=2_000_000)
    assert da2.reconcile_status == "RECOVERED"
    assert not da2.halted
    rep = da2.run([(_state(), _fc(), 102, 1_000_200, UP)], {"m1": _active()},
                  allocator=solve_allocation)
    assert not rep.halted, rep.halt_reason
    assert not da2.sx["gw"]._divergence_latched
    assert rep.sessions_completed == 1


def test_restart_preserves_session_and_positions(tmp_path):
    d = str(tmp_path)
    da = PaperDaemon(d, cash=2_000_000)
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    pos = dict(da.sx["ledger"].state.positions)
    da2 = PaperDaemon(d, cash=2_000_000)
    assert da2.sx["ledger"].state.trade_date == 101
    assert dict(da2.sx["ledger"].state.positions) == pos
    assert da2.sx["venue"].positions == pos


def test_divergence_halts_and_persists(tmp_path):
    d = str(tmp_path)
    da = PaperDaemon(d, cash=2_000_000)
    da.sx["gw"].latch_divergence("FAULT")
    rep = da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
                 allocator=solve_allocation)
    assert rep.halted
    da2 = PaperDaemon(d, cash=2_000_000)
    assert da2.halted
    rep2 = da2.run([(_state(), _fc(), 102, 1_000_200, M0)], {"m1": _active()},
                   allocator=solve_allocation)
    assert rep2.halted and rep2.sessions_attempted == 0


def test_tampered_checkpoint_fails_closed(tmp_path):
    d = str(tmp_path)
    da = PaperDaemon(d, cash=2_000_000)
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    import json, os
    p = os.path.join(d, "daemon_checkpoint.json")
    ck = json.load(open(p))
    ck["venue_eid"] = 0          # force exec-id reuse
    json.dump(ck, open(p, "w"))
    da2 = PaperDaemon(d, cash=2_000_000)
    # Fail-fast: restored exec sequence precedes already-applied exec IDs.
    assert da2.halted
    assert "CHECKPOINT_EXEC_SEQ_REGRESSION" in da2.halt_reason
    rep = da2.run([(_state(), _fc(), 102, 1_000_200, UP)], {"m1": _active()},
                  allocator=solve_allocation)
    assert rep.halted and rep.sessions_attempted == 0


def test_limit_respected(tmp_path):
    da = PaperDaemon(str(tmp_path), cash=2_000_000)
    sessions = [(_state(), _fc(), 101 + i, 1_000_100 * (i + 1), M0) for i in range(5)]
    rep = da.run(sessions, {"m1": _active()}, allocator=solve_allocation, limit=2)
    assert rep.sessions_attempted == 2 and rep.sessions_completed == 2


def test_step_error_halts(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("step exploded")
    da = PaperDaemon(str(tmp_path), cash=2_000_000, step_fn=boom)
    rep = da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
                 allocator=solve_allocation)
    assert rep.halted and "STEP_ERROR" in rep.halt_reason


def test_reset_halt_requires_cleared_divergence(tmp_path):
    da = PaperDaemon(str(tmp_path), cash=2_000_000)
    da.sx["gw"].latch_divergence("FAULT")
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    with pytest.raises(ValueError, match="GATEWAY_STILL_DIVERGENT"):
        da.reset_halt(operator_reason="trying")


def test_reset_halt_requires_reason(tmp_path):
    da = PaperDaemon(str(tmp_path), cash=2_000_000)
    with pytest.raises(ValueError):
        da.reset_halt(operator_reason="")


def test_injected_step_fn_is_used(tmp_path):
    calls = []
    def spy(sx, ms, tr, fc, **kw):
        calls.append(kw["new_trade_date"])
        from quant_m1.orchestrator.paper_loop import run_paper_step
        return run_paper_step(sx, ms, tr, fc, **kw)
    da = PaperDaemon(str(tmp_path), cash=2_000_000, step_fn=spy)
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    assert calls == [101]


def test_session_regression_halts(tmp_path):
    da = PaperDaemon(str(tmp_path), cash=2_000_000)
    da.run([(_state(), _fc(), 101, 1_000_100, M0)], {"m1": _active()},
           allocator=solve_allocation)
    rep = da.run([(_state(), _fc(), 100, 1_000_000, M0)], {"m1": _active()},
                 allocator=solve_allocation)
    assert rep.halted
