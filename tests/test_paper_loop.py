"""Paper-loop step: end-to-end submission/fill/reconciliation, partial fills,
restart continuity, fail-closed behaviour, and contribution vs hold baseline."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation
from quant_m1.authority import Intent
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.orchestrator.paper_loop import run_paper_step
from quant_m1.system import build

MARKS = {"A": 10_000, "B": 10_000, "C": 10_000}


def _state(**kw):
    base = dict(as_of="2026-11-26T15:00:00+00:00", avg_corr=0.4, vol_ann=0.15,
                liquidity_score=1.0, data_age_secs=10, model_disagreement=0.1, n_assets=3)
    base.update(kw); return MarketState(**base)


def _active():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200): t.update(0.01)
    return t


def _forecasts():
    # risk_aversion=4 keeps notionals small enough to clear the kernel caps.
    return {"expected_returns": np.array([0.06, 0.04, 0.02]),
            "cov": np.eye(3) * 0.04, "horizon_days": 1, "risk_aversion": 4.0}


def test_successful_order_reaches_fill_and_reconciles(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks=MARKS, allocator=solve_allocation)
    assert rep.blocked is False
    assert rep.intents_submitted > 0                 # clock fix: no STALE_MARKS
    assert rep.filled_orders == rep.intents_submitted
    assert any(q > 0 for q in sx["ledger"].state.positions.values())
    assert sx["gw"]._reconcile_snapshot() is True
    assert sx["flags"].exec == "OK"
    # Flat marks: trading only costs, so contribution vs hold is negative.
    assert rep.ensemble_contribution < 0


def test_rejected_order_is_recorded_not_filled(tmp_path):
    # Huge cash -> order notional exceeds the per-order cap -> rejection.
    sx = build(str(tmp_path), cash=100_000_000, positions={}, marks=MARKS)
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks=MARKS, allocator=solve_allocation)
    assert rep.intents_rejected > 0
    assert rep.filled_orders == 0
    assert not any(q > 0 for q in sx["ledger"].state.positions.values())


def test_partial_fill_completes_on_next_step(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    sx["clock"].advance_wall(100)
    i = Intent("C", "BUY", 10, 10_000, 10)
    tok = sx["kernel"].authorize(sx["ledger"], i, sx["flags"], sx["clock"], False)
    sx["gw"].submit(tok, i)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 3, price=10_000))  # 3 of 10
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_200,
                         new_marks=MARKS, allocator=solve_allocation)
    o = next(o for o in sx["ledger"].state.working if o.order_id == tok.order_id)
    assert o.cum_filled == 10 and o.status == "FILLED"   # incremental fill fix
    assert rep.blocked is False


def test_restart_continuity_keeps_session(tmp_path):
    d = str(tmp_path)
    sx = build(d, cash=2_000_000, positions={}, marks=MARKS)
    run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                   new_trade_date=101, new_mark_wall=1_000_100,
                   new_marks=MARKS, allocator=solve_allocation)
    # R6-02: the session is restored from the signed composite, not rolled back.
    sx2 = build(d, cash=2_000_000, positions={}, marks=MARKS)
    assert sx2["ledger"].state.trade_date == 101
    assert sx2["ledger"].state.mark_wall == 1_000_100
    # A naive restart rebuilds the mock venue, which reuses exec_ids. The gateway
    # must detect the exec_id payload conflict and FAIL CLOSED, not trade blindly.
    # (Continued trading after restart requires venue reconciliation with
    # persistent broker exec_ids — part of the open full-runner work O-07/O-08.)
    rep = run_paper_step(sx2, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=102, new_mark_wall=1_000_200,
                         new_marks=MARKS, allocator=solve_allocation)
    assert rep.blocked is True
    assert rep.ensemble_contribution == 0.0


def test_same_session_rerun_is_idempotent(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    kw = dict(new_trade_date=101, new_mark_wall=1_000_100, new_marks=MARKS,
              allocator=solve_allocation)
    run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(), **kw)
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(), **kw)
    assert rep.blocked is False   # no SESSION_REGRESSION crash on retry


def test_cash_and_positions_reconcile_with_venue(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                   new_trade_date=101, new_mark_wall=1_000_100,
                   new_marks=MARKS, allocator=solve_allocation)
    snap = sx["venue"].snapshot()
    led = sx["ledger"].state
    assert dict(led.positions) == {k: v for k, v in snap["positions"].items() if v}
    assert led.cash_settled + led.receivable_total() == snap["cash"]


def test_no_allocation_means_no_orders(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts={},
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks=MARKS, allocator=solve_allocation)
    assert rep.intents_submitted == 0 and rep.blocked is False


def test_latched_divergence_blocks_step(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    sx["gw"].latch_divergence("FAULT")
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks=MARKS, allocator=solve_allocation)
    assert rep.blocked is True and rep.intents_submitted == 0
    assert rep.ensemble_contribution == 0.0


def test_missing_mark_for_held_symbol_rejected(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={"A": 10}, marks=MARKS)
    with pytest.raises(ValueError):
        run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                     new_trade_date=101, new_mark_wall=1_000_100,
                     new_marks={"A": 10_000}, allocator=solve_allocation)


def test_attribution_updates_tracker_only_after_real_trade(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=MARKS)
    t = _active(); before = t.n
    rep = run_paper_step(sx, _state(), {"m1": t}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks=MARKS, allocator=solve_allocation,
                         attribution=lambda c: {"m1": c})
    assert rep.intents_submitted > 0      # proves a trade actually occurred
    assert t.n == before + 1
