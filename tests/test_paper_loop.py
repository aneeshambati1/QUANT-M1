"""Paper-loop step: contribution vs hold baseline, fail-closed, latch block."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.orchestrator.paper_loop import run_paper_step
from quant_m1.system import build


def _state(**kw):
    base = dict(as_of="2026-11-25T15:00:00+00:00", avg_corr=0.4, vol_ann=0.15,
                liquidity_score=1.0, data_age_secs=10, model_disagreement=0.1, n_assets=3)
    base.update(kw); return MarketState(**base)


def _active():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200): t.update(0.01)
    return t


def _forecasts():
    return {"expected_returns": np.array([0.06, 0.04, 0.02]),
            "cov": np.eye(3) * 0.04, "horizon_days": 1}


def test_no_allocation_means_no_orders(tmp_path):
    sx = build(str(tmp_path))
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts={},
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks={"A": 10_000, "B": 10_000, "C": 10_000},
                         allocator=solve_allocation)
    assert rep.intents_submitted == 0
    assert rep.blocked is False


def test_flat_marks_hold_contribution_near_zero(tmp_path):
    sx = build(str(tmp_path))
    # No eligible model -> cash/hold; flat marks -> contribution ~ 0.
    t = ContributionTracker()  # probation, not active
    rep = run_paper_step(sx, _state(), {"m1": t}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks={"A": 10_000, "B": 10_000, "C": 10_000},
                         allocator=solve_allocation)
    assert abs(rep.ensemble_contribution) < 1e-9
    assert rep.intents_submitted == 0


def test_latched_divergence_blocks_step(tmp_path):
    sx = build(str(tmp_path))
    sx["gw"].latch_divergence("FAULT")
    rep = run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                         new_trade_date=101, new_mark_wall=1_000_100,
                         new_marks={"A": 10_000, "B": 10_000, "C": 10_000},
                         allocator=solve_allocation)
    assert rep.blocked is True
    assert rep.intents_submitted == 0
    assert rep.ensemble_contribution == 0.0


def test_missing_mark_for_held_symbol_rejected(tmp_path):
    sx = build(str(tmp_path))
    with pytest.raises(ValueError):
        run_paper_step(sx, _state(), {"m1": _active()}, forecasts=_forecasts(),
                     new_trade_date=101, new_mark_wall=1_000_100,
                     new_marks={"A": 10_000},  # B held but missing
                     allocator=solve_allocation)


def test_attribution_callback_updates_trackers(tmp_path):
    sx = build(str(tmp_path))
    t = _active()
    before = t.n
    run_paper_step(sx, _state(), {"m1": t}, forecasts=_forecasts(),
                   new_trade_date=101, new_mark_wall=1_000_100,
                   new_marks={"A": 10_100, "B": 10_000, "C": 10_000},
                   allocator=solve_allocation,
                   attribution=lambda c: {"m1": c})
    assert t.n == before + 1
