"""Allocation/orchestrator hardening: infeasibility, leverage cap, fail-closed."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation, AllocationError
from quant_m1.allocate.optimize import solve_cost_aware, OptimizationError
from quant_m1.allocate.engine import AllocationResult
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.orchestrator import run_cycle
from quant_m1.orchestrator.execution import generate_intents
from quant_m1.system import build


def _state(**kw):
    base = dict(as_of="2026-11-25T15:00:00+00:00", avg_corr=0.4, vol_ann=0.15,
                liquidity_score=1.0, data_age_secs=10, model_disagreement=0.1, n_assets=3)
    base.update(kw); return MarketState(**base)


def _active():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200): t.update(0.01)
    return t


def test_engine_rejects_infeasible_current_weights():
    with pytest.raises(AllocationError):
        solve_allocation(np.array([0.05, 0.05]), np.eye(2) * 0.01, current_weights=np.array([1.2, 0.0]))
    with pytest.raises(AllocationError):
        solve_allocation(np.array([0.05, 0.05]), np.eye(2) * 0.01, current_weights=np.array([-0.1, 0.5]))


def test_engine_rejects_nan_current_weights():
    with pytest.raises(AllocationError):
        solve_allocation(np.array([0.05, 0.05]), np.eye(2) * 0.01, current_weights=np.array([float("nan"), 0.5]))


def test_optimize_rejects_leverage_cap():
    with pytest.raises(OptimizationError):
        solve_cost_aware(np.array([0.05, 0.05]), np.eye(2) * 0.01, np.array([0.0, 0.0]), np.array([1.0, 1.0]), max_gross=1.5)


def test_optimize_cash_frac_is_true_cash():
    r = solve_cost_aware(np.array([0.05, 0.05]), np.eye(2) * 0.04, np.array([0.0, 0.0]), np.array([0.0, 0.0]), max_gross=0.9)
    assert r.cash_frac == pytest.approx(1.0 - r.invested_frac)
    assert r.invested_frac <= 0.9 + 1e-6


def test_cycle_allocator_error_fails_closed():
    def bad_allocator(*a, **k):
        raise AllocationError("solver exploded")
    rec = run_cycle(_state(), {"m1": _active()},
                    forecasts={"expected_returns": np.array([0.06, 0.04, 0.02]), "cov": np.eye(3) * 0.04},
                    allocator=bad_allocator)
    assert rec.allocation is None
    assert "AllocationError" in rec.note


def test_cycle_mixed_horizon_fails_closed():
    t1 = _active(); t2 = _active(); t2.horizon_days = 5
    rec = run_cycle(_state(), {"m1": t1, "m2": t2},
                    forecasts={"expected_returns": np.array([0.06, 0.04, 0.02]), "cov": np.eye(3) * 0.04},
                    allocator=solve_allocation)
    assert rec.allocation is None
    assert rec.any_eligible_model is False


def test_generate_intents_rejects_overgross_weights(tmp_path):
    sx = build(str(tmp_path))
    alloc = AllocationResult(weights=np.array([0.8, 0.8, 0.8]), invested_frac=2.4, cash_frac=-1.4,
                             expected_return=0.0, variance=0.0, turnover=0.0, band_applied=False, solver_ok=True)
    with pytest.raises(ValueError):
        generate_intents(alloc, sx["ledger"], {"A": 10_000, "B": 10_000, "C": 10_000}, sx["instruments"])


def test_generate_intents_rejects_negative_weights(tmp_path):
    sx = build(str(tmp_path))
    alloc = AllocationResult(weights=np.array([-0.1, 0.5, 0.5]), invested_frac=0.9, cash_frac=0.1,
                             expected_return=0.0, variance=0.0, turnover=0.0, band_applied=False, solver_ok=True)
    with pytest.raises(ValueError):
        generate_intents(alloc, sx["ledger"], {"A": 10_000, "B": 10_000, "C": 10_000}, sx["instruments"])
