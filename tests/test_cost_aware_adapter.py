"""Adapter contract tests: cycle compatibility, fail-closed, no leverage."""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pytest
from quant_m1.allocate.cost_aware import cost_aware_allocator
from quant_m1.allocate.optimize import OptimizationError
from quant_m1.allocate import solve_allocation
from quant_m1.meta.contribution import ContributionTracker
from quant_m1.meta.state import MarketState
from quant_m1.orchestrator.cycle import run_cycle


def test_contract_fields_and_constraints():
    mu = np.array([0.001, 0.0, -0.001]); cov = np.eye(3) * 0.0002
    res = cost_aware_allocator(mu, cov, risk_aversion=2.0,
                               current_weights=None, band_frac=0.25, cost_bps=5)
    assert hasattr(res, "solver_ok") and hasattr(res, "weights")
    w = np.asarray(res.weights)
    assert np.all(w >= -1e-9) and w.sum() <= 1.0 + 1e-9 and np.all(np.isfinite(w))


def test_scalar_and_vector_cost_broadcast():
    mu = np.array([0.001, 0.0005]); cov = np.eye(2) * 0.0002
    r1 = cost_aware_allocator(mu, cov, cost_bps=5)
    r2 = cost_aware_allocator(mu, cov, cost_bps=[5, 5])
    assert np.allclose(r1.weights, r2.weights)
    with pytest.raises(OptimizationError):
        cost_aware_allocator(mu, cov, cost_bps=-1)
    with pytest.raises(OptimizationError):
        cost_aware_allocator(mu, cov, cost_bps=True)


def test_infeasible_current_weights_fail_closed():
    mu = np.array([0.001, 0.0]); cov = np.eye(2) * 0.0002
    with pytest.raises(OptimizationError):
        cost_aware_allocator(mu, cov, current_weights=[0.8, 0.8])


def test_high_cost_declines_marginal_trade():
    mu = np.array([0.0006, 0.0005]); cov = np.eye(2) * 0.0002
    w0 = np.array([0.5, 0.5])
    ca = cost_aware_allocator(mu, cov, risk_aversion=2.0,
                              current_weights=w0, band_frac=0.0, cost_bps=100)
    eng = solve_allocation(mu, cov, risk_aversion=2.0,
                           current_weights=w0, band_frac=0.0)
    assert ca.turnover <= eng.turnover + 1e-9


def _ms():
    return MarketState(as_of=datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat(),
                       avg_corr=0.3, vol_ann=0.15, liquidity_score=1.0,
                       data_age_secs=0, model_disagreement=0.0, n_assets=3,
                       drawdown_bps=0, drawdown_trigger_bps=1_000)


def test_cycle_accepts_adapter_with_earned_authority():
    t = ContributionTracker()
    for _ in range(300):
        t.update(0.02)
    assert t.eligible
    fc = {"model_signals": {"m1": np.array([0.001, 0.0, -0.001])},
          "cov": np.eye(3) * 0.0002, "risk_aversion": 2.0, "horizon_days": 1,
          "current_weights": np.zeros(3), "band_frac": 0.25}
    rec = run_cycle(_ms(), {"m1": t}, forecasts=fc, allocator=cost_aware_allocator)
    assert rec.any_eligible_model and rec.allocation is not None
    w = np.asarray(rec.allocation.weights)
    assert np.all(w >= -1e-9) and w.sum() <= 1.0 + 1e-9


def test_cycle_fail_closed_on_bad_forecast():
    t = ContributionTracker()
    for _ in range(300):
        t.update(0.02)
    fc = {"model_signals": {"m1": np.array([np.nan, 0.0, 0.0])},
          "cov": np.eye(3) * 0.0002, "risk_aversion": 2.0, "horizon_days": 1}
    rec = run_cycle(_ms(), {"m1": t}, forecasts=fc, allocator=cost_aware_allocator)
    assert rec.allocation is None
