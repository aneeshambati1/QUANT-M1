"""Acceptance tests for the allocation engine and the orchestrator cycle."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation, AllocationError
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.meta.policy import Action
from quant_m1.orchestrator import SessionSchedule, Phase, run_cycle, apply_realized


def _state(**kw) -> MarketState:
    base = dict(as_of="2026-11-25T15:00:00+00:00", avg_corr=0.436,
                vol_ann=0.155, liquidity_score=1.0, data_age_secs=10,
                model_disagreement=0.1, n_assets=46)
    base.update(kw)
    return MarketState(**base)


def _active_tracker(name="m1", n=200, contrib=0.01):
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(n):
        t.update(contrib)
    return t


# ---- allocation engine ----

def test_allocation_interior_solution():
    mu = np.array([0.06, 0.04])
    cov = np.diag([0.04, 0.04])
    r = solve_allocation(mu, cov, risk_aversion=1.0)
    assert r.solver_ok
    assert r.weights[0] == pytest.approx(0.75, abs=0.03)
    assert r.weights[1] == pytest.approx(0.25, abs=0.03)
    assert r.invested_frac == pytest.approx(1.0, abs=1e-6)
    assert r.cash_frac == pytest.approx(0.0, abs=1e-6)


def test_allocation_all_negative_returns_goes_to_cash():
    mu = np.array([-0.01, -0.01])
    cov = np.diag([0.04, 0.04])
    r = solve_allocation(mu, cov, risk_aversion=1.0)
    assert r.invested_frac < 1e-3
    assert r.cash_frac == pytest.approx(1.0, abs=1e-3)


def test_allocation_respects_no_leverage_and_long_only():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.0, 0.2, 5)
    A = rng.standard_normal((5, 5))
    cov = A @ A.T / 5 + 0.01 * np.eye(5)
    r = solve_allocation(mu, cov, risk_aversion=0.5)
    assert np.all(r.weights >= -1e-9)
    assert r.weights.sum() <= 1.0 + 1e-6


def test_allocation_is_deterministic():
    mu = np.array([0.06, 0.04, 0.05])
    cov = np.diag([0.04, 0.03, 0.05])
    a = solve_allocation(mu, cov, risk_aversion=1.0)
    b = solve_allocation(mu, cov, risk_aversion=1.0)
    assert np.allclose(a.weights, b.weights, atol=1e-9)


def test_no_trade_band_holds_position():
    mu = np.array([0.06, 0.04])
    cov = np.diag([0.04, 0.04])
    cur = np.array([0.75, 0.25])
    r = solve_allocation(mu, cov, risk_aversion=1.0,
                         current_weights=cur, band_frac=0.10)
    assert r.band_applied is True
    assert np.allclose(r.weights, cur)
    assert r.turnover == pytest.approx(0.0, abs=1e-9)


def test_allocation_rejects_bad_shapes():
    with pytest.raises(AllocationError):
        solve_allocation(np.array([0.01]), np.eye(2))
    with pytest.raises(AllocationError):
        solve_allocation(np.array([0.01, 0.02]), np.eye(2), risk_aversion=-1)


def test_allocation_psd_repair():
    mu = np.array([0.06, 0.04])
    cov = np.array([[0.04, 0.05], [0.05, 0.04]])
    r = solve_allocation(mu, cov, risk_aversion=1.0)
    assert np.all(np.isfinite(r.weights))


# ---- scheduler ----

def test_phase_boundaries():
    s = SessionSchedule()
    assert s.phase_at(15) is Phase.MARKET_HOURS
    assert s.phase_at(12) is Phase.PRE_MARKET
    assert s.phase_at(22) is Phase.POST_MARKET
    assert s.phase_at(3) is Phase.NIGHT_RESEARCH


def test_should_run_cycle():
    s = SessionSchedule()
    assert s.should_run_cycle(15, True) is True
    assert s.should_run_cycle(3, False) is True
    assert s.should_run_cycle(15, False) is False
    assert s.should_run_cycle(12, True) is True


def test_schedule_rejects_invalid_hours():
    with pytest.raises(ValueError):
        SessionSchedule(market_open_utc_hour=22, market_close_utc_hour=21)
    with pytest.raises(ValueError):
        SessionSchedule().phase_at(25)


# ---- orchestrator cycle ----

def test_cycle_divergent_state_latches_and_does_not_allocate():
    trackers = {"m1": _active_tracker()}
    rec = run_cycle(_state(execution_divergent=True), trackers,
                    forecasts={"expected_returns": np.array([0.06, 0.04]),
                               "cov": np.diag([0.04, 0.04])},
                    allocator=solve_allocation)
    assert rec.action == Action.LATCH_FREEZE.value
    assert rec.permits_new_risk is False
    assert rec.allocation is None


def test_cycle_allocates_when_calm_and_model_eligible():
    trackers = {"m1": _active_tracker()}
    rec = run_cycle(_state(), trackers,
                    forecasts={"expected_returns": np.array([0.06, 0.04]),
                               "cov": np.diag([0.04, 0.04])},
                    allocator=solve_allocation)
    assert rec.action == Action.ALLOCATE_MARGINAL.value
    assert rec.any_eligible_model is True
    assert rec.allocation is not None
    assert rec.allocation.invested_frac == pytest.approx(1.0, abs=1e-6)
    assert rec.model_weights["m1"] == pytest.approx(1.0)


def test_cycle_no_eligible_model_means_cash():
    trackers = {"m1": ContributionTracker()}
    rec = run_cycle(_state(), trackers,
                    forecasts={"expected_returns": np.array([0.06, 0.04]),
                               "cov": np.diag([0.04, 0.04])},
                    allocator=solve_allocation)
    assert rec.permits_new_risk is True
    assert rec.any_eligible_model is False
    assert rec.allocation is None
    assert "cash is a position" in rec.note


def test_cycle_records_phase_from_schedule():
    s = SessionSchedule()
    rec = run_cycle(_state(), {"m1": _active_tracker()},
                    schedule=s, now_hour_utc=15)
    assert rec.phase == "market_hours"


def test_apply_realized_updates_present_and_decays_absent():
    a = _active_tracker()
    b = _active_tracker()
    n_a, idle_b = a.n, b.idle
    out = apply_realized({"a": a, "b": b}, {"a": 0.005})
    assert out["updated"] == ["a"]
    assert out["decayed"] == ["b"]
    assert a.n == n_a + 1
    assert b.idle == idle_b + 1


def test_cycle_record_as_dict_is_json_safe():
    rec = run_cycle(_state(), {"m1": _active_tracker()},
                    forecasts={"expected_returns": np.array([0.06, 0.04]),
                               "cov": np.diag([0.04, 0.04])},
                    allocator=solve_allocation)
    d = rec.as_dict()
    import json
    json.dumps(d)
