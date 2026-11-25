"""Cost-aware allocator tests. SYNTHETIC inputs; no edge claim."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.allocate.optimize import solve_cost_aware, OptimizationError


def test_high_cost_holds_current_weights():
    mu = np.array([0.001, 0.001])
    cov = np.eye(2) * 0.01
    w0 = np.array([0.5, 0.5])
    r = solve_cost_aware(mu, cov, w0, np.array([1000.0, 1000.0]))
    assert r.solver_ok
    assert r.turnover < 1e-6
    assert np.allclose(r.weights, w0)


def test_zero_cost_trades_toward_signal():
    mu = np.array([0.06, 0.04])
    cov = np.eye(2) * 0.04
    w0 = np.array([0.0, 0.0])
    r = solve_cost_aware(mu, cov, w0, np.array([0.0, 0.0]))
    assert r.solver_ok
    assert r.turnover > 0.1
    assert r.weights[0] > r.weights[1]


def test_cost_reduces_turnover():
    mu = np.array([0.05, 0.03])
    cov = np.eye(2) * 0.02
    w0 = np.array([0.0, 0.0])
    free = solve_cost_aware(mu, cov, w0, np.array([0.0, 0.0]))
    costly = solve_cost_aware(mu, cov, w0, np.array([200.0, 200.0]))
    assert costly.turnover < free.turnover


def test_per_asset_band_holds_small_trades():
    mu = np.array([0.050, 0.049])
    cov = np.eye(2) * 0.05
    w0 = np.array([0.5, 0.5])
    r = solve_cost_aware(mu, cov, w0, np.array([1.0, 1.0]),
                         band_frac=np.array([0.10, 0.10]))
    assert r.band_applied
    assert r.turnover < 1e-6


def test_invalid_current_weights_rejected():
    mu = np.array([0.05, 0.05])
    cov = np.eye(2) * 0.01
    with pytest.raises(OptimizationError):
        solve_cost_aware(mu, cov, np.array([1.1, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(OptimizationError):
        solve_cost_aware(mu, cov, np.array([-0.1, 0.5]), np.array([1.0, 1.0]))


def test_gross_cap_respected():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0, 0.2, 5)
    A = rng.standard_normal((5, 5))
    cov = A @ A.T / 5 + 0.01 * np.eye(5)
    r = solve_cost_aware(mu, cov, np.zeros(5), np.full(5, 5.0),
                         risk_aversion=0.5)
    assert r.weights.sum() <= 1.0 + 1e-6
    assert np.all(r.weights >= -1e-9)


def test_deterministic():
    mu = np.array([0.06, 0.04, 0.05])
    cov = np.diag([0.04, 0.03, 0.05])
    w0 = np.array([0.2, 0.2, 0.2])
    a = solve_cost_aware(mu, cov, w0, np.array([5.0, 5.0, 5.0]))
    b = solve_cost_aware(mu, cov, w0, np.array([5.0, 5.0, 5.0]))
    assert np.allclose(a.weights, b.weights, atol=1e-9)


def test_objective_never_worse_than_hold():
    rng = np.random.default_rng(3)
    for _ in range(5):
        mu = rng.uniform(-0.05, 0.1, 4)
        cov = np.eye(4) * 0.03
        w0 = np.full(4, 0.2)
        r = solve_cost_aware(mu, cov, w0, np.full(4, 10.0))
        assert r.objective <= r.hold_objective + 1e-7


def test_report_is_json_safe():
    mu = np.array([0.05, 0.05])
    r = solve_cost_aware(mu, np.eye(2) * 0.01, np.array([0.5, 0.5]),
                         np.array([5.0, 5.0]))
    import json
    json.dumps(r.as_dict())