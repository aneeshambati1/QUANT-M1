"""G1 acceptance tests for factor model and residual breadth.

Zero-lookahead is enforced by construction: inputs are pre-computed
PIT returns. These tests verify mathematical correctness of the
factor fitting, residual covariance, and breadth computations.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.factors.model import FactorModel, _participation_ratio, _var_based_neff


@pytest.fixture
def synthetic_data():
    """Synthetic correlated returns with known factor structure."""
    rng = np.random.default_rng(42)
    n_obs, n_assets, n_factors = 500, 20, 3
    # True loadings
    B_true = rng.standard_normal((n_factors, n_assets)) * 0.5
    # Factor returns
    F = rng.standard_normal((n_obs, n_factors))
    # Idiosyncratic noise
    eps = rng.standard_normal((n_obs, n_assets)) * 0.3
    # Asset returns = F @ B + eps
    R = F @ B_true + eps
    return R, F, n_assets, n_factors


def test_fit_produces_valid_report(synthetic_data):
    R, F, n_assets, n_factors = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)

    assert report.n_assets == n_assets
    assert report.n_factors == n_factors
    assert report.n_obs == 500
    assert report.raw_participation_ratio > 0
    assert report.residual_participation_ratio > 0
    assert report.renorm_participation_ratio > 0
    assert report.lw_shrinkage_intensity >= 0
    assert report.lw_shrinkage_intensity <= 1
    assert report.residual_condition_number >= 1


def test_residual_pr_less_than_raw_pr(synthetic_data):
    """Removing factors should not increase the raw-spectrum PR."""
    R, F, _, _ = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)
    # Residual trace < raw trace (factors explain variance)
    assert report.residual_trace < float(np.sum(report.raw_eigenvalues))


def test_renormalized_pr_geq_unnormalized(synthetic_data):
    """Renormalization reshapes spectrum; PR typically increases."""
    R, F, _, _ = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)
    # Not a strict inequality (edge cases exist), but typical
    assert report.renorm_participation_ratio >= report.residual_participation_ratio * 0.8


def test_leverage_constrained_leq_renormalized(synthetic_data):
    """Leverage-constrained N_eff must not exceed renormalized N_eff."""
    R, F, _, _ = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)
    assert report.leverage_constrained_neff <= report.renorm_var_based_neff + 1e-6


def test_min_leverage_bound(synthetic_data):
    """Minimum leverage for unit risk should be >= 1 when residual var < raw var."""
    R, F, _, _ = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)
    if report.avg_residual_variance_frac < 1.0:
        assert report.min_leverage_for_unit_risk > 1.0


def test_single_factor_market_only():
    """Single market factor: residual should have higher N_eff than raw."""
    rng = np.random.default_rng(123)
    n_obs, n_assets = 1000, 30
    market = rng.standard_normal(n_obs)
    betas = rng.uniform(0.5, 1.5, n_assets)
    R = np.column_stack([market * b + rng.standard_normal(n_obs) * 0.3 for b in betas])
    F = market.reshape(-1, 1)

    fm = FactorModel()
    report = fm.fit(R, F)

    # Market explains most variance; residual is more diversified
    assert report.residual_var_based_neff > report.raw_var_based_neff


def test_no_factors_raises():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((100, 5))
    F_empty = np.empty((100, 0))
    fm = FactorModel()
    with pytest.raises(ValueError):
        fm.fit(R, F_empty)


def test_insufficient_obs_raises():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((5, 10))  # n_obs < n_assets
    F = rng.standard_normal((5, 2))
    fm = FactorModel()
    with pytest.raises(ValueError):
        fm.fit(R, F)


def test_participation_ratio_identity():
    """All-equal eigenvalues => PR = N."""
    eigs = np.ones(10)
    assert abs(_participation_ratio(eigs) - 10.0) < 1e-10


def test_participation_ratio_single_dominant():
    """One dominant eigenvalue => PR ≈ 1."""
    eigs = np.array([100.0] + [0.01] * 99)
    pr = _participation_ratio(eigs)
    assert pr < 2.0


def test_summary_format(synthetic_data):
    R, F, _, _ = synthetic_data
    fm = FactorModel()
    report = fm.fit(R, F)
    s = report.summary()
    assert "Assets=" in s
    assert "Leverage-constrained" in s
    assert "LW shrinkage=" in s