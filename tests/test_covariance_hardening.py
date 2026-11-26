"""Covariance finite-input guards and single-factor edge case."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.portfolio.covariance import (
    ledoit_wolf_identity, ledoit_wolf_constant_correlation, factor_covariance,
)


def test_ledoit_identity_rejects_nan():
    with pytest.raises(ValueError):
        ledoit_wolf_identity(np.array([[0.01, 0.02], [float("nan"), 0.01], [0.0, 0.03]]))


def test_ledoit_cc_rejects_inf():
    with pytest.raises(ValueError):
        ledoit_wolf_constant_correlation(np.array([[0.01, 0.02], [float("inf"), 0.01], [0.0, 0.03]]))


def test_ledoit_identity_psd():
    X = np.random.default_rng(0).normal(0, 0.01, size=(60, 4))
    S, d = ledoit_wolf_identity(X)
    assert np.all(np.isfinite(S))
    assert np.linalg.eigvalsh(S).min() >= -1e-12
    assert 0.0 <= d <= 1.0


def test_factor_covariance_single_factor():
    rng = np.random.default_rng(1)
    F = rng.normal(0, 0.01, size=(60, 1))
    R = F @ np.array([[1.2, 0.8]]) + rng.normal(0, 0.005, size=(60, 2))
    S = factor_covariance(R, F)
    assert S.shape == (2, 2)
    assert np.all(np.isfinite(S))
    assert np.linalg.eigvalsh(S).min() >= -1e-12


def test_factor_covariance_rejects_nan():
    rng = np.random.default_rng(2)
    F = rng.normal(0, 0.01, size=(60, 2))
    R = rng.normal(0, 0.01, size=(60, 3))
    R[0, 0] = float("nan")
    with pytest.raises(ValueError):
        factor_covariance(R, F)
