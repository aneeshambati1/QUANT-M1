"""IC estimator acceptance tests.

Every fixture here is SYNTHETIC. These tests validate the estimator, not a
strategy. No result in this file is evidence of a tradeable edge.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.research.ic import (
    ICError, ic_series, newey_west_se, ar1_rho, effective_n,
    measure_ic, synthetic_ic_fixture,
)

LABEL = "SYNTHETIC - NOT AN EDGE CLAIM"


def test_recovers_known_positive_ic():
    s, r = synthetic_ic_fixture(400, 30, 0.6, seed=7)
    rep = measure_ic(s, r, horizon="1d", evidence_label=LABEL)
    assert rep.mean_ic > 0.4
    assert rep.t_hac > 3.0
    assert rep.evidence_label == LABEL


def test_zero_signal_gives_near_zero_ic():
    rng = np.random.default_rng(11)
    s = rng.standard_normal((300, 20))
    r = rng.standard_normal((300, 20))
    rep = measure_ic(s, r, horizon="1d", evidence_label=LABEL)
    assert abs(rep.mean_ic) < 0.05
    assert abs(rep.t_hac) < 3.0


def test_negative_ic_is_detected():
    s, r = synthetic_ic_fixture(300, 25, 0.6, seed=3)
    rep = measure_ic(s, -r, horizon="1d", evidence_label=LABEL)
    assert rep.mean_ic < -0.4


def test_hac_se_exceeds_naive_under_positive_autocorrelation():
    rng = np.random.default_rng(5)
    n = 600
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    x = x + 0.5
    naive = x.std(ddof=1) / np.sqrt(n)
    hac = newey_west_se(x, lags=8)
    assert hac > naive
    assert ar1_rho(x) > 0.3


def test_effective_n_shrinks_with_positive_rho():
    assert effective_n(100, 0.5) == pytest.approx(33.333, abs=1e-3)
    assert effective_n(100, 0.0) == pytest.approx(100.0)
    assert effective_n(100, -0.5) == pytest.approx(300.0)


def test_overlapping_returns_reduce_t_stat():
    """Serial dependence must lower the corrected t-statistic."""
    rng = np.random.default_rng(19)
    n = 500
    base = rng.standard_normal(n) * 0.02 + 0.05
    smooth = np.convolve(base, np.ones(5) / 5.0, mode="same")
    naive = smooth.std(ddof=1) / np.sqrt(n)
    hac = newey_west_se(smooth, lags=6)
    assert (smooth.mean() / hac) < (smooth.mean() / naive)


def test_constant_row_is_dropped_not_crashed():
    s, r = synthetic_ic_fixture(50, 10, 0.5, seed=2)
    s[0] = 1.0
    rep = measure_ic(s, r, horizon="1d", evidence_label=LABEL)
    assert rep.n_dropped == 1
    assert rep.n_obs == 49


def test_report_carries_horizon_and_label():
    s, r = synthetic_ic_fixture(60, 10, 0.4, seed=1)
    rep = measure_ic(s, r, horizon="5d", evidence_label=LABEL)
    d = rep.as_dict()
    assert d["horizon"] == "5d"
    assert d["evidence_label"] == LABEL
    import json
    json.dumps(d)


def test_rejects_bad_inputs():
    s, r = synthetic_ic_fixture(30, 5, 0.3, seed=1)
    with pytest.raises(ICError):
        measure_ic(s, r[:, :-1], horizon="1d", evidence_label=LABEL)
    with pytest.raises(ICError):
        measure_ic(s, r, horizon="", evidence_label=LABEL)
    with pytest.raises(ICError):
        measure_ic(s, r, horizon="1d", evidence_label="")
    with pytest.raises(ICError):
        ic_series(np.zeros((10, 1)), np.zeros((10, 1)))
    with pytest.raises(ICError):
        newey_west_se(np.ones(5), lags=5)


def test_ic_series_bounds():
    s, r = synthetic_ic_fixture(80, 12, 0.9, seed=4)
    x = ic_series(s, r)
    assert np.all(x[np.isfinite(x)] >= -1.0 - 1e-9)
    assert np.all(x[np.isfinite(x)] <= 1.0 + 1e-9)