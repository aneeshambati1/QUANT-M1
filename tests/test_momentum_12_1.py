"""Deterministic checks for the 12-1 / reversal exploratory trial."""
from __future__ import annotations
import importlib.util
import os
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "mom121", os.path.join(ROOT, "experiments", "momentum_12_1.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _panel(n=600, seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="UTC")
    cols = ["A", "B", "C", "D", "E"]
    px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, (n, 5)), axis=0))
    return pd.DataFrame(px, index=idx, columns=cols)


def test_formation_return_math():
    idx = pd.bdate_range("2015-01-01", periods=300, tz="UTC")
    close = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
    # t=280, skip=21, lookback=252 -> a=close[259], b=close[7]
    fr = m.formation_return(close, 280, 252, 21)
    expected = close.iloc[259]["A"] / close.iloc[7]["A"] - 1.0
    assert abs(fr["A"] - expected) < 1e-12


def test_hac_tstat_finite_and_sign():
    x = np.full(100, 0.001) + np.random.default_rng(0).normal(0, 0.001, 100)
    t = m.hac_tstat(x, 3)
    assert np.isfinite(t) and t > 0
    assert np.isnan(m.hac_tstat(np.array([0.0, 0.0]), 3))


def test_short_panel_fails_closed():
    with pytest.raises(ValueError):
        m.run_trial(_panel(n=100))


def test_run_trial_structure_and_no_lookahead():
    rep = m.run_trial(_panel(n=600))
    assert "momentum_12_1" in rep and "reversal_5" in rep
    assert rep["momentum_12_1"]["n_holds"] > 0
    assert "sha" in rep
    # no-lookahead: formation at t uses only data <= t-skip (verified in math test)
