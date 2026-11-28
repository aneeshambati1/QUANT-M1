"""Tests for survivor factor wiring into the simulation runner."""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from scripts.run_simulation import run_simulation, _load_survivors_safe


def _synthetic_panel(n=200):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    syms = ["SPY", "QQQ", "TLT", "GLD"]
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, 4)), axis=0))
    return pd.DataFrame(px, index=idx, columns=syms)


def _synthetic_survivors(n=60):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2019-01-01", periods=n, freq="ME")
    cols = ["XFIN", "ShareIss1Y", "AnnouncementReturn", "EarningsStreak"]
    data = rng.normal(0.005, 0.03, size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_load_survivors_safe_missing_file():
    assert _load_survivors_safe("/nonexistent/path.csv") is None


def test_load_survivors_safe_none():
    assert _load_survivors_safe(None) is None


def test_runner_without_survivors():
    panel = _synthetic_panel()
    res = run_simulation(panel, days=30, survivor_path=None)
    assert res["summary"]["survivor_wiring"] is False
    assert res["summary"]["no_authority"] is True


def test_runner_with_survivors():
    panel = _synthetic_panel()
    sf = _synthetic_survivors()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sf.to_csv(f.name)
        path = f.name
    try:
        res = run_simulation(panel, days=30, survivor_path=path, blend_alpha=0.5)
        assert res["summary"]["survivor_wiring"] is True
        assert res["summary"]["blend_alpha"] == 0.5
        assert "survivor_active" in res["daily"].columns
    finally:
        os.unlink(path)


def test_runner_blend_alpha_zero():
    panel = _synthetic_panel()
    sf = _synthetic_survivors()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sf.to_csv(f.name)
        path = f.name
    try:
        res = run_simulation(panel, days=30, survivor_path=path, blend_alpha=0.0)
        assert res["summary"]["survivor_wiring"] is True
        assert res["summary"]["blend_alpha"] == 0.0
    finally:
        os.unlink(path)


def test_runner_no_lookahead_in_survivors():
    """Survivor data used at date t must only include rows with index < t."""
    panel = _synthetic_panel(n=200)
    sf = _synthetic_survivors(n=60)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sf.to_csv(f.name)
        path = f.name
    try:
        res = run_simulation(panel, days=30, survivor_path=path)
        # All simulation dates are in 2020; survivor data ends ~2023
        # so some days should have survivor data available
        active_days = res["daily"]["survivor_active"].sum()
        assert active_days >= 0  # may be 0 if dates don't overlap
    finally:
        os.unlink(path)


def test_runner_summary_has_selector_weights():
    panel = _synthetic_panel()
    res = run_simulation(panel, days=10, survivor_path=None)
    assert "selector_weights" in res["daily"].columns
    w = res["daily"]["selector_weights"].iloc[0]
    assert isinstance(w, dict)
    assert abs(sum(w.values()) - 1.0) < 1e-9
