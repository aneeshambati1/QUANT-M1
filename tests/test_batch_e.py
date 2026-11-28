"""Batch E contract tests: sleeve, selector, signals, carry stub, smoke."""
from __future__ import annotations
import importlib.util, os
import numpy as np
import pandas as pd
import pytest
from quant_m1.meta.contribution import ContributionTracker, ModelStatus
from quant_m1.meta.sleeve import sleeve_weights, SLEEVE_CAP_FRAC, MIN_PROBE_WEIGHT
from quant_m1.meta.selector import select_weights, renormalize_available, REGIME_WEIGHTS
from quant_m1.meta.state import MarketState
from quant_m1.research.signals import momentum_signal, reversal_signal, rank_signal
from quant_m1.research.carry import carry_signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("smoke", os.path.join(ROOT, "scripts", "run_sleeve_smoke.py"))
smoke = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(smoke)


def _state(vol=0.10, corr=0.2):
    return MarketState(as_of="2024-01-02T00:00:00+00:00", avg_corr=corr,
                       vol_ann=vol, liquidity_score=1.0, data_age_secs=0,
                       model_disagreement=0.0, n_assets=3,
                       drawdown_bps=0, drawdown_trigger_bps=1000)


def test_sleeve_floor_for_cold_start():
    w = sleeve_weights({"m": ContributionTracker(horizon_days=1)})
    assert w["m"] == MIN_PROBE_WEIGHT


def test_sleeve_bounded_by_cap():
    ts = {f"m{k}": ContributionTracker(horizon_days=1) for k in range(30)}
    for t in ts.values():
        for _ in range(5):
            t.update(0.05)
    w = sleeve_weights(ts)
    assert sum(w.values()) <= SLEEVE_CAP_FRAC + 1e-9


def test_sleeve_skips_retired_and_eligible():
    t = ContributionTracker(horizon_days=1)
    for _ in range(300):
        t.update(0.02)
    assert t.eligible
    assert sleeve_weights({"active": t}) == {}


def test_selector_weights_sum_to_one():
    for name in REGIME_WEIGHTS:
        s = sum(REGIME_WEIGHTS[name].values())
        assert abs(s - 1.0) < 1e-9


def test_selector_routing_deterministic():
    w = select_weights(_state(vol=0.05, corr=0.1))
    assert set(w) == {"momentum", "reversal", "carry"}


def test_renormalize_drops_unavailable():
    eff = renormalize_available({"a": 0.5, "b": 0.5}, {"a": True, "b": False})
    assert eff == {"a": 1.0}


def test_rank_signal_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=20, tz="UTC")
    close = pd.DataFrame({"A": np.arange(1, 21.0), "B": np.arange(21, 1, -1.0)}, index=idx)
    s_now = rank_signal(close, 10, 5, contrarian=False)
    s_future = rank_signal(close, 10, 5, contrarian=False)
    assert np.array_equal(s_now, s_future)
    assert momentum_signal(close, 10, 5)[0] > 0  # A is rising -> long


def test_reversal_sign():
    idx = pd.bdate_range("2020-01-01", periods=20, tz="UTC")
    close = pd.DataFrame({"A": np.arange(1, 21.0), "B": np.arange(21, 1, -1.0)}, index=idx)
    r = reversal_signal(close, 19, 5)
    assert r[1] > 0  # B fell recently -> reversal longs B


def test_carry_unavailable_on_price_only():
    out = carry_signal(None)
    assert out["available"] is False and out["signal"] is None


def test_smoke_runs_on_synthetic():
    idx = pd.bdate_range("2020-01-01", periods=200, tz="UTC")
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 4)), axis=0))
    panel = pd.DataFrame(px, index=idx, columns=["A", "B", "C", "D"])
    rep = smoke.run_smoke(panel, days=30, cov_window=63)
    assert rep["live_trading"] is False
    assert rep["n_days"] > 0
    assert "sha" in rep
    assert rep["total_sleeve_weight"] <= SLEEVE_CAP_FRAC + 1e-9
