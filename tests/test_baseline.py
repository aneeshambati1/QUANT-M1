"""Baseline plumbing tests. SYNTHETIC fixtures only.

These validate alignment and no-lookahead. They are NOT evidence about the
pre-registered hypothesis and must never be cited as an edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import panel_from_frame, DataError
from quant_m1.research.baseline import (
    momentum_signal, forward_return, run_baseline, PRE_REGISTERED, UNIVERSE,
)

LABEL = "SYNTHETIC - NOT AN EDGE CLAIM"


def _panel(n=200, k=5, seed=0, phi=0.0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, k)) * 0.01
    if phi:
        for t in range(1, n):
            r[t] = phi * r[t - 1] + rng.standard_normal(k) * 0.01
    px = 100.0 * np.exp(np.cumsum(r, axis=0))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])
    return panel_from_frame(df, source="synthetic", vintage="test",
                            pit=False, label=LABEL)


def test_pre_registration_is_frozen():
    assert PRE_REGISTERED["lookback_sessions"] == 21
    assert PRE_REGISTERED["horizon_sessions"] == 5
    assert PRE_REGISTERED["trials_planned"] == 1
    assert len(UNIVERSE) == 10


def test_signal_matches_manual_trailing_return():
    p = _panel()
    s = momentum_signal(p.close, 5)
    t = 40
    manual = p.close.iloc[t] / p.close.iloc[t - 5] - 1.0
    assert np.allclose(s.iloc[t].to_numpy(), manual.to_numpy())


def test_signal_has_no_lookahead():
    p = _panel()
    s1 = momentum_signal(p.close, 5)
    altered = p.close.copy()
    altered.iloc[40:] = altered.iloc[40:] * 3.0
    s2 = momentum_signal(altered, 5)
    pd.testing.assert_frame_equal(s1.iloc[:40], s2.iloc[:40])


def test_forward_return_uses_future_only():
    p = _panel()
    f = forward_return(p.close, 5)
    t = 30
    manual = p.close.iloc[t + 5] / p.close.iloc[t] - 1.0
    assert np.allclose(f.iloc[t].to_numpy(), manual.to_numpy())
    assert f.iloc[-5:].isna().all().all()


def test_tail_rows_are_dropped_not_zero_filled():
    p = _panel(n=120)
    rep = run_baseline(p, lookback=5, horizon=5)
    assert rep.n_obs <= 120 - 5 - 5


def test_persistent_returns_produce_positive_ic():
    """Plumbing check with phi=0.6, lookback=5, horizon=1. NOT the hypothesis."""
    p = _panel(n=400, k=8, seed=3, phi=0.6)
    rep = run_baseline(p, lookback=5, horizon=1)
    assert rep.mean_ic > 0.05
    assert rep.evidence_label == LABEL


def test_report_is_labelled_and_horizon_tagged():
    p = _panel()
    rep = run_baseline(p, lookback=21, horizon=5)
    assert rep.horizon == "5d"
    assert "SYNTHETIC" in rep.evidence_label


def test_loader_rejects_bad_panels():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    good = pd.DataFrame(np.ones((100, 3)), index=idx, columns=list("ABC"))
    with pytest.raises(DataError):
        panel_from_frame(good.iloc[::-1], source="x", vintage="v", pit=False, label=LABEL)
    with pytest.raises(DataError):
        panel_from_frame(good.iloc[:, :1], source="x", vintage="v", pit=False, label=LABEL)
    bad = good.copy(); bad.iloc[5, 0] = -1.0
    with pytest.raises(DataError):
        panel_from_frame(bad, source="x", vintage="v", pit=False, label=LABEL)
    with pytest.raises(DataError):
        panel_from_frame(good, source="x", vintage="v", pit=False, label="")


def test_insufficient_sessions_rejected():
    p = _panel(n=45)
    with pytest.raises(DataError):
        run_baseline(p, lookback=21, horizon=5)