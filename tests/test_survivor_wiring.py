"""Tests for survivor factor wiring into selector."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from quant_m1.meta.selector import (
    select_weights, select_weights_with_survivors, renormalize_available,
    REGIME_WEIGHTS,
)
from quant_m1.research.survivor_factors import (
    compute_factor_signals, blend_factor_signals, rank_families,
    factor_integration_report, FACTOR_FAMILIES, REGIME_FAMILY_WEIGHTS,
)


class FakeState:
    def __init__(self, regime_val):
        self.regime = type("R", (), {"value": regime_val})()


def _synthetic_factors(n=60):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2015-01-01", periods=n, freq="ME")
    cols = ["XFIN", "ShareIss1Y", "AnnouncementReturn", "EarningsStreak", "ShortInterest"]
    data = rng.normal(0.005, 0.03, size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_select_weights_unchanged_without_survivors():
    state = FakeState("calm")
    w = select_weights(state)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w == REGIME_WEIGHTS["calm"]


def test_select_weights_with_none_survivors():
    state = FakeState("calm")
    w = select_weights_with_survivors(state, None)
    assert w == REGIME_WEIGHTS["calm"]


def test_select_weights_with_survivors_sums_to_one():
    state = FakeState("elevated")
    df = _synthetic_factors()
    rep = factor_integration_report(df, regime="elevated")
    w = select_weights_with_survivors(state, rep, blend_alpha=0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())


def test_blend_alpha_zero_equals_base():
    state = FakeState("calm")
    df = _synthetic_factors()
    rep = factor_integration_report(df)
    w = select_weights_with_survivors(state, rep, blend_alpha=0.0)
    assert w == REGIME_WEIGHTS["calm"]


def test_blend_alpha_one_pure_empirical():
    state = FakeState("calm")
    df = _synthetic_factors()
    rep = factor_integration_report(df)
    w = select_weights_with_survivors(state, rep, blend_alpha=1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_rank_families_descending():
    scores = {"a": 0.5, "b": 0.1, "c": 0.9}
    ranked = rank_families(scores)
    assert ranked[0][0] == "c"
    assert ranked[-1][0] == "b"


def test_factor_report_no_live_flags():
    df = _synthetic_factors()
    rep = factor_integration_report(df)
    assert rep["live_trading"] is False
    assert rep["grants_authority"] is False
    assert rep["phase2_enabled"] is False


def test_family_weights_sum_to_one():
    for regime in REGIME_FAMILY_WEIGHTS:
        w = REGIME_FAMILY_WEIGHTS[regime]
        assert abs(sum(w.values()) - 1.0) < 1e-9
