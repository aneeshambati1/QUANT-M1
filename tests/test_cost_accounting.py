"""Cost attribution tests. Accounting identities and no-lookahead."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from quant_m1.research.cost_accounting import (
    POLICIES, COST_SCENARIOS_BP, AccountingError, drift, traded_fraction,
    equal_target, run_policy, summarize, attribute,
)


def _rets(T=300, N=5, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((T, N)) * 0.01
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(r, index=idx, columns=[f"S{i}" for i in range(N)])


def test_drift_preserves_weight_sum():
    w = equal_target(4)
    r = np.array([0.02, -0.01, 0.0, 0.03])
    assert drift(w, r).sum() == pytest.approx(1.0)


def test_traded_fraction_counts_both_legs():
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.4, 0.5, 0.1])
    assert traded_fraction(a, b) == pytest.approx(0.2)


def test_buy_hold_has_zero_turnover():
    res = run_policy(_rets(), "buy_hold")
    assert res["traded"].sum() == pytest.approx(0.0)


def test_daily_trades_more_than_monthly():
    r = _rets()
    d = run_policy(r, "daily")["traded"].sum()
    m = run_policy(r, "monthly_21d")["traded"].sum()
    assert d > m > 0.0


def test_band_trades_less_than_daily():
    r = _rets()
    b = run_policy(r, "band_rel25")["traded"].sum()
    d = run_policy(r, "daily")["traded"].sum()
    assert b < d


def test_chase_proxy_has_highest_turnover():
    r = _rets()
    ch = run_policy(r, "chase_top3_wk")["traded"].sum()
    mo = run_policy(r, "monthly_21d")["traded"].sum()
    assert ch > mo


def test_cost_is_linear_in_bp():
    res = run_policy(_rets(), "monthly_21d")
    a = summarize(res, 1.0)["annual_cost_bp"]
    b = summarize(res, 3.0)["annual_cost_bp"]
    assert b == pytest.approx(3.0 * a)


def test_higher_cost_lowers_net_return():
    res = run_policy(_rets(), "daily")
    lo = summarize(res, 1.0)["net_ann_return"]
    hi = summarize(res, 10.0)["net_ann_return"]
    assert hi < lo


def test_weights_valid_every_period():
    for p in POLICIES:
        W = run_policy(_rets(), p)["weights"]
        assert np.all(W >= -1e-12)
        assert np.allclose(W.sum(axis=1), 1.0)


def test_no_lookahead_in_rebalance_decisions():
    r = _rets()
    a = run_policy(r, "band_rel25")["traded"]
    altered = r.copy()
    altered.iloc[200:] = altered.iloc[200:] * 3.0
    b = run_policy(altered, "band_rel25")["traded"]
    assert np.allclose(a[:200], b[:200])


def test_all_cells_present_and_json_safe():
    out = attribute(_rets())
    assert len(out["rows"]) == len(POLICIES) * len(COST_SCENARIOS_BP)
    json.dumps(out)


def test_rejects_bad_inputs():
    with pytest.raises(AccountingError):
        run_policy(_rets(), "nope")
    with pytest.raises(AccountingError):
        run_policy(_rets(T=30), "daily")
    with pytest.raises(AccountingError):
        summarize(run_policy(_rets(), "daily"), -1.0)