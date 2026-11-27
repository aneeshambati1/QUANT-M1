"""Phase 2 mechanics tests (fast). Calibration claims live in run_null_suite.py."""
from __future__ import annotations
import math
import tempfile
import numpy as np
import pytest

from quant_m1.system import build
from quant_m1.universe import etf_instruments
from quant_m1.meta.contribution import ContributionTracker, se_weights, se_raw_weights
from quant_m1.meta.probe import assign_families, probe_coefficient, _ProbeGate
from quant_m1.orchestrator.paper_loop import _shadow_system
from quant_m1.research.promotion_ledger import PromotionLedger, nppf

SYMS = ["SPY", "QQQ", "TLT"]
MARKS = {"SPY": 50_000, "QQQ": 45_000, "TLT": 9_000}


def _sx(tmp_path):
    return build(str(tmp_path), cash=10_000_000, positions={}, marks=dict(MARKS),
                 instruments=etf_instruments(SYMS))


def test_shadow_config_and_universe_sync(tmp_path):
    sx = _sx(tmp_path)
    st = sx["ledger"].state
    start = {"cash_settled": st.cash_settled, "positions": dict(st.positions),
             "marks": dict(st.marks), "receivables": tuple(st.receivables),
             "peak_nav": st.peak_nav, "trade_date": st.trade_date,
             "mark_wall": st.mark_wall}
    sh = _shadow_system(sx, start, tempfile.mkdtemp())
    assert sh["cfg"] is sx["cfg"]
    assert sh["kernel"].cfg is sx["cfg"] and sh["gw"].cfg is sx["cfg"]
    assert sh["ledger"].state.config_digest == sx["cfg"].mandate.digest()
    assert sh["ledger"].state.mandate_version == sx["cfg"].mandate.version
    assert set(SYMS) <= set(sh["instruments"])
    for sc in sh["scenarios"]:
        assert set(SYMS) <= set(sc)


def test_se_stats_fail_closed_and_finite(tmp_path):
    t = ContributionTracker()
    assert t.se_stats(2.0)["valid"] is False  # insufficient obs
    for _ in range(130):
        t.update(0.0)  # constant series -> gamma0 = 0 -> fail-closed
    st = t.se_stats(2.0)
    assert st["valid"] is False and st["reason"] == "nonpositive_gamma0"
    t2 = ContributionTracker()
    rng = np.random.default_rng(0)
    for _ in range(399):
        t2.update(float(rng.normal(0.001, 0.01)))
    st2 = t2.se_stats(2.865)
    assert st2["valid"] and math.isfinite(st2["se_lcb"]) and st2["n_eff_corr"] > 0


def test_original_rule_unchanged():
    t = ContributionTracker()
    for _ in range(300):
        t.update(0.02)
    assert t.eligible and t.kappa == 1.0
    assert t.lcb == pytest.approx(t.mean - t.std)


def test_duplicate_family_guard():
    s = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 4.0, 6.0]),
         "c": np.array([3.0, 1.0, 2.0])}
    fam = assign_families(s)
    assert fam["b"] == "a" and fam["a"] == "a"
    assert probe_coefficient("a", fam, {}) == 0.05
    assert probe_coefficient("b", fam, {}) == 0.0   # duplicate -> zero coefficient
    assert probe_coefficient("a", fam, {"a": 0.5}) == 0.0  # rep already active


def test_se_weights_all_zero_and_caps():
    ts = {"m1": ContributionTracker(), "m2": ContributionTracker()}
    w = se_weights(ts, 2.865)
    assert all(v == 0.0 for v in w.values())  # no history -> cash
    rng = np.random.default_rng(1)
    big = ContributionTracker()
    for _ in range(399):
        big.update(float(rng.normal(0.01, 0.001)))
    w2 = se_weights({"m1": big}, 2.865, c_model=0.40)
    assert w2["m1"] <= 0.40 + 1e-9  # remainder -> cash


def test_ledger_one_test_and_lifetime_closure(tmp_path):
    led = PromotionLedger(str(tmp_path / "ledger.json"))
    assert led.z_cell == pytest.approx(2.865, abs=0.01)
    led.register("m1", 1, "famA")
    t = ContributionTracker()
    rng = np.random.default_rng(2)
    for _ in range(399):
        t.update(float(rng.normal(0.0, 0.01)))  # zero edge -> fail
    rep = led.evaluate("m1", 1, t)
    assert rep["promoted"] is False
    with pytest.raises(ValueError):
        led.evaluate("m1", 1, t)          # no re-test
    with pytest.raises(ValueError):
        led.register("m1_dup", 1, "famA")  # family closed for lifetime
    with pytest.raises(ValueError):
        led.register("m2", 1, "famB") or led.evaluate("m2", 1, t)  # n check path
    led2 = PromotionLedger(str(tmp_path / "ledger.json"))
    assert "famA" in led2.closed_families  # persisted


def test_probe_gate_is_research_only():
    g = _ProbeGate(1)
    assert g.eligible and g.raw_weight() == 1.0
    assert g.snapshot()["probe_gate"] is True
