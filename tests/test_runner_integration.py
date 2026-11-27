"""Phase 1 Batch 2 runner integration tests.

Verify plumbing only: registration from real marks, $100k capital, model-ID
signal wiring through the blend path, covariance/MarketState reconciliation,
actual drawdown vs the mandate trigger, honest cost/baseline reporting, and
the NO AUTHORITY / NO TRADES outcome. No alpha or performance claim.
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_PATH = os.path.join(ROOT, "scripts", "run_simulation.py")
PANEL_PATH = os.path.join(ROOT, "evidence", "panel_real_v1.csv")

_spec = importlib.util.spec_from_file_location("run_simulation", RUNNER_PATH)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

SYMS = ["EEM", "EFA", "GLD", "IEF", "IWM", "QQQ", "SPY", "TLT", "USO", "UUP"]


def _synthetic_panel(n_days=140, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days, tz="UTC")
    rets = rng.normal(0.0004, 0.01, size=(n_days, len(SYMS)))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=SYMS)


def test_cold_start_no_authority_no_trades():
    res = rs.run_simulation(_synthetic_panel(), days=30, cash=10_000_000)
    s = res["summary"]
    assert s["no_authority"] is True
    assert s["intents_submitted_total"] == 0
    assert s["authority_blocks"] == s["days"]
    assert s["ending_nav_cents"] == 10_000_000      # all cash, no costs
    assert s["total_cost_cents"] == 0
    assert s["net_return"] == 0.0
    assert s["tracker_snapshot"]["n"] == 0
    assert s["tracker_snapshot"]["status"] == "probation"
    # No divergence latch: "blocked" is False; the no-authority state is in the note.
    assert (res["daily"]["blocked"] == False).all()
    assert res["daily"]["note"].str.contains(rs.AUTHORITY_NOTE).all()
    assert (res["daily"]["intents_submitted"] == 0).all()


def test_capital_default_is_100k():
    res = rs.run_simulation(_synthetic_panel(), days=10)  # default cash
    assert res["summary"]["starting_nav_cents"] == 10_000_000
    assert res["system"]["ledger"].state.nav() == 10_000_000


def test_registration_allowlist_scenarios_digest():
    res = rs.run_simulation(_synthetic_panel(), days=10)
    sx = res["system"]
    cfg = sx["cfg"]
    assert frozenset(SYMS) <= cfg.mandate.allowlist
    assert cfg.mandate.product_classes == frozenset({"PLAIN_ETF"})
    for sc in sx["scenarios"]:
        assert set(SYMS) <= set(sc)
    assert sx["ledger"].state.config_digest == cfg.mandate.digest()
    assert sx["ledger"].state.mandate_version == cfg.mandate.version
    # Risk limits preserved
    assert cfg.mandate.issuer_bps == 4_000
    assert cfg.mandate.drawdown_trigger_bps == 1_000


def test_drawdown_trigger_from_mandate_and_actual_drawdown():
    res = rs.run_simulation(_synthetic_panel(), days=15)
    d = res["daily"]
    assert (d["drawdown_trigger_bps"] == 1_000).all()   # not the old 5000
    assert (d["drawdown_bps"] == 0).all()               # all-cash -> no drawdown
    # Unit-check the actual drawdown math with a held position that drops.
    from quant_m1.system import build
    from quant_m1.universe import etf_instruments
    import tempfile
    sx = build(tempfile.mkdtemp(), cash=0, positions={"SPY": 100},
               marks={"SPY": 50_000}, instruments=etf_instruments(["SPY"]))
    dd = rs.prospective_drawdown_bps(sx["ledger"], {"SPY": 40_000})
    assert dd == 2_000  # (5.0M - 4.0M) / 5.0M = 20%


def test_covariance_and_market_state_reconciled(monkeypatch):
    import quant_m1.orchestrator.paper_loop as pl
    captured = {}
    real_run_cycle = pl.run_cycle

    def spy(state, trackers, forecasts=None, allocator=None, **kw):
        captured["fc"] = forecasts
        captured["ms"] = state
        return real_run_cycle(state, trackers, forecasts=forecasts, allocator=allocator, **kw)

    monkeypatch.setattr(pl, "run_cycle", spy)
    panel = _synthetic_panel()
    res = rs.run_simulation(panel, days=12, cov_window=63)
    fc, ms = captured["fc"], captured["ms"]

    # Blend path, not pre-blended expected_returns.
    assert "model_signals" in fc and rs.MODEL_ID in fc["model_signals"]
    assert "expected_returns" not in fc
    assert "current_weights" in fc and fc["band_frac"] == 0.25
    assert fc["horizon_days"] == 1

    # Covariance is the repository estimator on the same window, not 0.0002*I.
    cov = np.asarray(fc["cov"])
    assert not np.allclose(cov, np.eye(len(SYMS)) * 0.0002)
    assert np.any(cov[~np.eye(len(SYMS), dtype=bool)] != 0.0)

    # MarketState derived from the same trailing window (last captured day = last row).
    i = len(panel) - 1
    X = panel.iloc[i - 63:i].pct_change().dropna().values
    corr = np.corrcoef(X, rowvar=False)
    expected_corr = float(corr[~np.eye(len(SYMS), dtype=bool)].mean())
    assert ms.avg_corr == pytest.approx(expected_corr, abs=1e-9)
    assert ms.vol_ann != 0.15                      # not the old hardcoded value
    assert ms.drawdown_trigger_bps == 1_000


def test_baselines_present_and_correct():
    panel = _synthetic_panel()
    res = rs.run_simulation(panel, days=20, cost_bps=5)
    b = res["summary"]["baselines"]
    start_idx = len(panel) - 20
    gross_spy = float(panel["SPY"].iloc[-1] / panel["SPY"].iloc[start_idx] - 1.0)
    assert b["cash"] == 0.0
    assert b["benchmark_symbol"] == "SPY"
    assert b["spy_buy_hold_gross"] == pytest.approx(gross_spy, abs=1e-9)
    assert b["spy_buy_hold_net"] == pytest.approx(gross_spy - 0.0005, abs=1e-9)
    ew = float((panel.iloc[-1] / panel.iloc[start_idx] - 1.0).mean())
    assert b["equal_weight_gross"] == pytest.approx(ew, abs=1e-9)


def test_report_banner_and_no_success_message():
    res = rs.run_simulation(_synthetic_panel(), days=10)
    report = rs.format_report(res)
    assert "NO AUTHORITY / NO TRADES" in report
    assert "SYSTEM BEHAVIOR CORRECT" not in report
    assert "non-PIT" in report


@pytest.mark.skipif(not os.path.exists(PANEL_PATH), reason="real panel not present")
def test_real_panel_smoke():
    panel = rs.load_panel(PANEL_PATH)
    assert sorted(panel.columns.tolist()) == SYMS
    res = rs.run_simulation(panel, days=20, cash=10_000_000)
    s = res["summary"]
    assert s["no_authority"] is True
    assert s["ending_nav_cents"] == 10_000_000
    assert "NO AUTHORITY / NO TRADES" in rs.format_report(res)
