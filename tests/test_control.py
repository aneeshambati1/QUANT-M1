"""Instrument controls at the real panel shape. SYNTHETIC - no edge claim."""
from __future__ import annotations

import pytest

from quant_m1.research.control import (
    CONTROLS, MARKET_TRIAL_COUNT, LOOKBACK, HORIZON, N_SESSIONS, N_ASSETS,
    synthetic_edge_panel, run_control, run_all_controls,
)
from quant_m1.research.governor import TrialLedger, bonferroni_threshold


def test_control_panel_matches_market_shape():
    p = synthetic_edge_panel(0.05)
    assert p.close.shape == (N_SESSIONS, N_ASSETS)
    assert "SYNTHETIC" in p.label


def test_zero_edge_is_not_detected(tmp_path):
    out = run_control("zero", str(tmp_path / "c.jsonl"),
                      str(tmp_path / "raw"), "abc123")
    assert abs(out["report"].mean_ic) < 0.01
    assert abs(out["report"].t_hac) < 1.0
    assert out["gate_at_market_N"]["passed"] is False


def test_strong_edge_is_detected(tmp_path):
    """POSITIVE CONTROL: the assembled pipeline must see a real edge."""
    out = run_control("strong", str(tmp_path / "c.jsonl"),
                      str(tmp_path / "raw"), "abc123")
    assert out["report"].mean_ic > 0.07
    assert out["gate_at_market_N"]["passed"] is True


def test_tradeable_edge_is_missed_documents_underpower(tmp_path):
    """A genuine 0.030 IC exists but is NOT resolvable at N=13.

    This is the empirical basis for labelling the market null INCONCLUSIVE
    rather than 'no edge'. If this test ever starts passing the gate, the
    power conclusion must be revisited.
    """
    out = run_control("tradeable_003", str(tmp_path / "c.jsonl"),
                      str(tmp_path / "raw"), "abc123")
    assert 0.02 < out["report"].mean_ic < 0.04
    assert out["gate_at_market_N"]["passed"] is False
    assert abs(out["report"].t_hac) < bonferroni_threshold(MARKET_TRIAL_COUNT)


def test_controls_use_separate_family_and_ledger(tmp_path):
    led_path = str(tmp_path / "c.jsonl")
    run_all_controls(led_path, str(tmp_path / "raw"), "abc123")
    led = TrialLedger(led_path)
    assert led.n_trials() == 3
    assert all(t["family"] == "CONTROL_SYNTHETIC" for t in led.trials())
    assert all(t["content_hash"] for t in led.trials())


def test_control_summary_shape(tmp_path):
    s = run_all_controls(str(tmp_path / "c.jsonl"), str(tmp_path / "raw"), "abc")
    assert s["market_trial_count"] == MARKET_TRIAL_COUNT
    assert len(s["controls"]) == 3
    det = {c["name"]: c["detected_at_market_N"] for c in s["controls"]}
    assert det["ctrl_strong"] is True
    assert det["ctrl_zero"] is False
    assert det["ctrl_tradeable_003"] is False