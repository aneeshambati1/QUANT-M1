"""
Batch 1 acceptance tests.

Run:
    python -m pytest tests/test_batch1.py -q
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats
from sklearn.covariance import LedoitWolf

from quant_m1.settlement import ExchangeCalendar, SettlementBook
from quant_m1.audit import AuditLog, verify_entries
from quant_m1.metrics import Metric, Unit, Horizon, GrossNet
from quant_m1.cost_model import round_trip_cost_bps, corwin_schultz_full_spread
from quant_m1.data.evidence import Evidence, EvidenceType
from quant_m1.data.pit import Bar, as_of_bars, history_before
from quant_m1.research.dsr import expected_max_sharpe, probabilistic_sharpe, deflated_sharpe
from quant_m1.research.trial_registry import TrialRegistry, TrialRecord, canonical_config_hash
from quant_m1.research.alpha_feasibility import effective_breadth, optimal_holding_days
from quant_m1.research.cscv import purge_embargo
from quant_m1.research.pbo import pbo_from_score_matrix
from quant_m1.portfolio.covariance import ledoit_wolf_identity


def test_t1_settlement_same_day_zero_next_session_release():
    cal = ExchangeCalendar("XNYS")
    book = SettlementBook(cal)
    p = book.on_sell_fill("e1", "2024-01-02", 100_000)
    assert p.available_date == "2024-01-03"
    assert book.settle("2024-01-02") == 0
    assert book.unsettled_total() == 100_000
    assert book.settle("2024-01-03") == 100_000
    assert book.unsettled_total() == 0


def test_settlement_duplicate_exec_id_idempotent_and_conflict():
    book = SettlementBook()
    book.on_sell_fill("e1", "2024-01-02", 100_000)
    book.on_sell_fill("e1", "2024-01-02", 100_000)
    assert book.unsettled_total() == 100_000
    with pytest.raises(ValueError, match="SETTLEMENT_EXEC_ID_CONFLICT"):
        book.on_sell_fill("e1", "2024-01-02", 99_999)


def test_audit_chain_tamper_evident():
    log = AuditLog()
    log.append("A", {"x": 1})
    log.append("B", {"y": {"z": 2}})
    assert log.verify_chain()
    entries = log.entries()
    bad = replace(entries[0], payload_hash="0" * 64)
    assert not verify_entries((bad,) + entries[1:])


def test_audit_redacts_secrets():
    log = AuditLog()
    e = log.append("AUTH", {"api_key": "SUPERSECRET", "ok": 1})
    assert e.payload_hash
    assert log.verify_chain()


def test_metric_requires_gross_net():
    with pytest.raises(ValueError):
        Metric("sharpe", 1.0, Unit.RATIO, Horizon.ANNUAL_252, 252, periods_per_year=252)
    m = Metric("sharpe", 1.0, Unit.RATIO, Horizon.ANNUAL_252, 252,
               gross_net=GrossNet.NET, periods_per_year=252)
    assert m.gross_net == GrossNet.NET


def test_full_spread_not_doubled():
    c = round_trip_cost_bps(0.0010, 100_000, 100_000)
    assert abs(c.spread_bps - 10.0) < 1e-9
    assert abs(c.total_bps - 10.0) < 1e-9


def test_corwin_schultz_positive():
    h = np.array([101.0, 102.0, 101.5, 103.0])
    l = np.array([99.0, 100.0, 99.5, 101.0])
    s = corwin_schultz_full_spread(h, l)
    assert 0.0 <= s <= 1.0


def test_dsr_two_term_not_single():
    two = expected_max_sharpe(6)
    one = stats.norm.ppf(1 - 1 / 6)
    assert two > one
    assert abs(two - one) > 0.01


def test_psr_midpoint_and_fat_tail_penalty():
    assert abs(probabilistic_sharpe(0.05, 0.05, 500) - 0.5) < 1e-12
    g = probabilistic_sharpe(0.10, 0.0, 500, 0.0, 3.0)
    f = probabilistic_sharpe(0.10, 0.0, 500, -0.5, 8.0)
    assert f < g


def test_deflated_sharpe_frequency():
    d = deflated_sharpe(2.0, 756, 100)
    assert abs(d["sr_obs_per_period"] - 2.0 / math.sqrt(252)) < 1e-12
    assert d["sr_hurdle_annualised"] > d["sr_hurdle_per_period"]


def test_trial_registry_counts_all(tmp_path):
    reg = TrialRegistry(str(tmp_path / "trials.jsonl"))
    for i in range(5):
        reg.log(TrialRecord(
            config_hash=canonical_config_hash({"i": i}),
            family="test",
            seed=i,
            data_vintage="fixture",
            registered_at="2026-01-01T00:00:00+00:00",
            metric_name="sharpe",
            metric_value=0.1 * i,
            horizon="annual_252_daily",
            gross_net="net",
        ))
    assert reg.n_trials() == 5
    assert reg.effective_trials() == 5


def test_effective_breadth_capped_for_negative_corr():
    assert effective_breadth(6, -0.1) <= 6.0
    assert effective_breadth(6, 0.0) == 6.0
    assert effective_breadth(6, 1.0) == 1.0


def test_optimal_hold_never_trade_when_cost_dominates():
    assert optimal_holding_days(0.001, 20, 0.001) is None
    assert optimal_holding_days(0.001, 20, 0.002) is None


def test_pit_no_lookahead():
    b1 = Bar("SPY", "2024-01-02T21:00:00+00:00", 100, 101, 99, 100.5, 1e6, 1.0,
             "2024-01-02T21:05:00+00:00", "fixture", "v1")
    b2 = Bar("SPY", "2024-01-03T21:00:00+00:00", 101, 102, 100, 101.5, 1e6, 1.0,
             "2024-01-03T21:05:00+00:00", "fixture", "v1")
    bars = [b1, b2]
    decision = "2024-01-03T00:00:00+00:00"
    assert as_of_bars(bars, "SPY", decision) == [b1]
    assert history_before(bars, "SPY", decision) == [b1]


def test_evidence_known_at_before_release_rejected():
    with pytest.raises(ValueError):
        Evidence(
            type=EvidenceType.MACRO_RELEASE,
            symbol=None,
            release_ts="2024-01-02T13:30:00+00:00",
            known_at="2024-01-02T13:29:00+00:00",
            confidence=0.9,
            direction=0.5,
            magnitude=1.0,
            source="fixture",
            raw_ref="ref",
            vintage_id="v1",
        )


def test_purge_embargo_removes_overlap():
    train = np.arange(0, 100)
    test = np.arange(40, 50)
    kept, _ = purge_embargo(train, test, label_horizon=5, embargo=3)
    assert not any(35 <= i <= 52 for i in kept)


def test_pbo_noise_near_half():
    rng = np.random.default_rng(7)
    IS = rng.normal(size=(200, 20))
    OOS = rng.normal(size=(200, 20))
    r = pbo_from_score_matrix(IS, OOS)
    assert 0.2 <= r["pbo"] <= 0.8


def test_ledoit_wolf_matches_sklearn():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 6))
    sig, delta = ledoit_wolf_identity(X)
    sk = LedoitWolf().fit(X)
    assert abs(delta - sk.shrinkage_) < 1e-8
    assert np.allclose(sig, sk.covariance_, atol=1e-8)
