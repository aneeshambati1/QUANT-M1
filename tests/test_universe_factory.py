"""Phase 1 plumbing tests: universe construction, config authority, scenario
coverage, capital accounting, cold-start no-authority behavior. No alpha claims.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pytest
from quant_m1.system import build
from quant_m1.universe import (make_etf_instrument, etf_instruments,
    register_etf_universe, ETF_PRODUCT_CLASS)
from quant_m1.core import Instrument, WorkingOrder, BPS
from quant_m1.authority import Intent, Reject
from quant_m1.risk import evaluate
from quant_m1.meta.contribution import ContributionTracker
from quant_m1.meta.state import MarketState
from quant_m1.orchestrator.cycle import run_cycle
from quant_m1.allocate import solve_allocation

SYMS = ["SPY", "QQQ", "TLT"]
MARKS = {"SPY": 50_000, "QQQ": 45_000, "TLT": 9_000}
LEGACY_MARKS = {"A": 10_000, "B": 10_000, "C": 10_000}
CASH_100K = 10_000_000

def _build_etf(tmp_path, cash=CASH_100K):
    return build(str(tmp_path), cash=cash, positions={}, marks=dict(MARKS),
                 instruments=etf_instruments(SYMS))

def test_etf_instrument_types_and_unknown_bucket():
    inst = make_etf_instrument("SPY")
    assert isinstance(inst, Instrument)
    assert inst.product_class == ETF_PRODUCT_CLASS == "PLAIN_ETF"
    assert inst.holdings is None
    assert inst.lookthrough() == (("UNK:ALL", BPS),)
    assert isinstance(inst.adv, int) and isinstance(inst.beta_bps, int)

def test_build_time_allowlist_and_digest(tmp_path):
    sx = _build_etf(tmp_path)
    cfg, ledger = sx["cfg"], sx["ledger"]
    assert cfg.mandate.allowlist == frozenset(SYMS)
    assert cfg.mandate.version == 1
    assert cfg.mandate.product_classes == frozenset({"PLAIN_ETF"})
    assert ledger.state.mandate_version == 1
    assert ledger.state.config_digest == cfg.mandate.digest()
    assert cfg.attested()

def test_build_requires_marks_for_custom_instruments(tmp_path):
    with pytest.raises(ValueError):
        build(str(tmp_path), instruments=etf_instruments(SYMS), marks=None)

def test_build_rejects_product_class_mismatch(tmp_path):
    inst = etf_instruments(SYMS)
    inst["SPY"] = Instrument("SPY", "EQUITY", 1_000_000_000, 10_000, None)
    with pytest.raises(ValueError):
        build(str(tmp_path), instruments=inst, marks=dict(MARKS))

def test_scenario_coverage_every_etf(tmp_path):
    sx = _build_etf(tmp_path)
    assert len(sx["scenarios"]) == 4
    for sc in sx["scenarios"]:
        assert set(SYMS) <= set(sc)
    assert sx["kernel"].scenarios is sx["scenarios"]
    assert sx["gw"].scenarios is sx["scenarios"]

def test_register_post_build_uses_config_authority(tmp_path):
    sx = build(str(tmp_path))
    rep = register_etf_universe(sx, SYMS, marks={**LEGACY_MARKS, **MARKS})
    cfg, ledger = sx["cfg"], sx["ledger"]
    assert rep["config_changed"] is True
    assert cfg.mandate.version == 2
    assert ledger.state.mandate_version == 2
    assert ledger.state.config_digest == cfg.mandate.digest()
    assert frozenset(SYMS) <= cfg.mandate.allowlist
    assert frozenset({"A", "B", "C"}) <= cfg.mandate.allowlist
    assert cfg.attested()

def test_register_fails_closed_on_missing_marks(tmp_path):
    sx = build(str(tmp_path))
    v0 = sx["cfg"].mandate.version
    with pytest.raises(ValueError):
        register_etf_universe(sx, SYMS)
    assert sx["cfg"].mandate.version == v0
    assert "SPY" not in sx["instruments"]

def test_register_idempotent(tmp_path):
    sx = build(str(tmp_path))
    m = {**LEGACY_MARKS, **MARKS}
    register_etf_universe(sx, SYMS, marks=m)
    rep2 = register_etf_universe(sx, SYMS, marks=m)
    assert rep2["config_changed"] is False
    assert sx["cfg"].mandate.version == 2

def test_risk_limits_preserved(tmp_path):
    sx = build(str(tmp_path))
    m0 = sx["cfg"].mandate
    floor0 = sx["cfg"].floor.digest()
    register_etf_universe(sx, SYMS, marks={**LEGACY_MARKS, **MARKS})
    m1 = sx["cfg"].mandate
    for f in ("gross_bps","issuer_bps","factor_bps","worst_loss_bps",
              "cvar_bps","drawdown_trigger_bps","order_notional_cap"):
        assert getattr(m0, f) == getattr(m1, f)
    assert m1.issuer_bps == 4_000
    assert sx["cfg"].floor.digest() == floor0

def test_unknown_issuer_shared_bucket_caps_book(tmp_path):
    sx = _build_etf(tmp_path)
    ledger, cfg = sx["ledger"], sx["cfg"]
    td = ledger.state.trade_date
    over = (WorkingOrder("O1","SPY","BUY",40,0,50_000,5,"RESERVED",td,"n1"),
            WorkingOrder("O2","QQQ","BUY",50,0,45_000,5,"RESERVED",td,"n2"))
    ev = evaluate(ledger.state, over, sx["instruments"], cfg.mandate, sx["scenarios"])
    assert not ev.ok and ev.reason == "LIMIT_ISSUER"
    ok = (WorkingOrder("O1","SPY","BUY",30,0,50_000,5,"RESERVED",td,"n1"),
          WorkingOrder("O2","QQQ","BUY",40,0,45_000,5,"RESERVED",td,"n2"))
    ev2 = evaluate(ledger.state, ok, sx["instruments"], cfg.mandate, sx["scenarios"])
    assert ev2.ok, ev2.reason

def test_authorize_accepts_build_time_etf(tmp_path):
    sx = _build_etf(tmp_path)
    tok = sx["kernel"].authorize(sx["ledger"], Intent("SPY","BUY",10,50_000,5),
                                 sx["flags"], sx["clock"], False)
    assert tok.mandate_version == 1

def test_authorize_rejects_unregistered_symbol(tmp_path):
    sx = _build_etf(tmp_path)
    with pytest.raises(Reject) as e:
        sx["kernel"].authorize(sx["ledger"], Intent("ZZZ","BUY",10,50_000,5),
                               sx["flags"], sx["clock"], False)
    assert e.value.code == "ALLOWLIST"

def test_authorize_after_post_build_register(tmp_path):
    sx = build(str(tmp_path))
    merged = {**LEGACY_MARKS, **MARKS}
    register_etf_universe(sx, SYMS, marks=merged)
    sx["ledger"].advance_session(101, 1_000_001, merged)
    sx["clock"].advance_wall(1)
    tok = sx["kernel"].authorize(sx["ledger"], Intent("SPY","BUY",10,50_000,5),
                                 sx["flags"], sx["clock"], False)
    assert tok.mandate_version == 2
    assert tok.config_digest == sx["cfg"].mandate.digest()

def test_cash_accounting_100k(tmp_path):
    sx = _build_etf(tmp_path)
    assert sx["ledger"].state.cash_settled == CASH_100K
    assert sx["ledger"].state.nav() == CASH_100K
    assert sx["ledger"].snapshot().nav == CASH_100K

def test_cold_start_no_authority_model_id(tmp_path):
    trackers = {"mom_21d_h1": ContributionTracker(half_life_obs=60.0, kappa=1.0,
                                                  probation_obs=30, horizon_days=1)}
    ms = MarketState(as_of=datetime(2024,1,2,tzinfo=timezone.utc).isoformat(),
                     avg_corr=0.3, vol_ann=0.15, liquidity_score=1.0,
                     data_age_secs=10, model_disagreement=0.0, n_assets=3,
                     drawdown_bps=0, drawdown_trigger_bps=1_000)
    fc = {"model_signals": {"mom_21d_h1": np.array([0.001,0.0,-0.001])},
          "cov": np.eye(3)*0.0002, "risk_aversion":2.0, "horizon_days":1}
    rec = run_cycle(ms, trackers, forecasts=fc, allocator=solve_allocation)
    assert rec.any_eligible_model is False
    assert rec.allocation is None
    assert "No model has earned positive authority" in rec.note
    assert all(w == 0.0 for w in rec.model_weights.values())

def test_model_signal_blend_path_when_earned(tmp_path):
    t = ContributionTracker(half_life_obs=60.0, kappa=1.0, probation_obs=30, horizon_days=1)
    for _ in range(300):
        t.update(0.02)
    assert t.eligible
    ms = MarketState(as_of=datetime(2024,1,2,tzinfo=timezone.utc).isoformat(),
                     avg_corr=0.3, vol_ann=0.15, liquidity_score=1.0,
                     data_age_secs=10, model_disagreement=0.0, n_assets=3,
                     drawdown_bps=0, drawdown_trigger_bps=1_000)
    fc = {"model_signals": {"m1": np.array([0.001,0.0,-0.001])},
          "cov": np.eye(3)*0.0002, "risk_aversion":2.0, "horizon_days":1}
    rec = run_cycle(ms, {"m1": t}, forecasts=fc, allocator=solve_allocation)
    assert rec.any_eligible_model is True
    assert rec.allocation is not None
    assert "blended" in rec.note

def test_legacy_build_unchanged(tmp_path):
    sx = build(str(tmp_path))
    assert set(sx["instruments"]) == {"A","B","C"}
    assert sx["cfg"].mandate.allowlist == frozenset({"A","B","C"})
    assert sx["ledger"].state.positions == {"A":300,"B":300}
    assert sx["ledger"].state.nav() == 10_000_000 + 300*10_000*2
    assert len(sx["scenarios"]) == 4
