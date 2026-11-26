"""PCDL adversarial accounting tests. Measurement only; no edge claims."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quant_m1.allocate import solve_allocation
from quant_m1.authority import Intent
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.research.paired_decision import (
    PairedLedger, PairedSpec, capture_start, clean_start, hold_current,
    liquidate_to_cash, run_episode, runtime_manifest, verify_reproducibility,
)
from quant_m1.system import build

M0 = {"A": 10_000, "B": 10_000, "C": 10_000}
UP = {"A": 10_500, "B": 10_400, "C": 10_200}
DN = {"A": 9_500, "B": 9_600, "C": 9_800}


def _state():
    return MarketState(as_of="2026-11-26T15:00:00+00:00", avg_corr=0.4, vol_ann=0.15,
                       liquidity_score=1.0, data_age_secs=10, model_disagreement=0.1,
                       n_assets=3)


def _active():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200):
        t.update(0.01)
    return t


def _fc():
    return {"expected_returns": np.array([0.06, 0.04, 0.02]),
            "cov": np.eye(3) * 0.04, "horizon_days": 1, "risk_aversion": 4.0}


def _util(book):
    return math.log(book.end_nav / book.start_nav)


def _spec(comparators):
    return PairedSpec(utility=_util, utility_id="net_log_growth_v1",
                      comparators=comparators, horizon_id="one_session_v1",
                      block_rule_id="contiguous_blocks_v1",
                      manifest=runtime_manifest())


def _ep(sx, spec, date, wall, marks, trackers=None, eid="e1", valuation=None):
    return run_episode(sx, _state(), {"m1": _active()} if trackers is None else trackers,
                       _fc(), new_trade_date=date, new_mark_wall=wall,
                       new_marks=marks, allocator=solve_allocation, spec=spec,
                       episode_id=eid, valuation=valuation)


def test_identical_policies_produce_zero_differential(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    spec = _spec({"mirror": lambda s, t, f: (t, f)})
    rec = _ep(sx, spec, 101, 1_000_100, M0)
    assert rec.valid
    assert rec.differentials["mirror"] == pytest.approx(0.0, abs=1e-12)


def test_flat_marks_hold_beats_active_costs(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_100, M0)
    assert rec.valid and rec.books["active"].turnover_shares > 0
    assert rec.differentials["hold"] < 0          # trading only paid costs


def test_favourable_move_active_beats_hold(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_100, M0,
              valuation=(102, 1_000_200, UP))
    assert rec.valid and rec.differentials["hold"] > 0


def test_adverse_move_hold_beats_active(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_100, M0,
              valuation=(102, 1_000_200, DN))
    assert rec.valid and rec.differentials["hold"] < 0


def test_cash_liquidation_pays_costs_and_settles(tmp_path):
    sx = build(str(tmp_path), cash=1_000_000, positions={"A": 100}, marks=M0)
    spec = _spec({"hold": hold_current, "cash": liquidate_to_cash})
    rec = _ep(sx, spec, 101, 1_000_100, M0)
    assert rec.valid
    cash_book = rec.books["cash"]
    assert cash_book.turnover_shares == 100
    assert cash_book.filled_cost > 0
    assert cash_book.receivables                      # T+1, not settled cash
    assert cash_book.cash_settled == rec.start["cash_settled"]
    assert rec.books["hold"].turnover_shares == 0     # hold is NOT liquidation


def test_hold_and_cash_are_distinct_comparators(tmp_path):
    sx = build(str(tmp_path), cash=1_000_000, positions={"A": 100}, marks=M0)
    spec = _spec({"hold": hold_current, "cash": liquidate_to_cash})
    rec = _ep(sx, spec, 101, 1_000_100, M0)
    assert rec.books["hold"].turnover_shares == 0
    assert rec.books["cash"].turnover_shares == 100
    assert rec.books["hold"].economic_digest() != rec.books["cash"].economic_digest()


def test_comparator_failure_invalidates_episode(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)

    def broken(s, t, f):
        raise RuntimeError("comparator exploded")

    rec = _ep(sx, _spec({"broken": broken}), 101, 1_000_100, M0)
    assert not rec.valid and "COMPARATOR_FAILED" in rec.invalid_reason
    assert rec.differentials == {}


def test_active_block_invalidates_episode(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    sx["gw"].latch_divergence("FAULT")
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_100, M0)
    assert not rec.valid and rec.invalid_reason == "ACTIVE_BOOK_BLOCKED"


def test_comparator_cannot_mutate_active_book(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    rec = _ep(sx, _spec({"hold": hold_current, "cash": liquidate_to_cash}),
              101, 1_000_100, M0)
    led = sx["ledger"]
    assert led.state.nav() == rec.books["active"].end_nav
    assert tuple(sorted((k, v) for k, v in led.state.positions.items() if v)) == \
        rec.books["active"].positions


def test_replay_reproduces_economic_outputs(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    spec = _spec({"hold": hold_current})
    trackers = {"m1": _active()}
    rec = run_episode(sx, _state(), trackers, _fc(), new_trade_date=101,
                      new_mark_wall=1_000_100, new_marks=M0,
                      allocator=solve_allocation, spec=spec, episode_id="e1")
    ok, mismatches = verify_reproducibility(
        rec, sx, _state(), trackers, _fc(), new_trade_date=101,
        new_mark_wall=1_000_100, new_marks=M0, allocator=solve_allocation, spec=spec)
    assert ok, mismatches


def test_unclean_start_is_invalid_and_retained(tmp_path):
    d = str(tmp_path)
    sx = build(d, cash=2_000_000, positions={}, marks=M0)
    sx["clock"].advance_wall(100)
    i = Intent("C", "BUY", 10, 10_000, 10)
    tok = sx["kernel"].authorize(sx["ledger"], i, sx["flags"], sx["clock"], False)
    sx["gw"].submit(tok, i)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 3, price=10_000))
    ok, why = clean_start(sx)
    assert not ok and why == "ACTIVE_WORKING_ORDERS_AT_START"
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_200, M0)
    assert not rec.valid and rec.invalid_reason == "ACTIVE_WORKING_ORDERS_AT_START"
    led = PairedLedger(d + "/pcdl.jsonl")
    led.append(rec)
    assert len(led.records()) == 1
    assert led.records()[0]["valid"] is False
    assert led.verify_chain()


def test_spec_requires_registration_fields():
    with pytest.raises(ValueError):
        PairedSpec(utility=_util, utility_id="", comparators={"h": hold_current},
                   horizon_id="h", block_rule_id="b", manifest=runtime_manifest())
    with pytest.raises(ValueError):
        PairedSpec(utility=_util, utility_id="u", comparators={},
                   horizon_id="h", block_rule_id="b", manifest=runtime_manifest())


def test_spec_digest_changes_with_comparator_set():
    a = _spec({"hold": hold_current}).digest()
    b = _spec({"hold": hold_current, "cash": liquidate_to_cash}).digest()
    assert a != b


def test_replay_reproduces_valuation_episode(tmp_path):
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    spec = _spec({"hold": hold_current})
    trackers = {"m1": _active()}
    val = (102, 1_000_200, UP)
    rec = run_episode(sx, _state(), trackers, _fc(), new_trade_date=101,
                      new_mark_wall=1_000_100, new_marks=M0,
                      allocator=solve_allocation, spec=spec, episode_id="v1",
                      valuation=val)
    ok, mism = verify_reproducibility(
        rec, sx, _state(), trackers, _fc(), new_trade_date=101,
        new_mark_wall=1_000_100, new_marks=M0, allocator=solve_allocation,
        spec=spec, valuation=val)
    assert ok, mism


def test_decision_only_episode_is_cost_dominated(tmp_path):
    """Without valuation the differential cannot show directional value."""
    sx = build(str(tmp_path), cash=2_000_000, positions={}, marks=M0)
    rec = _ep(sx, _spec({"hold": hold_current}), 101, 1_000_100, M0)
    assert rec.valid and rec.differentials["hold"] < 0
