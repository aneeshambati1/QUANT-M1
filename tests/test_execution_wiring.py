"""End-to-end tests for the execution wiring (mapping + execution).

Mandate arithmetic used here, derived from system.py defaults:
  adv = 50_000_000 cents, participation_bps = 500
  -> max order notional = 2_500_000 cents ($25,000)
  order_notional_cap default = 2_000_000 cents ($20,000)
Both constraints bind on a $160k book at ~equal thirds. Tests that verify
wiring rather than mandate calibration raise BOTH via apply_config and then
propagate_config to keep the ledger binding consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.orchestrator.mapping import blend_signals
from quant_m1.orchestrator.execution import generate_intents, dispatch_intents
from quant_m1.allocate import solve_allocation
from quant_m1.system import build
from quant_m1.authority import Intent


def _relax_mandate(sx):
    """Raise both notional caps and sync the ledger config binding."""
    sx["cfg"].apply_config({"mandate": {
        "order_notional_cap": 100_000_000,
        "participation_bps": 10_000,
    }})
    sx["ledger"].state.mandate_version = sx["cfg"].mandate.version
    sx["ledger"].state.config_digest = sx["cfg"].mandate.digest()
    sx["ledger"].persist()


# ---- mapping ----

def test_blend_signals_weights_sum_to_one():
    w = {"m1": 0.7, "m2": 0.3}
    s = {"m1": np.array([0.05, 0.02]), "m2": np.array([0.01, 0.08])}
    res = blend_signals(w, s)
    assert res[0] == pytest.approx(0.7 * 0.05 + 0.3 * 0.01)
    assert res[1] == pytest.approx(0.7 * 0.02 + 0.3 * 0.08)


def test_blend_signals_ignores_zero_weight_models():
    w = {"m1": 1.0, "m2": 0.0}
    s = {"m1": np.array([0.05]), "m2": np.array([999.0])}
    assert blend_signals(w, s)[0] == pytest.approx(0.05)


def test_blend_signals_rejects_missing_signal():
    with pytest.raises(KeyError):
        blend_signals({"m1": 0.5, "m2": 0.5}, {"m1": np.array([0.05])})


def test_blend_signals_rejects_all_zero_weights():
    with pytest.raises(ValueError):
        blend_signals({"m1": 0.0}, {"m1": np.array([0.05])})


# ---- intent generation ----

def test_generate_intents_respects_integer_shares(tmp_path):
    sx = build(str(tmp_path))
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, _ = generate_intents(alloc, sx["ledger"], prices, sx["instruments"],
                                  min_trade_bps=0)
    assert len(intents) > 0
    assert all(isinstance(i.qty, int) and i.qty > 0 for i in intents)


def test_generate_intents_skips_dust_trades(tmp_path):
    """Ledger already holds equal thirds, so target equals actual -> no trades.

    NAV = 15_000_000 cents. Equal thirds = 5_000_000 each = 500 shares at 10_000.
    Building the book that way makes every delta zero.
    """
    sx = build(str(tmp_path), cash=0,
               positions={"A": 500, "B": 500, "C": 500},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    assert np.allclose(alloc.weights, [1/3, 1/3, 1/3], atol=1e-6)

    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=500)
    assert intents == []
    assert skipped == 0  # deltas are exactly zero, not dust-filtered


def test_dust_filter_counts_skipped_trades(tmp_path):
    """A small but nonzero delta below the threshold is counted as skipped."""
    sx = build(str(tmp_path), cash=0,
               positions={"A": 500, "B": 500, "C": 498},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=500)
    assert intents == []
    assert skipped >= 1


# ---- dispatch ----

def test_dispatch_submits_to_mock_gateway(tmp_path):
    sx = build(str(tmp_path))
    report = dispatch_intents([Intent("C", "BUY", 5, 10_000, 10)],
                              sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"])
    assert report.intents_submitted == 1
    assert report.intents_rejected == 0
    assert len(sx["ledger"].state.working) == 1


def test_dispatch_handles_kernel_rejection_gracefully(tmp_path):
    sx = build(str(tmp_path), m_vertex=1)
    dispatch_intents([Intent("C", "BUY", 5, 10_000, 10)],
                     sx["kernel"], sx["gw"], sx["ledger"], sx["flags"], sx["clock"])
    report = dispatch_intents([Intent("A", "BUY", 5, 10_000, 10)],
                              sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"])
    assert report.intents_submitted == 0
    assert report.intents_rejected == 1
    assert any("M_VERTEX" in r for r in report.rejection_reasons)


def test_default_mandate_blocks_large_orders(tmp_path):
    """Documents the real constraint: default caps reject ~$53k orders."""
    sx = build(str(tmp_path))
    alloc = solve_allocation(np.array([0.05, 0.03, 0.08]), np.eye(3) * 0.02)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, _ = generate_intents(alloc, sx["ledger"], prices, sx["instruments"],
                                  min_trade_bps=10)
    assert len(intents) > 0
    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"])
    assert report.intents_submitted == 0
    assert report.intents_rejected == len(intents)


# ---- end to end ----

def test_end_to_end_paper_cycle(tmp_path):
    """blend -> allocate -> generate -> dispatch, with mandate relaxed."""
    sx = build(str(tmp_path))
    _relax_mandate(sx)

    mu = blend_signals({"m1": 1.0}, {"m1": np.array([0.05, 0.03, 0.08])})
    alloc = solve_allocation(mu, np.eye(3) * 0.02)
    assert alloc.solver_ok

    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=10)
    assert len(intents) > 0

    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"],
                              skipped_dust_trades=skipped)
    assert report.intents_submitted > 0
    assert report.intents_rejected == 0