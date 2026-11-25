"""End-to-end tests for the execution wiring (mapping + execution)."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.orchestrator.mapping import blend_signals
from quant_m1.orchestrator.execution import generate_intents, dispatch_intents
from quant_m1.allocate import solve_allocation
from quant_m1.system import build
from quant_m1.authority import Intent


def test_blend_signals_weights_sum_to_one():
    w = {"m1": 0.7, "m2": 0.3}
    s = {"m1": np.array([0.05, 0.02]), "m2": np.array([0.01, 0.08])}
    res = blend_signals(w, s)
    assert res[0] == pytest.approx(0.7*0.05 + 0.3*0.01)
    assert res[1] == pytest.approx(0.7*0.02 + 0.3*0.08)


def test_blend_signals_ignores_zero_weight_models():
    w = {"m1": 1.0, "m2": 0.0}
    s = {"m1": np.array([0.05]), "m2": np.array([999.0])}
    res = blend_signals(w, s)
    assert res[0] == pytest.approx(0.05)


def test_blend_signals_rejects_missing_signal():
    w = {"m1": 0.5, "m2": 0.5}
    s = {"m1": np.array([0.05])}
    with pytest.raises(KeyError):
        blend_signals(w, s)


def test_generate_intents_respects_integer_shares():
    sx = build("/tmp/test_gen_int")
    alloc = solve_allocation(np.array([0.05, 0.05]), np.eye(2)*0.01)
    prices = {"A": 10_000, "B": 10_000}
    intents = generate_intents(alloc, sx["ledger"], prices, sx["instruments"], min_trade_bps=0)
    assert all(isinstance(i.qty, int) for i in intents)
    assert all(i.qty > 0 for i in intents)


def test_generate_intents_skips_dust_trades():
    """When target weights are very close to current, no intents should be generated."""
    sx = build("/tmp/test_dust")
    # Default build: A=300, B=300 at 10_000 each, cash=10M -> NAV=16M
    # Current weights: A=3M/16M=0.1875, B=3M/16M=0.1875, C=0
    # Give the allocator nearly identical weights so delta is tiny
    current_w = np.array([0.1875, 0.1875, 0.0])
    alloc = solve_allocation(
        np.array([0.05, 0.05, 0.05]),
        np.eye(3) * 0.01,
        current_weights=current_w,
        band_frac=0.0,  # disable band so we can test dust filter independently
    )
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    # 500 bps of 16M NAV = 8M cents = $80k. Any trade below that is dust.
    intents = generate_intents(alloc, sx["ledger"], prices, sx["instruments"], min_trade_bps=500)
    assert len(intents) == 0


def test_dispatch_submits_to_mock_gateway(tmp_path):
    sx = build(str(tmp_path))
    intents = [Intent("C", "BUY", 5, 10_000, 10)]
    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"])
    assert report.intents_generated == 1
    assert report.intents_submitted == 1
    assert report.intents_rejected == 0
    assert len(sx["ledger"].state.working) == 1


def test_dispatch_handles_kernel_rejection_gracefully(tmp_path):
    sx = build(str(tmp_path), m_vertex=1)
    i1 = Intent("C", "BUY", 5, 10_000, 10)
    dispatch_intents([i1], sx["kernel"], sx["gw"], sx["ledger"], sx["flags"], sx["clock"])

    i2 = Intent("A", "BUY", 5, 10_000, 10)
    report = dispatch_intents([i2], sx["kernel"], sx["gw"], sx["ledger"], sx["flags"], sx["clock"])

    assert report.intents_generated == 1
    assert report.intents_submitted == 0
    assert report.intents_rejected == 1
    assert any("M_VERTEX" in r for r in report.rejection_reasons)


def test_end_to_end_paper_cycle(tmp_path):
    """Full loop: blend -> allocate -> generate -> dispatch.

    Uses a raised order_notional_cap because the default $20k cap is below
    the per-asset notional at ~equal weights on a $160k book. This test
    verifies wiring, not mandate calibration.
    """
    sx = build(str(tmp_path))
    # Raise order cap to allow ~$53k per-asset orders on a $160k book
    sx["cfg"].apply_config({"mandate": {"order_notional_cap": 100_000_000}})

    # 1. Blend
    w = {"m1": 1.0}
    s = {"m1": np.array([0.05, 0.03, 0.08])}
    mu = blend_signals(w, s)

    # 2. Allocate
    cov = np.eye(3) * 0.02
    alloc = solve_allocation(mu, cov)
    assert alloc.solver_ok

    # 3. Generate
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents = generate_intents(alloc, sx["ledger"], prices, sx["instruments"], min_trade_bps=10)
    assert len(intents) > 0

    # 4. Dispatch
    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"], sx["flags"], sx["clock"])
    assert report.intents_submitted > 0
    assert report.intents_rejected == 0