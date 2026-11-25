"""End-to-end tests for execution wiring.

Fixture arithmetic (derived, not guessed):
  build(cash=10_000_000, A=B=C=300 @ 10_000) -> NAV = 19_000_000
  issuer cap = floor_bps(NAV, 4000) = 7_600_000
  C has holdings=None -> UNK:ALL bucket, so C weight must stay <= 40%.
  mu=[0.04,0.04,0.06], cov=0.2*I -> w=[0.2,0.2,0.3]
  targets A=380 B=380 C=570; deltas +80/+80/+270, all BUY
  buy cost incl 10bp fees = 4_304_300 < 10_000_000 settled cash
  UNK:ALL = 5_700_000 < 7_600_000 ; ISS1 = 4_180_000 < 7_600_000
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
    sx["cfg"].apply_config({"mandate": {
        "order_notional_cap": 100_000_000,
        "participation_bps": 10_000,
    }})
    sx["ledger"].state.mandate_version = sx["cfg"].mandate.version
    sx["ledger"].state.config_digest = sx["cfg"].mandate.digest()
    sx["ledger"].persist()


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


def test_generate_intents_respects_integer_shares(tmp_path):
    sx = build(str(tmp_path))
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, _ = generate_intents(alloc, sx["ledger"], prices, sx["instruments"],
                                  min_trade_bps=0)
    assert len(intents) > 0
    assert all(isinstance(i.qty, int) and i.qty > 0 for i in intents)


def test_generate_intents_skips_dust_trades(tmp_path):
    sx = build(str(tmp_path), cash=0,
               positions={"A": 500, "B": 500, "C": 500},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    assert np.allclose(alloc.weights, [1/3, 1/3, 1/3], atol=1e-6)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=500)
    assert intents == []
    assert skipped == 0


def test_dust_filter_counts_skipped_trades(tmp_path):
    sx = build(str(tmp_path), cash=0,
               positions={"A": 500, "B": 500, "C": 498},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    alloc = solve_allocation(np.array([0.05, 0.05, 0.05]), np.eye(3) * 0.01)
    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=500)
    assert intents == []
    assert skipped >= 1


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


KNOWN_CODES = {
    "SETTLED_CASH_NEGATIVE", "ORDER_NOTIONAL", "PARTICIPATION", "M_VERTEX",
    "SHORT_PROHIBITED", "STALE_MARKS", "MODE_BLOCKS_NEW_RISK",
    "MODE_BLOCKS_SUBMIT", "CONFIG_STATE_MISMATCH", "WATCHDOG_STOP",
    "NAV_NONPOSITIVE", "DRAWDOWN_TRIGGER", "BAD_SIGNATURE", "NONCE_REPLAY",
}


def test_end_to_end_paper_cycle(tmp_path):
    """Wiring test. T+1 correctly blocks the concentrated BUY.

    Low covariance lets the allocator tilt hard into C. Sells produce T+1
    receivables, not settled cash, so the kernel refuses the large same-day
    BUY. That rejection is the safety property working. This test verifies
    the wiring transports intents and captures rejections without raising.
    """
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
    assert report.intents_generated == report.intents_submitted + report.intents_rejected
    for reason in report.rejection_reasons:
        code = reason.split(":", 1)[1]
        assert code in KNOWN_CODES or code.startswith("LIMIT_")


def test_unk_all_cap_blocks_concentrated_c(tmp_path):
    """C has holdings=None -> UNK:ALL. A 60% C weight breaches the 40% cap.

    cov=0.05*I with mu=[0.04,0.04,0.06] yields w=[0.2,0.2,0.6].
    UNK:ALL = 11_400_000 vs cap 7_600_000 -> LIMIT_ISSUER on C.
    This locks in P0-D as expected behaviour rather than a surprise failure.
    """
    sx = build(str(tmp_path), cash=10_000_000,
               positions={"A": 300, "B": 300, "C": 300},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    _relax_mandate(sx)

    alloc = solve_allocation(np.array([0.04, 0.04, 0.06]), np.eye(3) * 0.05)
    assert alloc.weights[2] > 0.5

    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=10)
    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"],
                              skipped_dust_trades=skipped)
    assert any("C:LIMIT_ISSUER" in r for r in report.rejection_reasons)


def test_end_to_end_balanced_cycle_fully_submits(tmp_path):
    """Every order submits. Fixture sized from the actual optimizer output.

    cov=0.2*I with mu=[0.04,0.04,0.06] -> w=[0.2,0.2,0.3].
    NAV 19_000_000, targets A=380 B=380 C=570, deltas +80/+80/+270 all BUY.
    Buy cost incl fees 4_304_300 < 10_000_000 settled cash.
    UNK:ALL 5_700_000 and ISS1 4_180_000, both under the 7_600_000 cap.
    """
    sx = build(str(tmp_path), cash=10_000_000,
               positions={"A": 300, "B": 300, "C": 300},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    _relax_mandate(sx)

    mu = blend_signals({"m1": 1.0}, {"m1": np.array([0.04, 0.04, 0.06])})
    alloc = solve_allocation(mu, np.eye(3) * 0.2)
    assert alloc.solver_ok
    assert np.allclose(alloc.weights, [0.2, 0.2, 0.3], atol=1e-3)

    prices = {"A": 10_000, "B": 10_000, "C": 10_000}
    intents, skipped = generate_intents(alloc, sx["ledger"], prices,
                                        sx["instruments"], min_trade_bps=10)
    assert len(intents) == 3
    assert all(i.side == "BUY" for i in intents)

    report = dispatch_intents(intents, sx["kernel"], sx["gw"], sx["ledger"],
                              sx["flags"], sx["clock"],
                              skipped_dust_trades=skipped)
    assert report.intents_rejected == 0, report.rejection_reasons
    assert report.intents_submitted == report.intents_generated == 3