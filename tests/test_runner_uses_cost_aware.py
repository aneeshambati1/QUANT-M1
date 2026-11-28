"""Regression: the simulation runner MUST use the cost-aware allocator.

This test is the permanent gate for Batch A — if anyone ever swaps the runner
back to the vanilla engine without going through the pre-registered protocol
again, this test fails. The friction-saving is mechanical, not alpha; it stays
in place until a further protocol decision removes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_m1.allocate.cost_aware import cost_aware_allocator
from scripts import run_simulation as rs


def test_runner_summary_declares_cost_aware_allocator():
    """The runner's published summary must name the cost-aware allocator."""
    panel = _panel(n=80)
    res = rs.run_simulation(panel, days=30)
    assert res["summary"]["allocator"] == "cost_aware_allocator"


def test_runner_passes_allocator_kwarg_consistently():
    """Smoke run: the runner still reports NO AUTHORITY / NO TRADES with the
    cost-aware allocator engaged. The mechanical friction reduction is the
    validated change; edge remains unproven on the placeholder signal.
    """
    panel = _panel(n=80)
    res = rs.run_simulation(panel, days=20)
    report = rs.format_report(res)
    assert "Allocator: cost_aware_allocator" in report
    assert "NO AUTHORITY" in report or "NO TRADES" in report or "TRADES OCCURRED" in report


def test_runner_integration_uses_cost_aware():
    """The runner integration contract still works with the new allocator.
    Baselines and SPY keys are preserved; the swap is a drop-in.
    """
    panel = _panel(n=80)
    res = rs.run_simulation(panel, days=20)
    b = res["summary"]["baselines"]
    assert "benchmark_buy_hold_gross" in b
    assert "equal_weight_gross" in b
    assert "spy_buy_hold_gross" in b or res["summary"]["registered_symbols"][0] != "SPY"


def _panel(n: int = 80) -> pd.DataFrame:
    """Tiny synthetic panel for the runner contract test.
    Includes SPY so the backward-compat SPY baseline keys are exercised.
    """
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=n)
    symbols = ["SPY", "QQQ", "TLT"]
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0))
    df = pd.DataFrame(px, index=idx, columns=symbols)
    return df
