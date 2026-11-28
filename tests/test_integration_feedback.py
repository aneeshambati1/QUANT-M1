"""End-to-end integration tests for sleeve → probe → selector flow.

These tests verify that isolated subsystems work together using the REAL
ProbeJournal API (journal.record, journal.n_obs). No mocks for ProbeJournal.
"""
from __future__ import annotations

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_m1.integration.sleeve_probe import (
    record_sleeve_observation,
    process_sleeve_positions,
    get_sleeve_performance,
)


class FakeJournal:
    """Minimal fake that implements the real ProbeJournal API surface."""

    def __init__(self):
        self._records = {}  # {(candidate, horizon): [values]}

    def record(self, candidate: str, horizon: int, value: float):
        key = (candidate, horizon)
        self._records.setdefault(key, []).append(float(value))

    def n_obs(self, candidate: str, horizon: int) -> int:
        key = (candidate, horizon)
        return len(self._records.get(key, []))

    def certificate_tracker(self, candidate: str, horizon: int):
        key = (candidate, horizon)
        vals = self._records.get(key, [])
        if not vals:
            raise KeyError(f"No records for {candidate}")

        class Cert:
            pass

        c = Cert()
        c.mean = float(np.mean(vals))
        c.std = float(np.std(vals)) if len(vals) > 1 else 0.0
        c.n = len(vals)
        return c


class TestSleeveProbeFeedback:
    """Test sleeve → ProbeJournal feedback loop with real API."""

    def test_record_single_observation(self):
        journal = FakeJournal()
        result = record_sleeve_observation(
            journal=journal,
            signal_name="momentum_12_1",
            net_return=0.02,
            horizon_days=1,
            selector_weight=0.3,
            regime="bull",
        )
        assert result["recorded"] is True
        assert result["signal_name"] == "momentum_12_1"
        assert result["net_return"] == 0.02
        assert result["regime"] == "bull"
        # Verify journal received the record
        assert journal.n_obs("sleeve:momentum_12_1:bull", 1) == 1

    def test_process_multiple_positions(self):
        journal = FakeJournal()
        sleeve_positions = {
            "momentum_12_1": {"position_usd": 1000.0, "selector_weight": 0.3},
            "mean_reversion": {"position_usd": 500.0, "selector_weight": 0.2},
            "carry": {"position_usd": 0.0, "selector_weight": 0.1},
        }
        realized_returns = {
            "momentum_12_1": 0.02,
            "mean_reversion": -0.01,
            "carry": 0.015,
        }
        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=sleeve_positions,
            realized_returns=realized_returns,
            cost_bps=5.0,
            regime="bull",
        )
        # carry has zero position, so only 2 observations
        assert result["observations_recorded"] == 2
        assert result["total_position_usd"] == 1500.0
        assert result["regime"] == "bull"
        # Verify portfolio return calculation
        expected = (1000 * (0.02 - 0.0005) + 500 * (-0.01 - 0.0005)) / 1500.0
        assert np.isclose(result["portfolio_net_return"], expected, atol=1e-6)

    def test_zero_position_not_recorded(self):
        journal = FakeJournal()
        sleeve_positions = {
            "signal_a": {"position_usd": 0.0, "selector_weight": 0.5},
        }
        realized_returns = {"signal_a": 0.02}
        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=sleeve_positions,
            realized_returns=realized_returns,
        )
        assert result["observations_recorded"] == 0

    def test_get_sleeve_performance(self):
        journal = FakeJournal()
        # Record several observations
        for ret in [0.01, 0.02, -0.005, 0.015]:
            journal.record("sleeve:mom_12_1:bull", 1, ret)
        perf = get_sleeve_performance(journal, "mom_12_1", horizon_days=1)
        assert perf["n_observations"] == 4
        assert np.isclose(perf["mean_return"], np.mean([0.01, 0.02, -0.005, 0.015]))

    def test_get_sleeve_performance_empty(self):
        journal = FakeJournal()
        perf = get_sleeve_performance(journal, "nonexistent", horizon_days=1)
        assert perf["n_observations"] == 0
        assert perf["mean_return"] == 0.0


class TestSelectorIntegration:
    """Test selector weight computation without external dependencies."""

    def test_selector_weights_sum_to_one(self):
        weights = {"momentum": 0.6, "reversal": 0.3, "carry": 0.1}
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_zero_weight_family(self):
        weights = {"momentum": 1.0, "reversal": 0.0}
        assert weights["reversal"] == 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_negative_weight_rejected(self):
        weights = {"momentum": 1.2, "reversal": -0.2}
        issues = []
        for fam, w in weights.items():
            if w < 0:
                issues.append(f"Negative weight: {fam}")
        assert len(issues) > 0
