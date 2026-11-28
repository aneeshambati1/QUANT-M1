"""Tests for sleeve -> real ProbeJournal integration."""

import numpy as np
import pytest

from quant_m1.meta.probe import ProbeJournal
from quant_m1.integration.sleeve_probe import (
    record_sleeve_observation,
    process_sleeve_positions,
)
from quant_m1.integration.selector_executor import (
    apply_selector_weights,
    validate_selector_integration,
)


def make_journal(tmp_path):
    path = tmp_path / "probe_journal.json"
    return ProbeJournal(str(path)), path


class TestSleeveProbeFeedback:

    def test_record_single_observation_persists(self, tmp_path):
        journal, path = make_journal(tmp_path)

        result = record_sleeve_observation(
            journal=journal,
            signal_name="momentum_12_1",
            net_return=0.02,
            horizon_days=1,
            selector_weight=0.3,
            regime="bull",
        )

        candidate = "sleeve:momentum_12_1:bull"

        assert result["recorded"] is True
        assert result["candidate"] == candidate
        assert result["net_return"] == pytest.approx(0.02)

        assert journal.families[candidate] == "sleeve_exploration"
        assert journal.n_obs(candidate, 1) == 1
        assert journal.observations[(candidate, 1)] == [0.02]

        # Verify persistence and hash-chain replay.
        reopened = ProbeJournal(str(path))
        assert reopened.n_obs(candidate, 1) == 1
        assert reopened.observations[(candidate, 1)] == [0.02]

    def test_process_multiple_positions(self, tmp_path):
        journal, _ = make_journal(tmp_path)

        positions = {
            "momentum_12_1": {
                "position_usd": 1000.0,
                "selector_weight": 0.3,
            },
            "mean_reversion": {
                "position_usd": 500.0,
                "selector_weight": 0.2,
            },
            "carry": {
                "position_usd": 0.0,
                "selector_weight": 0.1,
            },
        }

        returns = {
            "momentum_12_1": 0.02,
            "mean_reversion": -0.01,
            "carry": 0.015,
        }

        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=positions,
            realized_returns=returns,
            cost_bps=5.0,
            regime="bull",
            horizon_days=1,
        )

        assert result["observations_recorded"] == 2
        assert result["total_position_usd"] == 1500.0

        expected = (
            1000.0 * (0.02 - 0.0005)
            + 500.0 * (-0.01 - 0.0005)
        ) / 1500.0

        assert result["portfolio_net_return"] == pytest.approx(expected)

        assert journal.n_obs("sleeve:momentum_12_1:bull", 1) == 1
        assert journal.n_obs("sleeve:mean_reversion:bull", 1) == 1
        assert journal.n_obs("sleeve:carry:bull", 1) == 0

    def test_zero_position_is_not_recorded(self, tmp_path):
        journal, _ = make_journal(tmp_path)

        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions={
                "signal_a": {
                    "position_usd": 0.0,
                    "selector_weight": 0.5,
                }
            },
            realized_returns={},
        )

        assert result["observations_recorded"] == 0
        assert journal.chain == []
        assert journal.observations == {}

    def test_missing_realized_return_fails_closed(self, tmp_path):
        journal, _ = make_journal(tmp_path)

        positions = {
            "signal_a": {
                "position_usd": 1000.0,
                "selector_weight": 0.5,
            },
            "signal_b": {
                "position_usd": 500.0,
                "selector_weight": 0.2,
            },
        }

        # signal_b has no completed return.
        with pytest.raises(ValueError, match="missing realized return"):
            process_sleeve_positions(
                journal=journal,
                sleeve_positions=positions,
                realized_returns={"signal_a": 0.02},
            )

        # Validation occurs before journal writes.
        assert journal.chain == []
        assert journal.observations == {}

    def test_invalid_family_registration_is_not_suppressed(self):
        class BrokenJournal:
            def set_family(self, candidate, family):
                raise ValueError("family registration failed")

            def record(self, candidate, horizon, value):
                raise AssertionError("record must not be called")

        with pytest.raises(ValueError, match="family registration failed"):
            record_sleeve_observation(
                BrokenJournal(),
                "signal_a",
                0.01,
            )


class TestSelectorWeights:

    def test_apply_selector_weights_basic(self):
        result = apply_selector_weights(
            {"momentum": 0.6, "reversal": 0.4},
            {
                "momentum": ["mom_12_1", "mom_6_1"],
                "reversal": ["rev_5"],
            },
        )

        assert result["mom_12_1"] == pytest.approx(0.3)
        assert result["mom_6_1"] == pytest.approx(0.3)
        assert result["rev_5"] == pytest.approx(0.4)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_apply_selector_weights_normalizes(self):
        result = apply_selector_weights(
            {"momentum": 2.0, "reversal": 1.0},
            {
                "momentum": ["mom_12_1"],
                "reversal": ["rev_5"],
            },
        )

        assert result["mom_12_1"] == pytest.approx(2.0 / 3.0)
        assert result["rev_5"] == pytest.approx(1.0 / 3.0)

    def test_zero_weight_family(self):
        result = apply_selector_weights(
            {"momentum": 1.0, "reversal": 0.0},
            {
                "momentum": ["mom_12_1"],
                "reversal": ["rev_5", "rev_10"],
            },
        )

        assert result["mom_12_1"] == pytest.approx(1.0)
        assert result["rev_5"] == 0.0
        assert result["rev_10"] == 0.0

    def test_validate_selector_integration(self):
        result = validate_selector_integration(
            {"momentum": 0.7, "reversal": 0.3},
            {
                "momentum": ["mom_12_1", "mom_6_1"],
                "reversal": ["rev_5"],
            },
            {
                "allocations": {
                    "mom_12_1": 5000.0,
                    "mom_6_1": 2000.0,
                    "rev_5": 3000.0,
                }
            },
        )

        assert result["status"] == "PASSED"
        assert result["issues"] == []

    def test_negative_weight_is_rejected_by_validator(self):
        result = validate_selector_integration(
            {"momentum": 1.2, "reversal": -0.2},
            {
                "momentum": ["mom_12_1"],
                "reversal": ["rev_5"],
            },
            {"allocations": {}},
        )

        assert result["status"] == "FAILED"
        assert any(
            "negative weight" in issue.lower()
            for issue in result["issues"]
        )


class TestSleeveSelectorJournalFlow:

    def test_selector_to_sleeve_to_real_journal(self, tmp_path):
        journal, _ = make_journal(tmp_path)

        family_weights = {
            "momentum": 0.7,
            "reversal": 0.3,
        }

        signal_families = {
            "momentum": ["mom_12_1", "mom_6_1"],
            "reversal": ["rev_5"],
        }

        signal_weights = apply_selector_weights(
            family_weights,
            signal_families,
        )

        positions = {
            signal: {
                "position_usd": weight * 10000.0,
                "selector_weight": weight,
            }
            for signal, weight in signal_weights.items()
        }

        realized_returns = {
            "mom_12_1": 0.02,
            "mom_6_1": 0.015,
            "rev_5": -0.01,
        }

        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=positions,
            realized_returns=realized_returns,
            regime="bull",
            horizon_days=1,
        )

        assert result["observations_recorded"] == 3

        for signal in positions:
            candidate = f"sleeve:{signal}:bull"
            assert journal.n_obs(candidate, 1) == 1
            assert journal.families[candidate] == "sleeve_exploration"
