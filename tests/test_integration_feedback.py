"""End-to-end integration tests for sleeve → probe → selector → allocator flow.

These tests verify that the isolated subsystems actually work together:
1. Sleeve positions are recorded in ProbeJournal
2. Selector weights influence allocator decisions
3. Complete feedback loop from signal → position → observation → authority
"""

import pytest
import numpy as np
from unittest.mock import Mock
from datetime import datetime

from quant_m1.integration.sleeve_probe import (
    record_sleeve_observation,
    process_sleeve_positions,
)
from quant_m1.integration.selector_executor import (
    apply_selector_weights,
    validate_selector_integration,
)


class TestSleeveProbeFeedback:
    """Test sleeve → ProbeJournal feedback loop."""

    def test_record_single_observation(self):
        """Verify a single sleeve observation is recorded correctly."""
        journal = Mock()
        journal.set_family.return_value = None
        journal.record.return_value = 'obs_123'

        result = record_sleeve_observation(
            journal=journal,
            signal_name='momentum_12_1',
            net_return=0.02,
            horizon_days=1,
            selector_weight=0.3,
            regime='bull',
        )

        # Verify journal was called
        assert journal.set_family.called
        assert journal.record.called
        assert result['recorded'] is True
        assert result['candidate'] == 'sleeve:momentum_12_1:bull'
        assert result['signal_name'] == 'momentum_12_1'
        assert result['net_return'] == 0.02
        assert result['regime'] == 'bull'

    def test_process_multiple_positions(self):
        """Verify batch processing of sleeve positions."""
        journal = Mock()
        journal.set_family.return_value = None
        journal.record.return_value = 'obs_id'

        sleeve_positions = {
            'momentum_12_1': {'position_usd': 1000.0, 'selector_weight': 0.3},
            'mean_reversion': {'position_usd': 500.0, 'selector_weight': 0.2},
            'carry': {'position_usd': 0.0, 'selector_weight': 0.1},  # Zero position
        }

        realized_returns = {
            'momentum_12_1': 0.02,
            'mean_reversion': -0.01,
            'carry': 0.015,
        }

        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=sleeve_positions,
            realized_returns=realized_returns,
            cost_bps=5.0,
            regime='bull',
        )

        # Should record 2 observations (carry has zero position)
        assert result['observations_recorded'] == 2
        assert result['total_position_usd'] == 1500.0
        assert result['regime'] == 'bull'

        # Verify portfolio return calculation
        # mom: 1000 * (0.02 - 0.0005) = 19.5
        # mr: 500 * (-0.01 - 0.0005) = -5.25
        # Total weighted return: 14.25
        # Portfolio return: 14.25 / 1500 = 0.0095
        expected_portfolio_return = 14.25 / 1500.0
        assert np.isclose(result['portfolio_net_return'], expected_portfolio_return, atol=1e-6)

    def test_zero_position_not_recorded(self):
        """Verify zero positions are not recorded as observations."""
        journal = Mock()
        journal.set_family.return_value = None
        journal.record.return_value = 'obs_id'

        sleeve_positions = {
            'signal_a': {'position_usd': 0.0, 'selector_weight': 0.5},
        }
        realized_returns = {'signal_a': 0.02}

        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=sleeve_positions,
            realized_returns=realized_returns,
        )

        # Should not call record for zero position
        assert not journal.record.called
        assert result['observations_recorded'] == 0


class TestSelectorAllocatorIntegration:
    """Test meta-selector → cost-aware allocator integration."""

    def test_apply_selector_weights_basic(self):
        """Verify selector weights are distributed to signals correctly."""
        selector_weights = {
            'momentum': 0.6,
            'reversal': 0.4,
        }

        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5'],
        }

        result = apply_selector_weights(selector_weights, signal_families)

        # Momentum: 0.6 split equally between 2 signals = 0.3 each
        assert np.isclose(result['mom_12_1'], 0.3)
        assert np.isclose(result['mom_6_1'], 0.3)

        # Reversal: 0.4 to single signal
        assert np.isclose(result['rev_5'], 0.4)

        # All weights should sum to 1.0
        assert np.isclose(sum(result.values()), 1.0)

    def test_apply_selector_weights_normalization(self):
        """Verify weights are normalized if they don't sum to 1.0."""
        selector_weights = {
            'momentum': 2.0,
            'reversal': 1.0,
        }

        signal_families = {
            'momentum': ['mom_12_1'],
            'reversal': ['rev_5'],
        }

        result = apply_selector_weights(selector_weights, signal_families)

        # Should normalize: 2.0/(2.0+1.0) = 0.667, 1.0/(2.0+1.0) = 0.333
        assert np.isclose(result['mom_12_1'], 2.0/3.0)
        assert np.isclose(result['rev_5'], 1.0/3.0)
        assert np.isclose(sum(result.values()), 1.0)

    def test_zero_weight_family(self):
        """Verify zero-weight families get zero allocation."""
        selector_weights = {
            'momentum': 1.0,
            'reversal': 0.0,
        }

        signal_families = {
            'momentum': ['mom_12_1'],
            'reversal': ['rev_5', 'rev_10'],
        }

        result = apply_selector_weights(selector_weights, signal_families)

        assert np.isclose(result['mom_12_1'], 1.0)
        assert result['rev_5'] == 0.0
        assert result['rev_10'] == 0.0

    def test_validate_selector_integration_passes(self):
        """Test validation of selector → allocator integration."""
        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5']
        }
        allocator_state = {
            'allocations': {
                'mom_12_1': 5000.0,
                'mom_6_1': 2000.0,
                'rev_5': 3000.0,
            }
        }

        result = validate_selector_integration(
            selector_weights, signal_families, allocator_state
        )

        # Should pass all checks
        assert result['status'] == 'PASSED'
        assert len(result['issues']) == 0
        assert np.isclose(result['total_weight'], 1.0)

    def test_integration_fails_on_negative_weight(self):
        """Verify integration validation catches negative weights."""
        selector_weights = {'momentum': 1.2, 'reversal': -0.2}
        signal_families = {
            'momentum': ['mom_12_1'],
            'reversal': ['rev_5']
        }
        allocator_state = {'allocations': {}}

        result = validate_selector_integration(
            selector_weights, signal_families, allocator_state
        )

        # Should fail due to negative weight
        assert result['status'] == 'FAILED'
        assert any('negative weight' in issue.lower() for issue in result['issues'])


class TestEndToEndIntegration:
    """End-to-end tests combining sleeve, probe, selector, and allocator."""

    def test_full_feedback_loop(self):
        """Test complete flow: selector → sleeve → observation → probe."""
        journal = Mock()
        journal.set_family.return_value = None
        journal.record.return_value = 'obs_id'

        # Mock selector output
        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5']
        }

        # Step 1: Selector produces weights
        effective_weights = apply_selector_weights(
            selector_weights, signal_families
        )

        # Step 2: Sleeve takes positions based on weights
        sleeve_positions = {
            signal: {'position_usd': weight * 10000, 'selector_weight': weight}
            for signal, weight in effective_weights.items()
        }

        # Step 3: Positions realize returns
        realized_returns = {
            'mom_12_1': 0.02,
            'mom_6_1': 0.015,
            'rev_5': -0.01,
        }

        # Step 4: Record observations in ProbeJournal
        result = process_sleeve_positions(
            journal=journal,
            sleeve_positions=sleeve_positions,
            realized_returns=realized_returns,
            regime='bull',
        )

        # Verify integration worked
        assert result['observations_recorded'] == 3
        assert journal.set_family.call_count == 3
        assert journal.record.call_count == 3

        # Verify all observations were tagged as sleeve_exploration
        for call in journal.set_family.call_args_list:
            family = call[0][1]  # Second positional arg
            assert family == 'sleeve_exploration'
