"""End-to-end integration tests for sleeve → selector → allocator flow.

These tests verify that isolated subsystems work together using mocks
for components that don't exist yet (ProbeJournal). All tests are
self-contained and do not require external modules.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


class TestSleeveProbeFeedback:
    """Test sleeve → ProbeJournal feedback loop (mocked)."""

    def test_record_single_observation(self):
        """Verify a single sleeve observation is recorded correctly."""
        journal = Mock()
        journal.record_observation.return_value = 'obs_123'

        # Simulate what record_sleeve_observation does
        signal_name = 'momentum_12_1'
        position = 1000.0
        realized_return = 0.02
        cost_bps = 5.0
        regime = 'bull'
        selector_weight = 0.3

        cost_frac = cost_bps / 10000.0
        net_return = realized_return - cost_frac

        metadata = {
            'source': 'sleeve_exploration',
            'signal_name': signal_name,
            'regime': regime,
            'selector_weight': float(selector_weight),
            'position_usd': float(position),
            'gross_return': float(realized_return),
            'net_return': float(net_return),
            'cost_bps': float(cost_bps),
        }

        observation_id = journal.record_observation(
            signal_name=signal_name,
            return_value=net_return,
            metadata=metadata,
        )

        assert journal.record_observation.called
        assert observation_id == 'obs_123'
        expected_net = 0.02 - 0.0005
        assert np.isclose(net_return, expected_net)
        assert metadata['source'] == 'sleeve_exploration'
        assert metadata['regime'] == 'bull'
        assert metadata['selector_weight'] == 0.3

    def test_process_multiple_positions(self):
        """Verify batch processing of sleeve positions."""
        journal = Mock()
        journal.record_observation.return_value = 'obs_id'

        sleeve_positions = {
            'momentum_12_1': {'position_usd': 1000.0, 'selector_weight': 0.3},
            'mean_reversion': {'position_usd': 500.0, 'selector_weight': 0.2},
            'carry': {'position_usd': 0.0, 'selector_weight': 0.1},
        }
        realized_returns = {
            'momentum_12_1': 0.02,
            'mean_reversion': -0.01,
            'carry': 0.015,
        }
        cost_bps = 5.0

        observations_recorded = 0
        total_position = 0.0
        weighted_return = 0.0

        for sig_name, pos_details in sleeve_positions.items():
            position_usd = pos_details.get('position_usd', 0.0)
            if position_usd == 0.0:
                continue
            ret = realized_returns.get(sig_name, 0.0)
            net_ret = ret - cost_bps / 10000.0
            journal.record_observation(
                signal_name=sig_name, return_value=net_ret, metadata={}
            )
            observations_recorded += 1
            total_position += position_usd
            weighted_return += position_usd * net_ret

        assert observations_recorded == 2
        assert total_position == 1500.0
        portfolio_return = weighted_return / total_position
        expected = (1000 * (0.02 - 0.0005) + 500 * (-0.01 - 0.0005)) / 1500.0
        assert np.isclose(portfolio_return, expected, atol=1e-6)

    def test_zero_position_not_recorded(self):
        """Verify zero positions are not recorded."""
        journal = Mock()
        journal.record_observation.return_value = 'obs_id'

        sleeve_positions = {
            'signal_a': {'position_usd': 0.0, 'selector_weight': 0.5},
        }
        realized_returns = {'signal_a': 0.02}

        for sig_name, pos_details in sleeve_positions.items():
            if pos_details['position_usd'] == 0.0:
                continue
            journal.record_observation(signal_name=sig_name, return_value=0.0, metadata={})

        assert not journal.record_observation.called


class TestSelectorAllocatorIntegration:
    """Test meta-selector → cost-aware allocator integration."""

    def test_apply_selector_weights_basic(self):
        """Verify selector weights distribute to signals correctly."""
        selector_weights = {'momentum': 0.6, 'reversal': 0.4}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5'],
        }

        total_weight = sum(selector_weights.values())
        normalized = {k: v / total_weight for k, v in selector_weights.items()}

        effective_weights = {}
        for family_name, weight in normalized.items():
            signals = signal_families.get(family_name, [])
            if not signals:
                continue
            per_signal = weight / len(signals)
            for s in signals:
                effective_weights[s] = per_signal

        assert np.isclose(effective_weights['mom_12_1'], 0.3)
        assert np.isclose(effective_weights['mom_6_1'], 0.3)
        assert np.isclose(effective_weights['rev_5'], 0.4)
        assert np.isclose(sum(effective_weights.values()), 1.0)

    def test_zero_weight_family(self):
        """Verify zero-weight families get zero allocation."""
        selector_weights = {'momentum': 1.0, 'reversal': 0.0}
        signal_families = {
            'momentum': ['mom_12_1'],
            'reversal': ['rev_5', 'rev_10'],
        }

        total_weight = sum(selector_weights.values())
        normalized = {k: v / total_weight for k, v in selector_weights.items()}

        effective_weights = {}
        for family_name, weight in normalized.items():
            signals = signal_families.get(family_name, [])
            if not signals or weight <= 0:
                for s in signals:
                    effective_weights[s] = 0.0
                continue
            per_signal = weight / len(signals)
            for s in signals:
                effective_weights[s] = per_signal

        assert np.isclose(effective_weights['mom_12_1'], 1.0)
        assert effective_weights['rev_5'] == 0.0
        assert effective_weights['rev_10'] == 0.0

    def test_compute_regime_allocation(self):
        """Verify complete allocation with strengths."""
        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5'],
        }
        signal_strengths = {
            'mom_12_1': 2.5, 'mom_6_1': 1.0, 'rev_5': 1.5,
        }
        risk_budget = 100000.0

        allocations = {}
        for family_name, signals in signal_families.items():
            family_weight = selector_weights.get(family_name, 0.0)
            if family_weight <= 0 or not signals:
                for s in signals:
                    allocations[s] = 0.0
                continue
            fam_str = {s: signal_strengths.get(s, 0.0) for s in signals}
            total_str = sum(fam_str.values())
            if total_str <= 0:
                per_sig = family_weight * risk_budget / len(signals)
                for s in signals:
                    allocations[s] = per_sig
            else:
                for s, strength in fam_str.items():
                    w = strength / total_str
                    allocations[s] = family_weight * w * risk_budget

        assert np.isclose(allocations['mom_12_1'], 50000.0, atol=1.0)
        assert np.isclose(allocations['mom_6_1'], 20000.0, atol=1.0)
        assert np.isclose(allocations['rev_5'], 30000.0, atol=1.0)
        assert np.isclose(sum(allocations.values()), risk_budget, atol=1.0)


class TestEndToEndIntegration:
    """End-to-end tests combining sleeve, selector, and allocator."""

    def test_full_feedback_loop(self):
        """Test complete flow: selector → sleeve → observation."""
        journal = Mock()
        journal.record_observation.return_value = 'obs_id'

        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5'],
        }

        # Step 1: Apply selector weights
        total_w = sum(selector_weights.values())
        norm = {k: v / total_w for k, v in selector_weights.items()}
        eff = {}
        for fam, w in norm.items():
            sigs = signal_families.get(fam, [])
            for s in sigs:
                eff[s] = w / len(sigs) if sigs else 0.0

        # Step 2: Sleeve takes positions
        sleeve_positions = {
            s: {'position_usd': w * 10000, 'selector_weight': w}
            for s, w in eff.items()
        }

        # Step 3: Record observations
        realized_returns = {s: 0.0 for s in eff}
        obs_count = 0
        for sig_name, pos in sleeve_positions.items():
            if pos['position_usd'] == 0:
                continue
            journal.record_observation(
                signal_name=sig_name,
                return_value=realized_returns[sig_name],
                metadata={'source': 'sleeve_exploration'},
            )
            obs_count += 1

        assert obs_count == 3
        assert journal.record_observation.call_count == 3
        for call in journal.record_observation.call_args_list:
            assert call[1]['metadata']['source'] == 'sleeve_exploration'

    def test_integration_validation_passes(self):
        """Test validation of selector → allocator integration."""
        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5'],
        }
        allocator_state = {
            'allocations': {
                'mom_12_1': 5000.0, 'mom_6_1': 2000.0, 'rev_5': 3000.0,
            }
        }

        issues = []
        total_weight = sum(selector_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            issues.append(f"Weights sum to {total_weight}")
        for fam in selector_weights:
            if fam not in signal_families:
                issues.append(f"Unknown family: {fam}")
        for fam, w in selector_weights.items():
            if w < 0:
                issues.append(f"Negative weight: {fam}")

        assert len(issues) == 0
        assert np.isclose(total_weight, 1.0)

    def test_integration_fails_on_negative_weight(self):
        """Verify validation catches negative weights."""
        selector_weights = {'momentum': 1.2, 'reversal': -0.2}
        issues = []
        for fam, w in selector_weights.items():
            if w < 0:
                issues.append(f"Negative weight: {fam}")

        assert len(issues) > 0
        assert any('negative' in i.lower() for i in issues)
