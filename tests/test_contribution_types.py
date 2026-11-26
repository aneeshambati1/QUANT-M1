"""Contribution tracker type hardening."""
from __future__ import annotations

import pytest

from quant_m1.meta.contribution import ContributionTracker, marginal_contribution, weights


def test_marginal_contribution_rejects_bool():
    with pytest.raises(ValueError):
        marginal_contribution(True, 0.0, 0.0)


def test_marginal_contribution_rejects_string():
    with pytest.raises(ValueError):
        marginal_contribution("0.01", 0.0, 0.0)


def test_tracker_float_horizon_rejected():
    with pytest.raises(ValueError):
        ContributionTracker(horizon_days=1.5)


def test_tracker_bool_horizon_rejected():
    with pytest.raises(ValueError):
        ContributionTracker(horizon_days=True)


def test_update_float_horizon_rejected():
    with pytest.raises(ValueError):
        ContributionTracker().update(0.01, horizon_days=1.0)


def test_weights_float_horizon_rejected():
    with pytest.raises(ValueError):
        weights({"m": ContributionTracker()}, horizon_days=1.0)


def test_decay_idle_bool_steps_rejected():
    with pytest.raises(ValueError):
        ContributionTracker().decay_idle(True)


def test_normal_contribution_still_works():
    assert marginal_contribution(0.02, 0.01, 0.002) == pytest.approx(0.008)
