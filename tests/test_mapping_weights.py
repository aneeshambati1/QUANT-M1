"""blend_signals weight validation."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.orchestrator.mapping import blend_signals


def test_inf_weight_rejected():
    with pytest.raises(ValueError):
        blend_signals({"m": float("inf")}, {"m": np.array([0.01, 0.02])})


def test_nan_weight_rejected():
    with pytest.raises(ValueError):
        blend_signals({"m": float("nan")}, {"m": np.array([0.01, 0.02])})


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        blend_signals({"m": -0.5}, {"m": np.array([0.01, 0.02])})


def test_bool_weight_rejected():
    with pytest.raises(ValueError):
        blend_signals({"m": True}, {"m": np.array([0.01, 0.02])})


def test_valid_blend():
    out = blend_signals({"a": 1.0, "b": 1.0},
                        {"a": np.array([0.02, 0.0]), "b": np.array([0.0, 0.04])})
    assert np.allclose(out, [0.01, 0.02])
