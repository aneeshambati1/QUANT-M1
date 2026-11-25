"""Model-to-asset signal mapping.

Translates model authority (meta-weights) and model forecasts into a single
blended asset-level expected return vector. This closes the gap between
"which model has authority" and "what the asset-level optimizer sees".
"""
from __future__ import annotations

import numpy as np
from typing import Mapping


def blend_signals(model_weights: Mapping[str, float],
                  model_signals: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute the weighted average of asset-level expected returns.

    Args:
        model_weights: {model_name: weight}. Models with 0 weight are ignored.
        model_signals: {model_name: 1D array of expected returns per asset}.
                       All arrays must share length and asset ordering.

    Returns:
        1D numpy array of blended expected returns, normalised by total
        active weight.
    """
    if not model_weights or not model_signals:
        raise ValueError("model_weights and model_signals must not be empty")

    active_models = [m for m, w in model_weights.items() if w > 0.0]
    if not active_models:
        raise ValueError("at least one model must have weight > 0")

    n_assets = len(model_signals[active_models[0]])
    blended = np.zeros(n_assets, dtype=float)
    total_weight = 0.0

    for m in active_models:
        if m not in model_signals:
            raise KeyError(f"model {m} has weight but no signal provided")
        sig = np.asarray(model_signals[m], dtype=float)
        if sig.shape != (n_assets,):
            raise ValueError(f"signal shape mismatch for model {m}")
        if not np.all(np.isfinite(sig)):
            raise ValueError(f"signal for model {m} contains non-finite values")

        w = model_weights[m]
        blended += w * sig
        total_weight += w

    if total_weight <= 0.0:
        raise ValueError("total active weight must be > 0")

    return blended / total_weight