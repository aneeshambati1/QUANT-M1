"""Regime-aware meta-selector across signal families.

Deterministic, pre-registered regime routing. Returns weights over signal
families based on MarketState.regime. No learning, no tuning, no live routing.
"""
from __future__ import annotations
from typing import Dict

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "calm":     {"momentum": 0.50, "reversal": 0.20, "carry": 0.30},
    "elevated": {"momentum": 0.35, "reversal": 0.25, "carry": 0.40},
    "risk_off": {"momentum": 0.20, "reversal": 0.20, "carry": 0.60},
    "crisis":   {"momentum": 0.10, "reversal": 0.10, "carry": 0.80},
    "unknown":  {"momentum": 1/3, "reversal": 1/3, "carry": 1/3},
}


def select_weights(state) -> Dict[str, float]:
    """Return signal-family weights for the current regime (sums to 1)."""
    regime = getattr(getattr(state, "regime", None), "value", "unknown")
    return dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["unknown"]))


def renormalize_available(weights: Dict[str, float], available: Dict[str, bool]) -> Dict[str, float]:
    """Drop unavailable families and renormalize the rest to sum to 1."""
    kept = {k: v for k, v in weights.items() if available.get(k, False)}
    total = sum(kept.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in kept.items()}
