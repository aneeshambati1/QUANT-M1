"""Regime-aware meta-selector across signal families.

Deterministic, pre-registered regime routing. Returns weights over signal
families based on MarketState.regime. No learning, no tuning, no live routing.

When survivor factor data is available, blends empirical family effectiveness
scores with the pre-registered regime weights. Falls back to pre-registered
weights when survivor data is absent.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

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


def select_weights_with_survivors(state, survivor_report: Optional[dict] = None,
                                   blend_alpha: float = 0.5) -> Dict[str, float]:
    """Blend pre-registered regime weights with empirical survivor family scores.

    Args:
        state: MarketState with .regime attribute
        survivor_report: output of factor_integration_report(), or None
        blend_alpha: weight on empirical scores (0=pure pre-reg, 1=pure empirical)

    Returns:
        Signal-family weights summing to 1. Maps survivor families to the
        three base families (momentum/reversal/carry) via:
          - industry_momentum, seasonality -> momentum
          - short_reversal -> reversal
          - earnings_events, external_financing, short_volume,
            options_informed, revenue_surprise, tax_accruals -> carry
    """
    base = select_weights(state)
    if survivor_report is None or blend_alpha <= 0:
        return base

    family_scores = survivor_report.get("family_scores", {})
    if not family_scores:
        return base

    FAMILY_TO_BASE = {
        "industry_momentum": "momentum",
        "seasonality": "momentum",
        "short_reversal": "reversal",
        "earnings_events": "carry",
        "external_financing": "carry",
        "short_volume": "carry",
        "options_informed": "carry",
        "revenue_surprise": "carry",
        "tax_accruals": "carry",
    }

    base_scores = {"momentum": 0.0, "reversal": 0.0, "carry": 0.0}
    for fam, score in family_scores.items():
        base_fam = FAMILY_TO_BASE.get(fam)
        if base_fam:
            base_scores[base_fam] += max(score, 0.0)

    total_score = sum(base_scores.values())
    if total_score <= 0:
        return base

    empirical = {k: v / total_score for k, v in base_scores.items()}

    blended = {}
    for k in base:
        blended[k] = (1 - blend_alpha) * base[k] + blend_alpha * empirical.get(k, 0.0)

    total = sum(blended.values())
    if total <= 0:
        return base
    return {k: v / total for k, v in blended.items()}
