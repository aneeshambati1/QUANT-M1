"""Exploration sleeve: bounded, evidence-proportional paper sizing.

RESEARCH OVERLAY. Does not route orders, does not grant authority, does not
bypass the Risk Kernel, and does not enable Phase 2. Its purpose is to break
the authority deadlock (a zero-weight model can never earn observations) by
sizing a small, capped PAPER exploration book whose weights are proportional
to measured after-cost evidence, with a pre-registered exploration floor.

Freezing this module changes no live path. Wiring it to feed observations into
the ProbeJournal is a SEPARATE, owner-approved activation step.
"""
from __future__ import annotations
from typing import Dict, Mapping
from .contribution import ModelStatus

SLEEVE_CAP_FRAC = 0.10      # max total sleeve weight (pre-registered)
MIN_PROBE_WEIGHT = 0.005    # exploration floor per provisional model
MAX_MODEL_WEIGHT = 0.05     # per-model cap within the sleeve
KELLY_FRACTION = 0.25       # fractional-Kelly multiplier on evidence


def sleeve_weights(trackers: Mapping[str, object], horizon_days: int = 1) -> Dict[str, float]:
    """Bounded exploration weights for provisional (not-yet-eligible) models.

    - retired models -> 0
    - already-eligible models -> excluded (they use the full-weight path)
    - n==0 models -> exploration floor MIN_PROBE_WEIGHT
    - otherwise -> fractional-Kelly on positive EWMA evidence, capped
    - total is capped at SLEEVE_CAP_FRAC (proportional shrink).
    """
    weights: Dict[str, float] = {}
    for name, t in trackers.items():
        if getattr(t, "horizon_days", horizon_days) != horizon_days:
            continue
        status = getattr(t, "status", None)
        if status is ModelStatus.RETIRED:
            weights[name] = 0.0
            continue
        if getattr(t, "eligible", False):
            continue
        n = getattr(t, "n", 0)
        if n == 0:
            weights[name] = MIN_PROBE_WEIGHT
            continue
        mean = getattr(t, "mean", 0.0)
        var = max(getattr(t, "var", 0.0), 1e-12)
        kelly = KELLY_FRACTION * max(mean, 0.0) / var
        weights[name] = max(MIN_PROBE_WEIGHT, min(kelly, MAX_MODEL_WEIGHT))
    total = sum(weights.values())
    if total > SLEEVE_CAP_FRAC:
        scale = SLEEVE_CAP_FRAC / total
        weights = {k: v * scale for k, v in weights.items()}
    return weights


def sleeve_report(weights: Dict[str, float]) -> dict:
    return {
        "sleeve_cap_frac": SLEEVE_CAP_FRAC,
        "total_sleeve_weight": sum(weights.values()),
        "n_provisional_models": sum(1 for v in weights.values() if v > 0),
        "weights": dict(weights),
        "live_routing": False,
        "grants_authority": False,
        "phase2_enabled": False,
        "disclosure": "Paper-only exploration sleeve. No live orders. No authority.",
    }
