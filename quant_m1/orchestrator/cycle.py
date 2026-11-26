"""Deterministic QUANT decision cycle.

Forward:
MarketState -> policy -> horizon-matched meta weights -> signal blend
-> allocator -> CycleRecord

Backward:
realized marginal after-cost contribution -> tracker update

This module computes decisions only. It does not submit orders.
Execution must pass through the existing Risk Kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from ..meta.policy import decide
from ..meta.contribution import weights as meta_weights
from ..meta.state import MarketState
from .mapping import blend_signals
from .schedule import Phase, SessionSchedule


@dataclass
class CycleRecord:
    as_of: str
    phase: Optional[str]
    action: str
    rule_id: str
    reason: str
    permits_new_risk: bool
    model_weights: dict
    any_eligible_model: bool
    allocation: Optional[object]
    note: str = ""

    def as_dict(self) -> dict:
        allocation = None
        if self.allocation is not None:
            if hasattr(self.allocation, "as_dict"):
                allocation = self.allocation.as_dict()
            else:
                allocation = self.allocation

        return {
            "as_of": self.as_of,
            "phase": self.phase,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "permits_new_risk": self.permits_new_risk,
            "model_weights": dict(self.model_weights),
            "any_eligible_model": self.any_eligible_model,
            "allocation": allocation,
            "note": self.note,
        }


def run_cycle(
    state: MarketState,
    trackers: Mapping,
    forecasts: Optional[Mapping] = None,
    allocator: Optional[Callable] = None,
    schedule: Optional[SessionSchedule] = None,
    now_hour_utc: Optional[int] = None,
) -> CycleRecord:
    """Run one pure decision cycle.

    `forecasts` may contain:
      expected_returns: already blended expected-return vector
      model_signals: {model_name: expected-return vector}
      cov: covariance matrix for the SAME return horizon
      horizon_days: forecast/observation horizon
      risk_aversion, current_weights, band_frac

    If model_signals is supplied, they are blended with tracker weights.
    Otherwise expected_returns is treated as an already blended vector.

    Missing/invalid data fails closed: no allocation is produced.
    """
    phase = None
    if schedule is not None and now_hour_utc is not None:
        phase = schedule.phase_at(now_hour_utc).value

    decision = decide(state)

    horizon_days = None
    if forecasts is not None:
        horizon_days = forecasts.get("horizon_days")

    # A mixed-horizon tracker set must specify the target horizon.
    w = meta_weights(trackers, horizon_days=horizon_days)
    eligible = any(value > 0.0 for value in w.values())

    allocation = None
    note = ""

    if not decision.permits_new_risk:
        note = (
            "Policy prohibits new risk. Existing risk reduction, if permitted, "
            "must be handled by the separate risk-reduction path."
        )

    elif not eligible:
        note = "No model has earned positive authority; cash is a valid position."

    elif allocator is None or forecasts is None:
        note = "Allocator or forecasts unavailable; no allocation."

    else:
        if "cov" not in forecasts:
            note = "Covariance unavailable; no allocation."
        else:
            try:
                if "model_signals" in forecasts:
                    model_signals = forecasts["model_signals"]
                    expected_returns = blend_signals(w, model_signals)
                elif "expected_returns" in forecasts:
                    expected_returns = np.asarray(
                        forecasts["expected_returns"], dtype=float
                    )
                else:
                    note = "Expected returns unavailable; no allocation."
                    expected_returns = None

                if expected_returns is not None:
                    cov = np.asarray(forecasts["cov"], dtype=float)

                    if (
                        expected_returns.ndim != 1
                        or cov.shape != (
                            expected_returns.size,
                            expected_returns.size,
                        )
                        or not np.all(np.isfinite(expected_returns))
                        or not np.all(np.isfinite(cov))
                    ):
                        note = "Forecast shape or finite-value validation failed."
                    else:
                        allocation = allocator(
                            expected_returns,
                            cov,
                            risk_aversion=forecasts.get("risk_aversion", 1.0),
                            current_weights=forecasts.get("current_weights"),
                            band_frac=forecasts.get("band_frac", 0.0),
                        )

                        if not getattr(allocation, "solver_ok", False):
                            allocation = None
                            note = "Allocator did not validate; no new allocation."
                        else:
                            note = (
                                "Model signals blended by horizon-matched "
                                "after-cost contribution weights."
                            )

            except (ValueError, KeyError, TypeError) as exc:
                allocation = None
                note = f"Forecast validation failed: {type(exc).__name__}."

    return CycleRecord(
        as_of=state.as_of,
        phase=phase,
        action=decision.action.value,
        rule_id=decision.rule_id,
        reason=decision.reason,
        permits_new_risk=decision.permits_new_risk,
        model_weights=dict(w),
        any_eligible_model=eligible,
        allocation=allocation,
        note=note,
    )


def apply_realized(
    trackers: Mapping,
    contributions: Mapping[str, float],
    *,
    horizon_days: Optional[int] = None,
) -> dict:
    """Backward pass using paired marginal after-cost contribution.

    Each tracker has a fixed observation horizon. If horizon_days is supplied,
    it must match the tracker's configured horizon. Different horizons must
    use separate trackers, e.g. model_5d and model_21d.
    """
    updated = []
    decayed = []

    for name, tracker in trackers.items():
        if name in contributions:
            tracker.update(
                contributions[name],
                horizon_days=horizon_days,
            )
            updated.append(name)
        else:
            tracker.decay_idle(1)
            decayed.append(name)

    return {"updated": updated, "decayed": decayed}