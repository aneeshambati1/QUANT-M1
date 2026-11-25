"""One deterministic decision cycle: the forward and backward information path.

Forward:  MarketState -> policy decision -> meta weights -> allocation
Backward: realised marginal contributions -> tracker updates -> next weights

The cycle is pure. It produces a CycleRecord describing what SHOULD happen.
It never transmits an order. apply_realized() is the backward step that closes
the loop: models that produced a realised contribution are updated; models that
were silent decay toward zero authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from ..meta.policy import decide
from ..meta.contribution import weights as meta_weights
from ..meta.state import MarketState
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
        return {
            "as_of": self.as_of,
            "phase": self.phase,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "permits_new_risk": self.permits_new_risk,
            "model_weights": dict(self.model_weights),
            "any_eligible_model": self.any_eligible_model,
            "allocation": self.allocation.as_dict() if self.allocation is not None else None,
            "note": self.note,
        }


def run_cycle(state: MarketState,
              trackers: Mapping,
              forecasts: Optional[Mapping] = None,
              allocator: Optional[Callable] = None,
              schedule: Optional[SessionSchedule] = None,
              now_hour_utc: Optional[int] = None) -> CycleRecord:
    phase = None
    if schedule is not None and now_hour_utc is not None:
        phase = schedule.phase_at(now_hour_utc).value

    decision = decide(state)
    w = meta_weights(trackers)
    eligible = any(v > 0.0 for v in w.values())

    allocation = None
    note = ""
    if not decision.permits_new_risk:
        note = "policy forbids new risk; no allocation"
    elif not eligible:
        note = "no model has demonstrated positive contribution; cash is a position"
    elif allocator is None or forecasts is None:
        note = "allocator or forecasts unavailable; no allocation"
    else:
        if "expected_returns" not in forecasts or "cov" not in forecasts:
            raise ValueError("forecasts must contain expected_returns and cov")
        allocation = allocator(
            forecasts["expected_returns"],
            forecasts["cov"],
            risk_aversion=forecasts.get("risk_aversion", 1.0),
            current_weights=forecasts.get("current_weights"),
            band_frac=forecasts.get("band_frac", 0.0),
        )
        note = "allocated by marginal after-cost contribution"

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


def apply_realized(trackers: Mapping, contributions: Mapping[str, float]) -> dict:
    """Backward pass. Update models with a realised contribution; decay the rest.

    contributions maps model name -> marginal after-cost contribution for the
    just-completed horizon. Models absent from contributions are silent this
    cycle and lose authority via decay_idle(1).
    """
    updated, decayed = [], []
    for name, t in trackers.items():
        if name in contributions:
            t.update(contributions[name])
            updated.append(name)
        else:
            t.decay_idle(1)
            decayed.append(name)
    return {"updated": updated, "decayed": decayed}
