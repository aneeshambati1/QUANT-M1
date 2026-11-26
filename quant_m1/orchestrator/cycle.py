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

from ..allocate.engine import AllocationError
from ..allocate.optimize import OptimizationError
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
    if schedule is not None and now_hour_
