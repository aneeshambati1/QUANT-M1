"""Horizon-isolated marginal after-cost contribution tracking.

Weights are earned from paired incremental contribution, not raw model PnL.
Each tracker represents ONE model at ONE holding horizon.

Do not mix daily, weekly, or monthly observations in the same tracker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional


class ModelStatus(Enum):
    PROBATION = "probation"
    ACTIVE = "active"
    DEMOTED = "demoted"
    RETIRED = "retired"


def marginal_contribution(
    with_model: float,
    without_model: float,
    cost_frac: float,
) -> float:
    """Incremental after-cost contribution over the same horizon."""
    values = (with_model, without_model, cost_frac)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("contribution inputs must be finite")
    if cost_frac < 0:
        raise ValueError("cost_frac must be >= 0")
    return (with_model - without_model) - cost_frac


def _ewma_lambda(half_life_obs: float) -> float:
    if not math.isfinite(half_life_obs) or half_life_obs <= 0:
        raise ValueError("half_life_obs must be positive and finite")
    return 1.0 - math.pow(2.0, -1.0 / half_life_obs)


@dataclass
class ContributionTracker:
    """EWMA contribution tracker for one model and one fixed horizon."""

    half_life_obs: float = 60.0
    kappa: float = 1.0
    probation_obs: int = 30
    prior_std: float = 0.01
    retire_min_obs: int = 120
    retire_band: float = 0.002
    idle_grace: int = 5

    # Fixed return horizon represented by each observation.
    horizon_days: int = 1

    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    idle: int = 0
    _retired: bool = field(default=False, init=False)
    _lam: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and >= 0")
        if self.probation_obs < 1:
            raise ValueError("probation_obs must be >= 1")
        if not math.isfinite(self.prior_std) or self.prior_std <= 0:
            raise ValueError("prior_std must be positive and finite")
        if self.retire_min_obs < self.probation_obs:
            raise ValueError("retire_min_obs must be >= probation_obs")
        if not math.isfinite(self.retire_band) or self.retire_band <= 0:
            raise ValueError("retire_band must be positive and finite")
        if self.idle_grace < 0:
            raise ValueError("idle_grace must be >= 0")
        if self.n < 0 or self.idle < 0:
            raise ValueError("n and idle must be >= 0")
        if not math.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not math.isfinite(self.var) or self.var < 0:
            raise ValueError("var must be finite and >= 0")

        self._lam = _ewma_lambda(self.half_life_obs)

        # Seed uncertainty so a new model cannot immediately earn authority.
        if self.var == 0.0:
            self.var = self.prior_std ** 2

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    @property
    def lcb(self) -> float:
        return self.mean - self.kappa * self.std

    def update(
        self,
        contribution: float,
        *,
        horizon_days: Optional[int] = None,
    ) -> None:
        """Update with one paired, net-of-cost contribution observation."""
        if self._retired:
            return

        if not math.isfinite(contribution):
            raise ValueError("contribution must be finite")

        observed_horizon = (
            self.horizon_days if horizon_days is None else horizon_days
        )
        if observed_horizon != self.horizon_days:
            raise ValueError(
                f"horizon mismatch: tracker={self.horizon_days}d, "
                f"observation={observed_horizon}d"
            )

        previous_mean = self.mean
        self.mean = (
            (1.0 - self._lam) * previous_mean
            + self._lam * contribution
        )

        residual = contribution - previous_mean
        self.var = (
            (1.0 - self._lam) * self.var
            + self._lam * residual * residual
        )

        self.n += 1
        self.idle = 0

        if self.n >= self.retire_min_obs and self.mean < -self.retire_band:
            self._retired = True

    def decay_idle(self, steps: int = 1) -> None:
        """Decay authority incrementally after the idle grace period."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if self._retired or steps == 0:
            return

        previous_idle = self.idle
        self.idle += steps

        previous_past_grace = max(0, previous_idle - self.idle_grace)
        current_past_grace = max(0, self.idle - self.idle_grace)
        effective_steps = current_past_grace - previous_past_grace

        if effective_steps > 0:
            self.mean *= math.pow(1.0 - self._lam, effective_steps)

    @property
    def status(self) -> ModelStatus:
        if self._retired:
            return ModelStatus.RETIRED
        if self.n < self.probation_obs:
            return ModelStatus.PROBATION
        if self.lcb > 0.0:
            return ModelStatus.ACTIVE
        return ModelStatus.DEMOTED

    @property
    def eligible(self) -> bool:
        return self.status is ModelStatus.ACTIVE

    def raw_weight(self) -> float:
        return max(0.0, self.lcb) if self.eligible else 0.0

    def snapshot(self) -> dict:
        return {
            "horizon_days": self.horizon_days,
            "n": self.n,
            "idle": self.idle,
            "mean": self.mean,
            "std": self.std,
            "lcb": self.lcb,
            "status": self.status.value,
            "half_life_obs": self.half_life_obs,
            "kappa": self.kappa,
        }


def weights(
    trackers: Mapping[str, ContributionTracker],
    *,
    horizon_days: Optional[int] = None,
) -> Dict[str, float]:
    """Normalize positive LCB weights within exactly one horizon.

    If tracker horizons differ, caller must explicitly select a horizon.
    All-zero weights are preserved; do not convert them to equal weights.
    """
    if horizon_days is None:
        horizons = {t.horizon_days for t in trackers.values()}
        if len(horizons) > 1:
            raise ValueError(
                "mixed tracker horizons; pass horizon_days explicitly"
            )
        selected_horizon = next(iter(horizons), None)
    else:
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        selected_horizon = horizon_days

    selected = {
        name: tracker
        for name, tracker in trackers.items()
        if selected_horizon is None
        or tracker.horizon_days == selected_horizon
    }

    raw = {name: tracker.raw_weight() for name, tracker in selected.items()}
    total = sum(raw.values())

    if total <= 0.0:
        return {name: 0.0 for name in selected}

    return {name: value / total for name, value in raw.items()}


def authority_report(
    trackers: Mapping[str, ContributionTracker],
    *,
    horizon_days: Optional[int] = None,
) -> dict:
    selected_weights = weights(trackers, horizon_days=horizon_days)

    selected_trackers = {
        name: tracker
        for name, tracker in trackers.items()
        if name in selected_weights
    }

    return {
        "unit": "fractional_return_per_observation",
        "cost_basis": "net_after_cost",
        "horizon_days": (
            horizon_days
            if horizon_days is not None
            else next(
                iter({t.horizon_days for t in selected_trackers.values()}),
                None,
            )
        ),
        "weight_convention": "max(0, mean - kappa*std), normalized over ACTIVE",
        "models": {
            name: {
                **tracker.snapshot(),
                "weight": selected_weights[name],
            }
            for name, tracker in selected_trackers.items()
        },
        "eligible_count": sum(
            1 for tracker in selected_trackers.values() if tracker.eligible
        ),
        "total_weight": sum(selected_weights.values()),
    }