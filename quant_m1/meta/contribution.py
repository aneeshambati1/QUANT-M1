"""Marginal after-cost contribution tracking and meta-weighting.

Backward path of the bidirectional loop:

    forward:  weights -> allocation -> execution
    backward: realized PnL -> marginal contribution -> EWMA mean/variance
              -> lower confidence bound -> weights

Exponential propagation is explicit and parameterised by half-life:

    lambda = 1 - 2^(-1/h)
    m_t    = (1-lambda) m_{t-1} + lambda c_t
    v_t    = (1-lambda) v_{t-1} + lambda (c_t - m_{t-1})^2
    LCB    = m_t - kappa sqrt(v_t)

Weight is allocated only when LCB > 0 AND probation is cleared. A model with
no demonstrated positive contribution receives zero weight.

UNITS: fractional return per observation, net of cost. The caller must supply
the same horizon for every model in a comparison set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping


class ModelStatus(Enum):
    PROBATION = "probation"
    ACTIVE = "active"
    DEMOTED = "demoted"
    RETIRED = "retired"


def marginal_contribution(with_model: float,
                          without_model: float,
                          cost_frac: float) -> float:
    """Incremental after-cost contribution of including a model.

    with_model / without_model are realised fractional returns over the SAME
    horizon and the SAME universe, differing only by inclusion of the model.
    cost_frac is the incremental round-trip cost attributable to the model.

    Paired difference, not raw return. A model earning 20bp while the book
    would have earned 25bp without it has NEGATIVE contribution.
    """
    if not all(math.isfinite(x) for x in (with_model, without_model, cost_frac)):
        raise ValueError("contribution inputs must be finite")
    if cost_frac < 0:
        raise ValueError("cost_frac must be >= 0")
    return (with_model - without_model) - cost_frac


def _ewma_lambda(half_life_obs: float) -> float:
    if half_life_obs <= 0 or not math.isfinite(half_life_obs):
        raise ValueError("half_life_obs must be positive and finite")
    return 1.0 - math.pow(2.0, -1.0 / half_life_obs)


@dataclass
class ContributionTracker:
    """Per-model exponentially-weighted contribution with confidence penalty.

    prior_std seeds variance so an unobserved model has LCB < 0 and therefore
    zero weight. Authority must be earned, not assumed.
    """
    half_life_obs: float = 60.0
    kappa: float = 1.0
    probation_obs: int = 30
    prior_std: float = 0.01
    retire_min_obs: int = 120
    retire_band: float = 0.002
    idle_grace: int = 5

    mean: float = 0.0
    var: float = field(default=0.0)
    n: int = 0
    idle: int = 0
    _retired: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.kappa < 0:
            raise ValueError("kappa must be >= 0")
        if self.probation_obs < 1:
            raise ValueError("probation_obs must be >= 1")
        if self.prior_std <= 0:
            raise ValueError("prior_std must be > 0")
        if self.retire_min_obs < self.probation_obs:
            raise ValueError("retire_min_obs must be >= probation_obs")
        if self.retire_band <= 0:
            raise ValueError("retire_band must be > 0")
        if self.idle_grace < 0:
            raise ValueError("idle_grace must be >= 0")
        self._lam = _ewma_lambda(self.half_life_obs)
        if self.var == 0.0:
            self.var = self.prior_std ** 2

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    @property
    def lcb(self) -> float:
        """Lower confidence bound. The only quantity that earns weight."""
        return self.mean - self.kappa * self.std

    def update(self, contribution: float) -> None:
        """Backward pass. One realised marginal after-cost observation."""
        if self._retired:
            return
        if not math.isfinite(contribution):
            raise ValueError("contribution must be finite")
        prev_mean = self.mean
        self.mean = (1.0 - self._lam) * prev_mean + self._lam * contribution
        resid = contribution - prev_mean
        self.var = (1.0 - self._lam) * self.var + self._lam * resid * resid
        self.n += 1
        self.idle = 0
        if self.n >= self.retire_min_obs and self.mean < -self.retire_band:
            self._retired = True

    def decay_idle(self, steps: int = 1) -> None:
        """No permanent authority: silence decays weight toward zero.

        Mean decays geometrically; variance is held, so the confidence penalty
        increasingly dominates and LCB falls.

        Grace accounting is INCREMENTAL. Only the portion of this call that
        crosses beyond idle_grace decays. This makes repeated small calls
        equivalent to one large call:

            decay_idle(1) x 10  ==  decay_idle(10)

        The earlier version recomputed the full past-grace span on every call,
        which double-counted and over-decayed under repeated invocation.
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if self._retired or steps == 0:
            return
        prev_idle = self.idle
        self.idle += steps
        if self.idle <= self.idle_grace:
            return
        already_past = max(0, prev_idle - self.idle_grace)
        now_past = self.idle - self.idle_grace
        effective = now_past - already_past
        if effective > 0:
            self.mean *= math.pow(1.0 - self._lam, effective)

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
            "n": self.n,
            "idle": self.idle,
            "mean": self.mean,
            "std": self.std,
            "lcb": self.lcb,
            "status": self.status.value,
            "half_life_obs": self.half_life_obs,
            "kappa": self.kappa,
        }


def weights(trackers: Mapping[str, ContributionTracker]) -> Dict[str, float]:
    """Normalise raw LCB weights across the eligible set.

    Returns all-zero when no model is eligible. That is not a failure; it is
    the correct answer. Cash is a position. The caller must not renormalise
    an all-zero result into equal weights.
    """
    raw = {name: t.raw_weight() for name, t in trackers.items()}
    total = sum(raw.values())
    if total <= 0.0:
        return {name: 0.0 for name in trackers}
    return {name: v / total for name, v in raw.items()}


def authority_report(trackers: Mapping[str, ContributionTracker]) -> dict:
    """Auditable snapshot. Every number carries its unit and convention."""
    w = weights(trackers)
    return {
        "unit": "fractional_return_per_observation",
        "cost_basis": "net_after_cost",
        "weight_convention": "max(0, mean - kappa*std), normalised over ACTIVE",
        "models": {
            name: {**t.snapshot(), "weight": w[name]}
            for name, t in trackers.items()
        },
        "eligible_count": sum(1 for t in trackers.values() if t.eligible),
        "total_weight": sum(w.values()),
    }