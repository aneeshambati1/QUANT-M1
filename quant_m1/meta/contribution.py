"""Horizon-isolated marginal after-cost contribution tracking.

Weights are earned from paired incremental contribution, not raw model PnL.
Each tracker represents ONE model at ONE holding horizon.

Do not mix daily, weekly, or monthly observations in the same tracker.

PHASE 2 (additive, opt-in): bounded observation history + standard-error
statistics (renormalized finite-sample EWMA mean, Bartlett-tapered weighted
covariance variance, n_eff_corr, se_lcb) and SE live-weight helpers. The
ORIGINAL lcb / status / kappa / weights() path is UNCHANGED. Nothing here is
used by the live path until the Phase 2 flag is enabled and the null suite
has passed. Promotion certificates are issued once by the promotion ledger,
never by repeated tracker status checks.
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
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError("contribution inputs must be real numbers")
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

    # Phase 2 SE parameters (additive; original path untouched).
    se_min_obs: int = 120
    se_max_obs: int = 399

    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    idle: int = 0
    _retired: bool = field(default=False, init=False)
    _lam: float = field(default=0.0, init=False, repr=False)
    _hist: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if (isinstance(self.horizon_days, bool)
                or not isinstance(self.horizon_days, int)
                or self.horizon_days < 1):
            raise ValueError("horizon_days must be an integer >= 1")
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
        if (isinstance(self.se_min_obs, bool) or not isinstance(self.se_min_obs, int)
                or self.se_min_obs < 2):
            raise ValueError("se_min_obs must be an integer >= 2")
        if (isinstance(self.se_max_obs, bool) or not isinstance(self.se_max_obs, int)
                or self.se_max_obs < self.se_min_obs):
            raise ValueError("se_max_obs must be an integer >= se_min_obs")

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

        if horizon_days is not None and (
                isinstance(horizon_days, bool) or not isinstance(horizon_days, int)):
            raise ValueError("horizon_days must be an integer")
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

        # Phase 2: bounded history for SE statistics (additive).
        self._hist.append(float(contribution))
        if len(self._hist) > self.se_max_obs:
            del self._hist[: len(self._hist) - self.se_max_obs]

        if self.n >= self.retire_min_obs and self.mean < -self.retire_band:
            self._retired = True

    def decay_idle(self, steps: int = 1) -> None:
        """Decay authority incrementally after the idle grace period."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be nonnegative")
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

    # ------------------------------------------------------------------
    # Phase 2 SE statistics (opt-in; never used by the original path).
    # ------------------------------------------------------------------
    def se_stats(self, z: float, lag_cap: int = 21) -> dict:
        """Renormalized finite-sample EWMA mean + Bartlett-tapered weighted
        covariance variance of that mean. Fail-closed on invalid variance.

        m_hat = sum(w_i x_i)/sum(w_i), w_i = lam*(1-lam)^(n-1-i)
        Var(m_hat) = [g0*sum(w^2) + 2*sum_k (1-k/(L+1))*g_k*sum_i w_i w_{i+k}]/W^2
        n_eff_corr = g0 / Var(m_hat);  se_lcb = m_hat - z*sqrt(Var).
        """
        if not math.isfinite(z) or z <= 0:
            raise ValueError("z must be finite and > 0")
        if isinstance(lag_cap, bool) or not isinstance(lag_cap, int) or lag_cap < 1:
            raise ValueError("lag_cap must be an integer >= 1")
        n = len(self._hist)
        out = {"valid": False, "n": n, "reason": ""}
        if n < self.se_min_obs:
            out["reason"] = "insufficient_observations"
            return out
        x = self._hist
        decay = 1.0 - self._lam
        w = [self._lam * decay ** (n - 1 - i) for i in range(n)]
        W = sum(w)
        mhat = sum(wi * xi for wi, xi in zip(w, x)) / W
        xbar = sum(x) / n
        d = [xi - xbar for xi in x]
        g0 = sum(di * di for di in d) / n
        if not math.isfinite(g0) or g0 <= 0:
            out["reason"] = "nonpositive_gamma0"
            return out
        L = min(lag_cap, n - 1)
        acc = g0 * sum(wi * wi for wi in w)
        for k in range(1, L + 1):
            gk = sum(d[t] * d[t - k] for t in range(k, n)) / n
            cross = sum(w[i] * w[i + k] for i in range(n - k))
            acc += 2.0 * (1.0 - k / (L + 1)) * gk * cross
        var = acc / (W * W)
        if not math.isfinite(var) or var <= 0:
            out["reason"] = "nonpositive_variance"
            return out
        se = math.sqrt(var)
        out.update({
            "valid": True, "reason": "ok", "mean": mhat, "var_mean": var,
            "se": se, "n_eff_corr": g0 / var, "se_lcb": mhat - z * se,
            "gamma0": g0, "lag_cap": L,
        })
        return out

    def se_snapshot(self, z: float, lag_cap: int = 21) -> dict:
        return {**self.snapshot(), "se": self.se_stats(z, lag_cap)}


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
        if (isinstance(horizon_days, bool) or not isinstance(horizon_days, int)
                or horizon_days < 1):
            raise ValueError("horizon_days must be an integer >= 1")
        selected_horizon = horizon_days

    selected = {
        name: tracker
        for name, tracker in trackers.items()
        if selected_horizon is None
        or tracker.horizon_days == selected_horizon
    }

    raw = {name: tracker.raw_weight() for name in selected.items()} if False else {
        name: tracker.raw_weight() for name, tracker in selected.items()
    }
    total = sum(raw.values())

    if total <= 0.0:
        return {name: 0.0 for name in selected}

    return {name: value / total for name, value in raw.items()}


def se_raw_weights(
    trackers: Mapping[str, ContributionTracker],
    z: float,
    lag_cap: int = 21,
) -> Dict[str, float]:
    """Phase 2 evidence-based raw weight: max(0, se_lcb). Opt-in only."""
    raw = {}
    for name, t in trackers.items():
        st = t.se_stats(z, lag_cap)
        raw[name] = max(0.0, st["se_lcb"]) if st["valid"] else 0.0
    return raw


def se_weights(
    trackers: Mapping[str, ContributionTracker],
    z: float,
    lag_cap: int = 21,
    c_model: float = 0.40,
    c_family: float = 0.50,
    families: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """Phase 2 live-weight specification (Amendment C), opt-in only.

    Normalize raw over the given set; deterministic water-filling with
    per-model and per-family caps (<=10 iterations); remainder -> cash.
    All-zero raw -> all-zero weights (cash), never equal weights.
    """
    if not (0 < c_model <= 1 and 0 < c_family <= 1):
        raise ValueError("caps must be in (0, 1]")
    raw = se_raw_weights(trackers, z, lag_cap)
    total = sum(raw.values())
    if total <= 0.0:
        return {name: 0.0 for name in trackers}
    fam = dict(families) if families else {name: name for name in trackers}
    w = {k: v / total for k, v in raw.items()}
    for _ in range(10):
        changed = False
        for k in list(w):
            if w[k] > c_model:
                w[k] = c_model
                changed = True
        fs: Dict[str, float] = {}
        for k, v in w.items():
            fs[fam[k]] = fs.get(fam[k], 0.0) + v
        for f, s in fs.items():
            if s > c_family:
                scale = c_family / s
                for k in w:
                    if fam[k] == f:
                        w[k] *= scale
                changed = True
        if not changed:
            break
        free = 1.0 - sum(w.values())
        elig = [k for k in w
                if w[k] < c_model - 1e-12 and fs.get(fam[k], 0.0) < c_family - 1e-12]
        rsum = sum(raw[k] for k in elig)
        if free > 1e-12 and rsum > 0:
            for k in elig:
                w[k] += free * raw[k] / rsum
        else:
            break
    return w


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
        "eligible_count": sum(1 for tracker in selected_trackers.values() if tracker.eligible),
        "total_weight": sum(selected_weights.values()),
    }
