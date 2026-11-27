"""Horizon-isolated marginal after-cost contribution tracking.

Phase 2 standard-error statistics use a prewhitened AR(1) filter followed by
a Bartlett HAC estimator on the filtered residuals.

The estimator is a proposed protocol amendment. It is NOT authorized for
promotion until the null suite passes and the protocol is formally re-frozen.

Missing promotion authority fails closed to cash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional


SE_ESTIMATOR_ID = "prewhitened_ar1_bartlett_hac_v1"
SE_PHI_ABS_LIMIT = 0.995


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
    values = (with_model, without_model, cost_frac)

    if any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        for x in values
    ):
        raise ValueError("contribution inputs must be real numbers")

    if not all(math.isfinite(x) for x in values):
        raise ValueError("contribution inputs must be finite")

    if cost_frac < 0:
        raise ValueError("cost_frac must be >= 0")

    return (with_model - without_model) - cost_frac


def _ewma_lambda(half_life_obs: float) -> float:
    if (
        isinstance(half_life_obs, bool)
        or not isinstance(half_life_obs, (int, float))
        or not math.isfinite(half_life_obs)
        or half_life_obs <= 0
    ):
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
    horizon_days: int = 1
    se_min_obs: int = 120
    se_max_obs: int = 399

    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    idle: int = 0

    _retired: bool = field(default=False, init=False)
    _lam: float = field(default=0.0, init=False, repr=False)
    _hist: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon_days, bool)
            or not isinstance(self.horizon_days, int)
            or self.horizon_days < 1
        ):
            raise ValueError("horizon_days must be an integer >= 1")

        if (
            isinstance(self.kappa, bool)
            or not isinstance(self.kappa, (int, float))
            or not math.isfinite(self.kappa)
            or self.kappa < 0
        ):
            raise ValueError("kappa must be finite and >= 0")

        if (
            isinstance(self.probation_obs, bool)
            or not isinstance(self.probation_obs, int)
            or self.probation_obs < 1
        ):
            raise ValueError("probation_obs must be an integer >= 1")

        if (
            isinstance(self.prior_std, bool)
            or not isinstance(self.prior_std, (int, float))
            or not math.isfinite(self.prior_std)
            or self.prior_std <= 0
        ):
            raise ValueError("prior_std must be positive and finite")

        if (
            isinstance(self.retire_min_obs, bool)
            or not isinstance(self.retire_min_obs, int)
            or self.retire_min_obs < self.probation_obs
        ):
            raise ValueError("retire_min_obs must be >= probation_obs")

        if (
            isinstance(self.retire_band, bool)
            or not isinstance(self.retire_band, (int, float))
            or not math.isfinite(self.retire_band)
            or self.retire_band <= 0
        ):
            raise ValueError("retire_band must be positive and finite")

        for name, value in (
            ("idle_grace", self.idle_grace),
            ("n", self.n),
            ("idle", self.idle),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")

        if not math.isfinite(self.mean):
            raise ValueError("mean must be finite")

        if not math.isfinite(self.var) or self.var < 0:
            raise ValueError("var must be finite and >= 0")

        if (
            isinstance(self.se_min_obs, bool)
            or not isinstance(self.se_min_obs, int)
            or self.se_min_obs < 2
        ):
            raise ValueError("se_min_obs must be an integer >= 2")

        if (
            isinstance(self.se_max_obs, bool)
            or not isinstance(self.se_max_obs, int)
            or self.se_max_obs < self.se_min_obs
        ):
            raise ValueError("se_max_obs must be an integer >= se_min_obs")

        self._lam = _ewma_lambda(self.half_life_obs)

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
        if self._retired:
            return

        if (
            isinstance(contribution, bool)
            or not isinstance(contribution, (int, float))
            or not math.isfinite(contribution)
        ):
            raise ValueError("contribution must be finite")

        if horizon_days is not None and (
            isinstance(horizon_days, bool)
            or not isinstance(horizon_days, int)
        ):
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

        self._hist.append(float(contribution))

        if len(self._hist) > self.se_max_obs:
            excess = len(self._hist) - self.se_max_obs
            del self._hist[:excess]

        if (
            self.n >= self.retire_min_obs
            and self.mean < -self.retire_band
        ):
            self._retired = True

    def decay_idle(self, steps: int = 1) -> None:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 0
        ):
            raise ValueError("steps must be nonnegative")

        if self._retired or steps == 0:
            return

        old_idle = self.idle
        self.idle += steps

        old_after = max(0, old_idle - self.idle_grace)
        new_after = max(0, self.idle - self.idle_grace)
        effective = new_after - old_after

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

    def se_stats(self, z: float, lag_cap: int = 21) -> dict:
        """Prewhitened AR(1) + Bartlett HAC variance of the EWMA mean.

        The EWMA-weighted statistic is written as a filtered initial state
        plus a weighted sum of AR(1) residuals:

            X[t] = phi * X[t-1] + e[t]

        The estimated variance includes:
        - the stationary initial-state contribution;
        - the residual variance contribution;
        - Bartlett-tapered residual autocovariances through lag_cap.

        A near-unit-root estimate fails closed. This estimator is a proposed
        protocol amendment and must not be treated as approved Phase 2 authority.
        """
        if (
            isinstance(z, bool)
            or not isinstance(z, (int, float))
            or not math.isfinite(z)
            or z <= 0
        ):
            raise ValueError("z must be finite and > 0")

        if (
            isinstance(lag_cap, bool)
            or not isinstance(lag_cap, int)
            or lag_cap < 1
        ):
            raise ValueError("lag_cap must be an integer >= 1")

        x = self._hist
        n = len(x)

        out = {
            "valid": False,
            "n": n,
            "reason": "",
            "estimator_id": SE_ESTIMATOR_ID,
        }

        if n < self.se_min_obs:
            out["reason"] = "insufficient_observations"
            return out

        if not all(math.isfinite(value) for value in x):
            out["reason"] = "nonfinite_observations"
            return out

        decay = 1.0 - self._lam

        raw_weights = [
            self._lam * decay ** (n - 1 - i)
            for i in range(n)
        ]

        weight_sum = math.fsum(raw_weights)

        if (
            not math.isfinite(weight_sum)
            or weight_sum <= 0.0
        ):
            out["reason"] = "invalid_weight_sum"
            return out

        weights = [value / weight_sum for value in raw_weights]

        mean_hat = math.fsum(
            weight * value
            for weight, value in zip(weights, x)
        )

        sample_mean = math.fsum(x) / n
        centered = [value - sample_mean for value in x]

        denominator = math.fsum(
            value * value
            for value in centered[:-1]
        )

        if not math.isfinite(denominator) or denominator <= 0.0:
            out["reason"] = "nonpositive_gamma0"
            return out

        numerator = math.fsum(
            centered[t] * centered[t - 1]
            for t in range(1, n)
        )

        phi_hat = numerator / denominator

        if (
            not math.isfinite(phi_hat)
            or abs(phi_hat) >= SE_PHI_ABS_LIMIT
        ):
            out["reason"] = "prewhitening_phi_out_of_bounds"
            out["phi_hat"] = phi_hat
            return out

        residuals = [
            centered[t] - phi_hat * centered[t - 1]
            for t in range(1, n)
        ]

        residual_count = len(residuals)
        residual_mean = math.fsum(residuals) / residual_count

        residuals = [
            value - residual_mean
            for value in residuals
        ]

        gamma0 = (
            math.fsum(value * value for value in residuals)
            / residual_count
        )

        if not math.isfinite(gamma0) or gamma0 <= 0.0:
            out["reason"] = "nonpositive_gamma0"
            out["phi_hat"] = phi_hat
            return out

        # Coefficient on the initial stationary state:
        # c0 = sum_t a[t] * phi**t.
        c0 = 0.0
        phi_power = 1.0

        for weight in weights:
            c0 += weight * phi_power
            phi_power *= phi_hat

        # Coefficients on innovations e[1], ..., e[n-1]:
        # c[j] = sum_{t=j+1}^{n-1} a[t] * phi**(t-j-1).
        residual_weights = [0.0] * residual_count
        next_weight = 0.0

        for j in range(residual_count - 1, -1, -1):
            next_weight = weights[j + 1] + phi_hat * next_weight
            residual_weights[j] = next_weight

        initial_state_variance = gamma0 / (1.0 - phi_hat * phi_hat)

        var_mean = (
            initial_state_variance * c0 * c0
            + gamma0 * math.fsum(
                value * value
                for value in residual_weights
            )
        )

        # Bartlett HAC correction on the prewhitened residual process.
        effective_lag = min(lag_cap, residual_count - 1)

        for k in range(1, effective_lag + 1):
            gamma_k = (
                math.fsum(
                    residuals[t] * residuals[t - k]
                    for t in range(k, residual_count)
                )
                / residual_count
            )

            cross_weight = math.fsum(
                residual_weights[i] * residual_weights[i + k]
                for i in range(residual_count - k)
            )

            bartlett_weight = 1.0 - k / (effective_lag + 1.0)

            var_mean += (
                2.0
                * bartlett_weight
                * gamma_k
                * cross_weight
            )

        if not math.isfinite(var_mean) or var_mean <= 0.0:
            out["reason"] = "nonpositive_variance"
            out["phi_hat"] = phi_hat
            out["gamma0"] = gamma0
            out["lag_cap"] = effective_lag
            return out

        standard_error = math.sqrt(var_mean)
        se_lcb = mean_hat - z * standard_error
        n_eff_corr = gamma0 / var_mean

        if not all(
            math.isfinite(value)
            for value in (
                mean_hat,
                standard_error,
                se_lcb,
                n_eff_corr,
            )
        ):
            out["reason"] = "nonfinite_statistic"
            return out

        out.update({
            "valid": True,
            "reason": "ok",
            "mean": mean_hat,
            "var_mean": var_mean,
            "se": standard_error,
            "n_eff_corr": n_eff_corr,
            "se_lcb": se_lcb,
            "gamma0": gamma0,
            "phi_hat": phi_hat,
            "lag_cap": effective_lag,
        })

        return out

    def se_snapshot(self, z: float, lag_cap: int = 21) -> dict:
        return {
            **self.snapshot(),
            "se": self.se_stats(z, lag_cap),
        }


def weights(
    trackers: Mapping[str, ContributionTracker],
    *,
    horizon_days: Optional[int] = None,
) -> Dict[str, float]:
    if horizon_days is None:
        horizons = {
            tracker.horizon_days
            for tracker in trackers.values()
        }

        if len(horizons) > 1:
            raise ValueError(
                "mixed tracker horizons; pass horizon_days explicitly"
            )

        selected_horizon = next(iter(horizons), None)

    else:
        if (
            isinstance(horizon_days, bool)
            or not isinstance(horizon_days, int)
            or horizon_days < 1
        ):
            raise ValueError("horizon_days must be an integer >= 1")

        selected_horizon = horizon_days

    selected = {
        name: tracker
        for name, tracker in trackers.items()
        if (
            selected_horizon is None
            or tracker.horizon_days == selected_horizon
        )
    }

    raw = {
        name: tracker.raw_weight()
        for name, tracker in selected.items()
    }

    total = sum(raw.values())

    if total <= 0:
        return {
            name: 0.0
            for name in selected
        }

    return {
        name: value / total
        for name, value in raw.items()
    }


def se_raw_weights(
    trackers: Mapping[str, ContributionTracker],
    z: float,
    lag_cap: int = 21,
) -> Dict[str, float]:
    """Diagnostic only. This function does not grant allocation authority."""
    out = {}

    for name, tracker in trackers.items():
        stats = tracker.se_stats(z, lag_cap)

        out[name] = (
            max(0.0, stats["se_lcb"])
            if stats["valid"]
            else 0.0
        )

    return out


def se_weights(
    trackers: Mapping[str, ContributionTracker],
    z: float,
    lag_cap: int = 21,
    c_model: float = 0.40,
    c_family: float = 0.50,
    families: Optional[Mapping[str, str]] = None,
    *,
    ledger=None,
) -> Dict[str, float]:
    """Certificate-gated Phase 2 weights; absent authority means cash."""
    if not (
        0 < c_model <= 1
        and 0 < c_family <= 1
    ):
        raise ValueError("caps must be in (0, 1]")

    if ledger is None:
        return {
            name: 0.0
            for name in trackers
        }

    authorized = {}

    for name, tracker in trackers.items():
        try:
            authorized[name] = bool(
                ledger.is_promoted(
                    name,
                    tracker.horizon_days,
                )
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            authorized[name] = False

    raw_all = se_raw_weights(trackers, z, lag_cap)

    raw = {
        name: (
            value
            if authorized.get(name, False)
            else 0.0
        )
        for name, value in raw_all.items()
    }

    total = sum(raw.values())

    if total <= 0:
        return {
            name: 0.0
            for name in trackers
        }

    fam = dict(
        families
        or {
            name: name
            for name in trackers
        }
    )

    if not set(trackers).issubset(fam):
        raise ValueError("family mapping missing tracker names")

    w = {
        name: value / total
        for name, value in raw.items()
    }

    for _ in range(10):
        changed = False

        for name in w:
            if w[name] > c_model:
                w[name] = c_model
                changed = True

        family_totals: Dict[str, float] = {}

        for name, value in w.items():
            family = fam[name]
            family_totals[family] = (
                family_totals.get(family, 0.0) + value
            )

        for family, amount in family_totals.items():
            if amount > c_family:
                scale = c_family / amount

                for name in w:
                    if fam[name] == family:
                        w[name] *= scale

                changed = True

        if not changed:
            break

        free = 1.0 - sum(w.values())

        eligible = [
            name
            for name in w
            if (
                authorized.get(name, False)
                and w[name] < c_model - 1e-12
                and family_totals.get(fam[name], 0.0)
                < c_family - 1e-12
            )
        ]

        raw_sum = sum(
            raw[name]
            for name in eligible
        )

        if free > 1e-12 and raw_sum > 0:
            for name in eligible:
                w[name] += free * raw[name] / raw_sum
        else:
            break

    return w


def authority_report(
    trackers: Mapping[str, ContributionTracker],
    *,
    horizon_days: Optional[int] = None,
) -> dict:
    selected_weights = weights(
        trackers,
        horizon_days=horizon_days,
    )

    selected = {
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
                iter({
                    tracker.horizon_days
                    for tracker in selected.values()
                }),
                None,
            )
        ),
        "weight_convention": (
            "max(0, mean - kappa*std), normalized over ACTIVE"
        ),
        "models": {
            name: {
                **tracker.snapshot(),
                "weight": selected_weights[name],
            }
            for name, tracker in selected.items()
        },
        "eligible_count": sum(
            1
            for tracker in selected.values()
            if tracker.eligible
        ),
        "total_weight": sum(selected_weights.values()),
    }
