"""
Deflated Sharpe Ratio.

Two-term Bailey / López de Prado expected maximum Sharpe.
Single-term approximation is forbidden.
"""
from __future__ import annotations

import math
from typing import Dict

from scipy import stats

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sigma_sr: float = 1.0) -> float:
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    if sigma_sr <= 0:
        raise ValueError("sigma_sr > 0")
    if n_trials == 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(
    sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis_raw: float = 3.0,
) -> float:
    """
    sr and sr_benchmark at OBSERVATION frequency.
    kurtosis_raw is RAW kurtosis, normal = 3.
    """
    if n_obs < 3:
        raise ValueError("need >= 3 observations")
    if kurtosis_raw < 1:
        raise ValueError("raw kurtosis >= 1")
    denom = 1.0 - skew * sr + ((kurtosis_raw - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    sr_ann: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis_raw: float = 3.0,
    periods_per_year: int = 252,
    sigma_sr: float = 1.0,
    n_eff: int | None = None,
) -> Dict[str, float]:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year > 0")
    n = n_eff if n_eff is not None else n_obs
    sr_d = sr_ann / math.sqrt(periods_per_year)
    sr_star_d = expected_max_sharpe(n_trials, sigma_sr) / math.sqrt(periods_per_year)
    dsr = probabilistic_sharpe(sr_d, sr_star_d, n, skew, kurtosis_raw)
    return {
        "dsr": dsr,
        "sr_obs_per_period": sr_d,
        "sr_hurdle_per_period": sr_star_d,
        "sr_hurdle_annualised": sr_star_d * math.sqrt(periods_per_year),
        "n_obs": n_obs,
        "n_eff": n,
        "n_trials": n_trials,
        "sigma_sr": sigma_sr,
        "periods_per_year": periods_per_year,
    }
