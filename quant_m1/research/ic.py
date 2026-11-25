"""Information Coefficient measurement with dependence-aware inference.

Rank IC per observation, Newey-West HAC standard error on the mean, and an
AR(1) effective-sample-size correction. Overlapping holding periods produce
serially dependent IC observations; treating them as independent inflates
the t-statistic. This module reports both the naive and corrected figures so
the inflation is visible rather than hidden.

UNITS: signals and forward_returns are dimensionless fractional returns over
the SAME horizon, aligned by observation and asset. No dollar amounts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import rankdata


class ICError(Exception):
    pass


def ic_series(signals, forward_returns) -> np.ndarray:
    """Cross-sectional Spearman rank IC for each observation.

    Args:
        signals: (n_obs, n_assets) array of forecasts.
        forward_returns: (n_obs, n_assets) array of realised forward returns.

    Returns:
        (n_obs,) array of rank correlations. Observations with zero variance
        in either input yield NaN and are dropped by measure_ic.
    """
    s = np.asarray(signals, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if s.ndim != 2 or r.ndim != 2:
        raise ICError("inputs must be 2D (n_obs, n_assets)")
    if s.shape != r.shape:
        raise ICError(f"shape mismatch {s.shape} vs {r.shape}")
    if s.shape[1] < 2:
        raise ICError("need at least 2 assets for cross-sectional IC")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(r)):
        raise ICError("inputs must be finite")

    out = np.empty(s.shape[0], dtype=float)
    for t in range(s.shape[0]):
        a, b = s[t], r[t]
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            out[t] = np.nan
            continue
        ra, rb = rankdata(a), rankdata(b)
        out[t] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the sample mean (Bartlett kernel)."""
    n = len(x)
    if n < 2:
        raise ICError("need n>=2 for a standard error")
    if lags < 0 or lags >= n:
        raise ICError("lags must be in [0, n-1]")
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    total = gamma0
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        gl = float(e[l:] @ e[:-l]) / n
        total += 2.0 * w * gl
    if total <= 0.0:
        total = gamma0
    return math.sqrt(total / n)


def ar1_rho(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a series, clipped to (-0.999, 0.999)."""
    n = len(x)
    if n < 3:
        return 0.0
    e = x - x.mean()
    denom = float(e @ e)
    if denom <= 0.0:
        return 0.0
    rho = float(e[1:] @ e[:-1]) / denom
    return max(-0.999, min(0.999, rho))


def effective_n(n: int, rho: float) -> float:
    """AR(1) effective sample size: n * (1-rho)/(1+rho)."""
    if n < 1:
        return 0.0
    rho = max(-0.999, min(0.999, rho))
    return float(n * (1.0 - rho) / (1.0 + rho))


@dataclass(frozen=True)
class ICReport:
    horizon: str
    n_obs: int
    n_assets: int
    n_dropped: int
    mean_ic: float
    naive_se: float
    hac_se: float
    lags: int
    t_naive: float
    t_hac: float
    rho1: float
    eff_n: float
    evidence_label: str

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "n_dropped": self.n_dropped,
            "mean_ic": self.mean_ic,
            "naive_se": self.naive_se,
            "hac_se": self.hac_se,
            "lags": self.lags,
            "t_naive": self.t_naive,
            "t_hac": self.t_hac,
            "rho1": self.rho1,
            "eff_n": self.eff_n,
            "evidence_label": self.evidence_label,
        }


def measure_ic(signals, forward_returns, *, horizon: str,
               lags: Optional[int] = None,
               evidence_label: str = "UNLABELLED") -> ICReport:
    """Measure rank IC with dependence-aware inference.

    evidence_label MUST state the provenance of the inputs, e.g.
    "SYNTHETIC - NOT AN EDGE CLAIM" or "PIT vendor X vintage Y".
    A report without an honest label is not usable evidence.
    """
    if not horizon or not horizon.strip():
        raise ICError("horizon is required")
    if not evidence_label or not evidence_label.strip():
        raise ICError("evidence_label is required")

    raw = ic_series(signals, forward_returns)
    n_dropped = int(np.sum(~np.isfinite(raw)))
    x = raw[np.isfinite(raw)]
    n = len(x)
    if n < 3:
        raise ICError("need at least 3 usable observations")

    if lags is None:
        lags = min(int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), n - 1)
    lags = max(0, min(lags, n - 1))

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    naive = sd / math.sqrt(n) if sd > 0 else 0.0
    hac = newey_west_se(x, lags) if sd > 0 else 0.0
    rho = ar1_rho(x)

    return ICReport(
        horizon=horizon,
        n_obs=n,
        n_assets=int(np.asarray(signals).shape[1]),
        n_dropped=n_dropped,
        mean_ic=mean,
        naive_se=naive,
        hac_se=hac,
        lags=lags,
        t_naive=(mean / naive) if naive > 0 else 0.0,
        t_hac=(mean / hac) if hac > 0 else 0.0,
        rho1=rho,
        eff_n=effective_n(n, rho),
        evidence_label=evidence_label,
    )


def synthetic_ic_fixture(n_obs: int, n_assets: int, true_ic: float,
                         seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SYNTHETIC fixture with a known signal-to-return relationship.

    FOR ESTIMATOR VALIDATION ONLY. This is not market data and must never be
    cited as evidence of an edge.
    """
    if n_obs < 3 or n_assets < 2:
        raise ICError("need n_obs>=3 and n_assets>=2")
    if not -1.0 <= true_ic <= 1.0:
        raise ICError("true_ic must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n_obs, n_assets))
    noise = rng.standard_normal((n_obs, n_assets))
    w = math.sqrt(max(0.0, min(1.0, true_ic ** 2)))
    r = w * s + math.sqrt(1.0 - w ** 2) * noise
    return s, r