"""Mean-variance allocation with long-only, no-leverage, and a no-trade band.

Objective (maximise):
    mu'w - (lambda/2) w'Cov w
subject to:
    sum(w) <= 1      (no leverage, unlevered long-only)
    w >= 0           (long-only floor)

Transaction cost is NOT placed inside the smooth optimizer (L1 is non-smooth
for SLSQP). Instead a no-trade band is applied after solving: if the optimal
weights are within band_frac of the current weights, the book is left alone.

The covariance matrix is an INPUT. Use portfolio.covariance shrinkage upstream.
This module does not estimate covariance.

UNITS: expected_returns and cov are per the SAME horizon. weights are fractions
of NAV, cash_frac = 1 - invested_frac. No dollar amounts here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


class AllocationError(Exception):
    pass


@dataclass
class AllocationResult:
    weights: np.ndarray
    invested_frac: float
    cash_frac: float
    expected_return: float
    variance: float
    turnover: float
    band_applied: bool
    solver_ok: bool

    def as_dict(self) -> dict:
        return {
            "weights": [float(x) for x in self.weights],
            "invested_frac": self.invested_frac,
            "cash_frac": self.cash_frac,
            "expected_return": self.expected_return,
            "variance": self.variance,
            "turnover": self.turnover,
            "band_applied": self.band_applied,
            "solver_ok": self.solver_ok,
        }


def solve_allocation(expected_returns, cov, risk_aversion: float = 1.0,
                     current_weights=None, band_frac: float = 0.0,
                     tol: float = 1e-10, max_iter: int = 500) -> AllocationResult:
    if minimize is None:
        raise AllocationError("scipy is required for allocation")
    mu = np.asarray(expected_returns, dtype=float)
    C = np.asarray(cov, dtype=float)
    if mu.ndim != 1 or mu.size < 1:
        raise AllocationError("expected_returns must be a non-empty 1D array")
    n = mu.size
    if C.shape != (n, n):
        raise AllocationError(f"cov must be ({n},{n}), got {C.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(C)):
        raise AllocationError("inputs must be finite")
    if risk_aversion <= 0:
        raise AllocationError("risk_aversion must be > 0")
    if band_frac < 0:
        raise AllocationError("band_frac must be >= 0")

    C = 0.5 * (C + C.T)
    min_eig = float(np.linalg.eigvalsh(C).min())
    if min_eig < 0.0:
        C = C + (-min_eig) * np.eye(n)

    w0 = None
    if current_weights is not None:
        w0 = np.asarray(current_weights, dtype=float)
        if w0.shape != (n,):
            raise AllocationError(f"current_weights must be ({n},), got {w0.shape}")

    def obj(w):
        return float(-mu @ w + 0.5 * risk_aversion * (w @ C @ w))

    def grad(w):
        return -mu + risk_aversion * (C @ w)

    x0 = np.full(n, 1.0 / n)
    cons = [{"type": "ineq",
             "fun": lambda w: 1.0 - float(np.sum(w)),
             "jac": lambda w: -np.ones(n)}]
    bounds = [(0.0, None)] * n

    res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iter, "ftol": tol})
    w = np.clip(res.x, 0.0, None)
    if w.sum() > 1.0:
        w = w / w.sum()

    band_applied = False
    if w0 is not None and band_frac > 0.0:
        if float(np.max(np.abs(w - w0))) < band_frac:
            w = np.clip(w0, 0.0, None)
            band_applied = True

    ref = w0 if w0 is not None else np.zeros(n)
    invested = float(w.sum())
    return AllocationResult(
        weights=w,
        invested_frac=invested,
        cash_frac=1.0 - invested,
        expected_return=float(mu @ w),
        variance=float(w @ C @ w),
        turnover=float(np.abs(w - ref).sum()),
        band_applied=band_applied,
        solver_ok=bool(res.success),
    )
