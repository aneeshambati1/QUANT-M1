"""Cost-aware convex allocation with solver validation.

Improves on engine.solve_allocation by putting transaction cost INSIDE the
objective via trade variables t_i >= |w_i - w0_i|:

    minimize   -mu'w + (lambda/2) w'Cov w + sum(c_i * t_i)
    subject to sum(w) <= max_gross, w >= 0, t >= |w - w0|

Convex and smooth. The optimizer sees friction and declines trades whose
cost exceeds their benefit. A per-asset no-trade band is applied afterward.

FAIL-CLOSED VALIDATION: holding current weights is always feasible. If the
solver returns an objective worse than holding, or violates constraints,
the result is marked solver_ok=False and current weights are returned.

MANDATE: max_gross is a CEILING in (0, 1]; it is not an investment target and
can never authorise leverage. cash_frac is always 1 - invested_frac.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


class OptimizationError(Exception):
    pass


@dataclass(frozen=True)
class CostAwareResult:
    weights: np.ndarray
    trades: np.ndarray
    invested_frac: float
    cash_frac: float
    expected_return: float
    variance: float
    cost_frac: float
    turnover: float
    objective: float
    hold_objective: float
    solver_ok: bool
    band_applied: bool

    def as_dict(self) -> dict:
        return {
            "weights": [float(x) for x in self.weights],
            "trades": [float(x) for x in self.trades],
            "invested_frac": self.invested_frac,
            "cash_frac": self.cash_frac,
            "expected_return": self.expected_return,
            "variance": self.variance,
            "cost_frac": self.cost_frac,
            "turnover": self.turnover,
            "objective": self.objective,
            "hold_objective": self.hold_objective,
            "solver_ok": self.solver_ok,
            "band_applied": self.band_applied,
        }


def solve_cost_aware(expected_returns, cov, current_weights, cost_bps,
                     *, risk_aversion: float = 1.0, band_frac=None,
                     max_gross: float = 1.0, tol: float = 1e-8,
                     max_iter: int = 1000) -> CostAwareResult:
    if minimize is None:
        raise OptimizationError("scipy required")
    mu = np.asarray(expected_returns, dtype=float)
    C = np.asarray(cov, dtype=float)
    w0 = np.asarray(current_weights, dtype=float)
    c = np.asarray(cost_bps, dtype=float) / 10_000.0
    n = mu.size
    if mu.ndim != 1 or n < 1:
        raise OptimizationError("expected_returns must be non-empty 1D")
    if C.shape != (n, n) or w0.shape != (n,) or c.shape != (n,):
        raise OptimizationError("shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in (mu, C, w0, c)):
        raise OptimizationError("inputs must be finite")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise OptimizationError("risk_aversion must be finite and > 0")
    if not np.isfinite(max_gross) or max_gross <= 0:
        raise OptimizationError("max_gross must be finite and > 0")
    if max_gross > 1.0:
        raise OptimizationError("max_gross must be <= 1; leverage is prohibited")
    if np.any(c < 0):
        raise OptimizationError("cost_bps must be >= 0")
    if np.any(w0 < -tol) or w0.sum() > max_gross + tol:
        raise OptimizationError("current_weights must be feasible")

    C = 0.5 * (C + C.T)
    me = float(np.linalg.eigvalsh(C).min())
    if me < 0.0:
        C = C + (-me) * np.eye(n)

    def obj(x):
        w, t = x[:n], x[n:]
        return float(-mu @ w + 0.5 * risk_aversion * (w @ C @ w) + c @ t)

    def grad(x):
        w = x[:n]
        return np.concatenate([-mu + risk_aversion * (C @ w), c])

    I = np.eye(n)
    cons = [
        {"type": "ineq", "fun": lambda x: max_gross - x[:n].sum(),
         "jac": lambda x: np.concatenate([-np.ones(n), np.zeros(n)])},
        {"type": "ineq", "fun": lambda x: x[n:] - x[:n] + w0,
         "jac": lambda x: np.hstack([-I, I])},
        {"type": "ineq", "fun": lambda x: x[n:] + x[:n] - w0,
         "jac": lambda x: np.hstack([I, I])},
    ]
    bounds = [(0.0, None)] * n + [(0.0, None)] * n
    x0 = np.concatenate([np.clip(w0, 0, None), np.zeros(n)])

    res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iter, "ftol": 1e-12})

    hold_obj = float(-mu @ w0 + 0.5 * risk_aversion * (w0 @ C @ w0))
    w = np.clip(res.x[:n], 0.0, None)
    trades = w - w0
    feasible = (w.sum() <= max_gross + 1e-6 and np.all(w >= -1e-6)
                and np.all(np.isfinite(w)))
    improved = obj(np.concatenate([w, np.abs(trades)])) <= hold_obj + 1e-7
    ok = bool(res.success) and feasible and improved

    if not ok:
        w = np.clip(w0, 0.0, None)
        trades = w - w0

    band_applied = False
    if band_frac is not None:
        b = np.asarray(band_frac, dtype=float)
        if b.shape != (n,) or np.any(b < 0) or not np.all(np.isfinite(b)):
            raise OptimizationError("band_frac must be finite non-negative 1D of length n")
        snapped = np.where(np.abs(trades) < b, w0, w)
        if snapped.sum() <= max_gross + 1e-6:
            ct = snapped - w0
            if obj(np.concatenate([snapped, np.abs(ct)])) <= hold_obj + 1e-7:
                w = snapped
                trades = ct
                band_applied = True

    final_obj = obj(np.concatenate([w, np.abs(trades)]))
    if final_obj > hold_obj + 1e-7:
        w = np.clip(w0, 0.0, None)
        trades = w - w0
        band_applied = False
        ok = False
        final_obj = obj(np.concatenate([w, np.abs(trades)]))

    invested = float(w.sum())
    turnover = float(np.abs(trades).sum())
    return CostAwareResult(
        weights=w, trades=trades, invested_frac=invested,
        cash_frac=1.0 - invested,
        expected_return=float(mu @ w), variance=float(w @ C @ w),
        cost_frac=float(c @ np.abs(trades)), turnover=turnover,
        objective=final_obj,
        hold_objective=hold_obj, solver_ok=ok, band_applied=band_applied)
