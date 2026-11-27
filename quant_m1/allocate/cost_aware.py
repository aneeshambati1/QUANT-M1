"""Adapter: cost-aware optimizer under the decision-cycle allocator contract.

The cycle calls allocator(expected_returns, cov, risk_aversion=...,
current_weights=..., band_frac=...) and requires .solver_ok / .weights.
solve_cost_aware instead requires current_weights and per-asset cost_bps and
band_frac vectors. This adapter bridges the two WITHOUT changing either
verified module:

  * scalar cost_bps (default 5, the runner's execution cost) -> per-asset vector
  * scalar band_frac -> per-asset vector; band_frac <= 0 -> None (no band)
  * current_weights None -> zeros (cash start), matching engine semantics
  * max_gross is FIXED at 1.0; leverage can never be authorised
  * infeasible current_weights fail closed (OptimizationError) rather than
    being silently repaired; the cycle catches this and produces no allocation

Research integration only. Wiring this into the live runner is a separate,
owner-approved decision gated on the pre-registered ablation
(evidence/batchA_cost_aware_protocol.md). Nothing here grants model authority.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .optimize import OptimizationError, solve_cost_aware, CostAwareResult

Number = Union[int, float]


def _as_vector(value, n: int, name: str, *, allow_none: bool = False) -> Optional[np.ndarray]:
    if value is None:
        if allow_none:
            return None
        raise OptimizationError(f"{name} must not be None")
    if isinstance(value, bool):
        raise OptimizationError(f"{name} must be numeric, not bool")
    if isinstance(value, (int, float)):
        if not np.isfinite(value) or value < 0:
            raise OptimizationError(f"{name} must be finite and >= 0")
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise OptimizationError(f"{name} must be a finite non-negative vector of length {n}")
    return arr


def cost_aware_allocator(expected_returns, cov, *, risk_aversion: float = 1.0,
                         current_weights=None, band_frac=0.0,
                         cost_bps: Union[Number, Sequence[float]] = 5) -> CostAwareResult:
    """Cycle-compatible cost-aware allocation. Long-only, gross <= 1, fail-closed."""
    mu = np.asarray(expected_returns, dtype=float)
    if mu.ndim != 1 or mu.size < 1:
        raise OptimizationError("expected_returns must be a non-empty 1D array")
    n = mu.size
    w0 = np.zeros(n) if current_weights is None else np.asarray(current_weights, dtype=float)
    c = _as_vector(cost_bps, n, "cost_bps")
    b = _as_vector(band_frac, n, "band_frac", allow_none=True)
    if b is not None and not np.any(b > 0):
        b = None
    return solve_cost_aware(mu, cov, w0, c, risk_aversion=risk_aversion,
                            band_frac=b, max_gross=1.0)
