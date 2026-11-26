"""Pre-Registered Case Specialization (PFCS) - governance scaffolding.

STATUS: DISABLED BY DEFAULT. This module provides the infrastructure for
case-specialized policies but will NOT activate any case deviation until the
pre-registered data and power gates are satisfied. On the current panel
(~16 independent volatility regimes, 10 correlated ETFs, non-PIT), multi-case
specialization is NOT IDENTIFIABLE and this module will correctly refuse to
activate.

DESIGN PRINCIPLES:
  1. Case definitions are FROZEN before outcomes via cryptographic digest.
  2. Each case has a pre-registered minimum independent observation count.
  3. Case effects are estimated via hierarchical partial pooling (James-Stein /
     empirical Bayes) with shrinkage toward the pooled prior.
  4. The family-wide multiple-testing budget includes ALL registered cases.
  5. Cases that fail their power gate remain at the pooled policy.
  6. Cases may be RETIRED back to pooled when evidence decays.
  7. This module never modifies the Risk Kernel, execution path, or mandate.

WHAT THIS IS NOT: an adaptive regime classifier. Case membership is determined
by a pre-registered observable function applied to PIT data. The function itself
is frozen; only its output varies with market state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from ..core import sha


@dataclass(frozen=True)
class CaseDefinition:
    """A single pre-registered market case. Frozen before outcomes."""
    case_id: str
    description: str
    observable_fn_id: str
    min_independent_obs: int
    min_effect_size: float
    created_at: str

    def __post_init__(self):
        if not self.case_id or not self.observable_fn_id:
            raise ValueError("case_id and observable_fn_id required")
        if self.min_independent_obs < 1:
            raise ValueError("min_independent_obs must be >= 1")
        if not math.isfinite(self.min_effect_size) or self.min_effect_size <= 0:
            raise ValueError("min_effect_size must be positive and finite")

    def digest(self) -> str:
        return sha({
            "case_id": self.case_id, "observable_fn_id": self.observable_fn_id,
            "min_independent_obs": self.min_independent_obs,
            "min_effect_size": self.min_effect_size,
            "created_at": self.created_at,
        })


@dataclass(frozen=True)
class CaseRegistry:
    """Immutable registry of pre-registered cases. All cases count in the family."""
    cases: Tuple[CaseDefinition, ...] = ()
    family_alpha: float = 0.05
    registry_digest: str = ""

    def __post_init__(self):
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case_ids in registry")
        if not (0 < self.family_alpha < 1):
            raise ValueError("family_alpha must be in (0,1)")

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def bonferroni_threshold(self) -> float:
        from scipy.stats import norm
        return float(norm.ppf(1.0 - self.family_alpha / (2 * max(self.n_cases, 1))))

    def get_case(self, case_id: str) -> CaseDefinition:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        raise KeyError(f"unknown case_id: {case_id}")

    def compute_registry_digest(self) -> str:
        return sha({
            "cases": [c.digest() for c in self.cases],
            "family_alpha": self.family_alpha,
        })


@dataclass
class CaseStats:
    """Accumulated statistics for one case. Updated externally."""
    n_independent_obs: int = 0
    mean_effect: float = 0.0
    var_effect: float = 0.0
    pooled_mean: float = 0.0
    pooled_var: float = 0.0


def power_gate(stats: CaseStats, case_def: CaseDefinition,
               family_threshold_z: float) -> Tuple[bool, str]:
    """Check whether a case has enough evidence to deviate from pooled."""
    if stats.n_independent_obs < case_def.min_independent_obs:
        return False, f"INSUFFICIENT_OBS:{stats.n_independent_obs}/{case_def.min_independent_obs}"
    if stats.var_effect <= 0 or stats.n_independent_obs < 2:
        return False, "INSUFFICIENT_VARIANCE_ESTIMATE"
    se = math.sqrt(stats.var_effect / stats.n_independent_obs)
    z = abs(stats.mean_effect) / se if se > 1e-15 else 0.0
    if z < family_threshold_z:
        return False, f"BELOW_FAMILY_THRESHOLD:z={z:.3f}<z_crit={family_threshold_z:.3f}"
    if abs(stats.mean_effect) < case_def.min_effect_size:
        return False, f"BELOW_MIN_EFFECT:{abs(stats.mean_effect):.6f}<{case_def.min_effect_size}"
    return True, "PASSED"


def hierarchical_shrinkage(case_stats: Mapping[str, CaseStats],
                           tau_sq: float) -> Dict[str, float]:
    """James-Stein shrinkage: theta_c = w*theta_c_hat + (1-w)*theta_pooled."""
    if tau_sq < 0:
        raise ValueError("tau_sq must be >= 0")
    result = {}
    for cid, st in case_stats.items():
        sigma_sq = st.var_effect if st.var_effect > 0 else float("inf")
        if not math.isfinite(sigma_sq) or sigma_sq <= 0:
            result[cid] = st.pooled_mean
            continue
        w = tau_sq / (tau_sq + sigma_sq)
        result[cid] = w * st.mean_effect + (1.0 - w) * st.pooled_mean
    return result


def classify_episode(observable_fn: Callable, market_state,
                     registry: CaseRegistry) -> Optional[str]:
    """Apply pre-registered observable function. Returns case_id or None."""
    return observable_fn(market_state)


def build_case_policy_weights(
    registry: CaseRegistry,
    case_stats: Mapping[str, CaseStats],
    active_case: Optional[str],
    tau_sq: float,
    pooled_weights: np.ndarray,
    case_weight_deltas: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Construct weights with case specialization. Falls back to pooled on any failure."""
    diag = {}
    if active_case is None:
        diag["_active"] = "POOLED_DEFAULT"
        return pooled_weights.copy(), diag
    try:
        case_def = registry.get_case(active_case)
    except KeyError:
        diag["_active"] = f"UNKNOWN_CASE:{active_case}"
        return pooled_weights.copy(), diag
    stats = case_stats.get(active_case)
    if stats is None:
        diag[active_case] = "NO_STATS"
        return pooled_weights.copy(), diag
    may_activate, reason = power_gate(stats, case_def, registry.bonferroni_threshold)
    diag[active_case] = reason
    if not may_activate:
        return pooled_weights.copy(), diag
    shrunk = hierarchical_shrinkage(case_stats, tau_sq)
    shrunk_effect = shrunk.get(active_case, 0.0)
    if abs(shrunk_effect) < case_def.min_effect_size:
        diag[active_case] = f"SHRUNK_BELOW_MIN:{abs(shrunk_effect):.6f}"
        return pooled_weights.copy(), diag
    delta = case_weight_deltas.get(active_case)
    if delta is None or delta.shape != pooled_weights.shape:
        diag[active_case] = "MISSING_OR_SHAPE_MISMATCH_DELTA"
        return pooled_weights.copy(), diag
    adjusted = pooled_weights + delta
    adjusted = np.clip(adjusted, 0.0, None)
    total = adjusted.sum()
    if total > 1.0 + 1e-9:
        adjusted = adjusted / total
    elif total <= 0:
        diag[active_case] = "ADJUSTED_WEIGHTS_INFEASIBLE"
        return pooled_weights.copy(), diag
    diag[active_case] = "ACTIVATED"
    return adjusted, diag
