"""
Probability of Backtest Overfitting.

PBO = fraction of CSCV splits where the IS-best config ranks below OOS median.
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
from scipy import stats

from .cscv import cscv_splits


def _sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 0.0
    s = r.std(ddof=1)
    return float(r.mean() / s * math.sqrt(252)) if s > 1e-12 else 0.0


def pbo_from_score_matrix(is_scores: np.ndarray, oos_scores: np.ndarray) -> Dict[str, float]:
    IS = np.asarray(is_scores, dtype=float)
    OOS = np.asarray(oos_scores, dtype=float)
    if IS.shape != OOS.shape or IS.ndim != 2:
        raise ValueError("IS/OOS must be same 2-D shape")
    S, C = IS.shape
    if C < 2:
        raise ValueError("need >= 2 configs")

    lam = []
    omega = []
    for s in range(S):
        nstar = int(np.argmax(IS[s]))
        rank = stats.rankdata(OOS[s], method="average")[nstar]
        w = rank / (C + 1.0)
        w = min(max(w, 1e-12), 1.0 - 1e-12)
        lam.append(math.log(w / (1.0 - w)))
        omega.append(w)

    lam = np.asarray(lam)
    return {
        "pbo": float((lam < 0).mean()),
        "lambda_mean": float(lam.mean()),
        "omega_mean": float(np.mean(omega)),
        "n_splits": S,
        "n_configs": C,
    }


def pbo_cscv(
    returns_by_config: Dict[str, np.ndarray],
    n_groups: int = 10,
    label_horizon: int = 1,
    embargo: int = 0,
    score_fn: Callable[[np.ndarray], float] | None = None,
) -> Dict[str, float]:
    if n_groups % 2 != 0:
        raise ValueError("n_groups must be even for symmetric CSCV")
    if len(returns_by_config) < 2:
        raise ValueError("need >= 2 configs")

    score_fn = score_fn or _sharpe
    names = list(returns_by_config)
    n_obs = len(returns_by_config[names[0]])
    if any(len(returns_by_config[n]) != n_obs for n in names):
        raise ValueError("all config return series must have same length")

    n_test = n_groups // 2
    IS, OOS = [], []
    for train_idx, test_idx in cscv_splits(n_obs, n_groups, n_test, label_horizon, embargo):
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        IS.append([score_fn(returns_by_config[n][train_idx]) for n in names])
        OOS.append([score_fn(returns_by_config[n][test_idx]) for n in names])

    if not IS:
        raise ValueError("no valid CSCV splits")
    return pbo_from_score_matrix(np.asarray(IS), np.asarray(OOS))
